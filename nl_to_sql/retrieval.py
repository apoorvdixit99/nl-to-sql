"""Schema retrieval for large databases. No LLM calls.

1. Rank tables two ways (v7 "hybrid", the default):
   - BM25 over one document per table: name (weighted), columns, value hints, FK targets,
     and descriptions when available,
   - vector search (local fastembed model) over table documents AND column documents,
     so "how long is a song" can find Track.Milliseconds via its description.
   Rankings are merged with Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank)).
2. Take the top-k tables, then expand along foreign keys:
   - neighbours of the strongest hits (join partners like InvoiceLine for Invoice/Track),
   - "bridge" tables adjacent to two or more selected tables (many-to-many links).
3. Cap the total so the prompt stays small.
"""
import math
import re
from collections import Counter
from pathlib import Path

from . import config
from .db import Database

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "by", "with", "is", "are",
         "was", "were", "how", "many", "much", "what", "which", "who", "whose", "each", "per",
         "give", "show", "list", "their", "that", "have", "has", "from", "at", "it", "its", "do",
         "does", "did", "be", "this", "most", "top", "more", "than", "all", "any", "name", "names"}


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    words = re.split(r"[^A-Za-z0-9]+", _CAMEL.sub(" ", text))
    return [_stem(w.lower()) for w in words if w and w.lower() not in _STOP and not w.isdigit()]


class TableIndex:
    def __init__(self, db: Database, k1: float = 1.2, b: float = 0.75):
        self.db = db
        self.lines = db.compact_lines(hints=config.VALUE_HINTS)
        self.graph = db.foreign_keys()
        profile = db.profile() if config.VALUE_HINTS else {}
        self.docs: dict[str, list[str]] = {}
        for table, line in self.lines.items():
            name_tokens = tokenize(table)
            # name counts 3x: a table called "Invoice" should beat one with an InvoiceId column
            tokens = name_tokens * 3 + tokenize(line.split("(", 1)[1])
            for hint in profile.get(table, {}).values():
                tokens += tokenize(hint)
            self.docs[table] = tokens
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs.values())) / max(len(self.docs), 1)
        df = Counter(t for toks in self.docs.values() for t in set(toks))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = {table: Counter(toks) for table, toks in self.docs.items()}

    # ---- BM25 ------------------------------------------------------------------

    def scores(self, query: str) -> list[tuple[str, float]]:
        q = tokenize(query)
        out = []
        for table, tf in self.tf.items():
            dl = len(self.docs[table])
            s = 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                out.append((table, s))
        return sorted(out, key=lambda x: -x[1])

    # ---- vectors (v7) ---------------------------------------------------------

    def _build_vectors(self) -> None:
        """Embed one document per table and one per column; cached on disk by content hash."""
        if hasattr(self, "_tvec"):
            return
        desc = self.db.descriptions
        cols = self.db.columns()
        profile = self.db.profile() if config.VALUE_HINTS else {}
        table_docs, col_docs, self.col_keys = [], [], []
        for t in self.lines:
            d = desc.get(t, {})
            names = ", ".join(f"{c} ({d.get('columns', {})[c]})" if c in d.get("columns", {}) else c
                              for c in cols.get(t, []))
            table_docs.append(f"{_words(t)}: {d.get('description', '')} Columns: {names}")
            for c in cols.get(t, []):
                hint = profile.get(t, {}).get(c, "")
                col_docs.append(f"{_words(t)} {_words(c)}: {d.get('columns', {}).get(c, '')} {hint}".strip())
                self.col_keys.append((t, c))
        self._tvec = embed_cached(self.db, "tables", table_docs)
        self._cvec = embed_cached(self.db, "columns", col_docs)
        self._tables = list(self.lines)

    def vector_rankings(self, query: str) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """(tables by table-doc similarity, tables by best column similarity, top columns)."""
        self._build_vectors()
        q = embed([query])[0]
        tsim = self._tvec @ q
        csim = self._cvec @ q
        by_table = [self._tables[i] for i in tsim.argsort()[::-1]]
        best: dict[str, float] = {}
        for (t, _), sim in zip(self.col_keys, csim):
            best[t] = max(best.get(t, -1.0), float(sim))
        by_column = sorted(best, key=lambda t: -best[t])
        top_cols = [self.col_keys[i] for i in csim.argsort()[::-1][:20]]
        return by_table, by_column, top_cols

    # ---- fused search ------------------------------------------------------------

    def ranked(self, query: str) -> list[str]:
        mode = config.RETRIEVAL
        rankings = []  # (weight, ranking)
        if mode in ("bm25", "hybrid"):
            rankings.append((config.BM25_WEIGHT, [t for t, _ in self.scores(query)]))
        if mode in ("vector", "hybrid") and _have_embedder():
            by_table, by_column, _ = self.vector_rankings(query)
            rankings += [(config.VECTOR_WEIGHT, by_table), (config.VECTOR_WEIGHT, by_column)]
        if len(rankings) == 1:
            return rankings[0][1]
        fused: dict[str, float] = {}
        for weight, ranking in rankings:
            for rank, t in enumerate(ranking):
                fused[t] = fused.get(t, 0.0) + weight / (config.RRF_K + rank + 1)
        return sorted(fused, key=lambda t: -fused[t])

    def search(self, query: str, k: int | None = None) -> list[str]:
        return self.ranked(query)[: k or config.RETRIEVE_TOP_K]

    def relevant_columns(self, query: str, tables: list[str], k: int = 12) -> list[str]:
        """Columns of `tables` most similar to the query, as 'Table.Column' (schema agent hint)."""
        if not _have_embedder():
            return []
        _, _, top = self.vector_rankings(query)
        keep = set(tables)
        return [f"{t}.{c}" for t, c in top if t in keep][:k]

    def select(self, question: str) -> list[str]:
        """Top-k hits plus FK join partners, capped at RETRIEVE_MAX_TABLES."""
        hits = self.search(question)
        chosen = list(hits)
        cap = config.RETRIEVE_MAX_TABLES

        def add(t: str):
            if t not in chosen and len(chosen) < cap:
                chosen.append(t)

        # Bridges first: tables linking two or more hits are almost always needed for joins.
        candidates = Counter(n for h in hits for n in self.graph.get(h, ()) if n not in hits)
        for t, links in candidates.most_common():
            if links >= 2:
                add(t)
        # Then direct neighbours of the strongest hits.
        for h in hits[:3]:
            for n in sorted(self.graph.get(h, ())):
                add(n)
        return chosen


# ---- embeddings (local, cached) ---------------------------------------------------

_EMBEDDER = None


def _have_embedder() -> bool:
    try:
        import fastembed  # noqa: F401  (availability check)
        return True
    except ImportError:
        return False


def embed(texts: list[str]):
    """L2-normalised embeddings, shape (n, dim). Loads the model once per process."""
    global _EMBEDDER
    import numpy as np
    if _EMBEDDER is None:
        from fastembed import TextEmbedding
        _EMBEDDER = TextEmbedding(config.EMBED_MODEL, cache_dir=str(config.ROOT / ".cache" / "fastembed"))
    vecs = np.array(list(_EMBEDDER.embed(texts)), dtype="float32")
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


def embed_cached(db: Database, kind: str, texts: list[str]):
    import hashlib

    import numpy as np
    key = hashlib.sha1((config.EMBED_MODEL + "\x00".join(texts)).encode()).hexdigest()[:16]
    p = db._sqlite_path()
    base = p.with_suffix("") if p else config.ROOT / "data" / "vectors" / db.name
    path = Path(f"{base}.{kind}.{key}.npy")
    if path.exists():
        return np.load(path)
    vecs = embed(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def _words(identifier: str) -> str:
    """music_InvoiceLine -> 'music invoice line' (embedding models read words, not identifiers)."""
    return " ".join(w.lower() for w in re.split(r"[^A-Za-z0-9]+", _CAMEL.sub(" ", identifier)) if w)
