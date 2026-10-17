"""v4: schema retrieval for large databases. No LLM calls and no extra dependencies.

1. BM25 over one document per table: table name (weighted), column names, value hints, FK targets.
2. Take the top-k tables, then expand along foreign keys:
   - neighbours of the strongest hits (join partners like InvoiceLine for Invoice/Track),
   - "bridge" tables adjacent to two or more selected tables (many-to-many links).
3. Cap the total so the prompt stays small.
"""
import math
import re
from collections import Counter

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

    def search(self, query: str, k: int | None = None) -> list[str]:
        return [t for t, _ in self.scores(query)[: k or config.RETRIEVE_TOP_K]]

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
