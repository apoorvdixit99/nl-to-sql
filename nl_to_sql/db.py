import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text

from . import config


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    rowcount: int = 0          # rows affected, for writes
    truncated: bool = False    # True if more than MAX_RESULT_ROWS were available


class Database:
    """Thin wrapper over SQLAlchemy + LangChain's SQLDatabase (used for schema extraction)."""

    def __init__(self, name: str):
        if name not in config.DATABASES:
            raise ValueError(f"Unknown database '{name}'. Options: {', '.join(config.DATABASES)}")
        self.name = name
        self.engine = create_engine(config.DATABASES[name])
        self._lc_db = SQLDatabase(self.engine, sample_rows_in_table_info=config.SAMPLE_ROWS)

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    def table_names(self) -> list[str]:
        return sorted(self._lc_db.get_usable_table_names())

    def schema(self, tables: list[str] | None = None) -> str:
        """CREATE TABLE statements plus a few sample rows per table."""
        return self._lc_db.get_table_info(tables)

    def compact_schema(self, hints: bool = False) -> str:
        """One line per table: columns with types, PK and FK markers. Much cheaper than schema().

        hints=True appends value hints from profile(): {'a','b'} for low-cardinality text
        columns, [min .. max] for date columns, so the model gets formats/codes right.
        """
        insp = inspect(self.engine)
        profile = self.profile() if hints else {}
        lines = []
        for table in self.table_names():
            pk = set(insp.get_pk_constraint(table).get("constrained_columns") or [])
            fks = {}
            for fk in insp.get_foreign_keys(table):
                for col, ref in zip(fk["constrained_columns"], fk["referred_columns"]):
                    fks[col] = f"{fk['referred_table']}.{ref}"
            cols = []
            for c in insp.get_columns(table):
                desc = f"{c['name']} {c['type']}"
                if c["name"] in pk:
                    desc += " PK"
                if c["name"] in fks:
                    desc += f" ->{fks[c['name']]}"
                hint = profile.get(table, {}).get(c["name"])
                if hint:
                    desc += f" {hint}"
                cols.append(desc)
            lines.append(f"{table}({', '.join(cols)})")
        return "\n".join(lines)

    # ---- value profiling (free: plain SQL, no LLM) ---------------------------

    def profile(self) -> dict[str, dict[str, str]]:
        """{table: {column: hint}}. Cached next to SQLite files, keyed by file mtime/size."""
        cache = self._profile_cache_path()
        if cache and cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("_stamp") == self._file_stamp():
                return data["profile"]
        prof = self._build_profile()
        if cache:
            cache.write_text(json.dumps({"_stamp": self._file_stamp(), "profile": prof},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
        return prof

    def _sqlite_path(self) -> Path | None:
        db = self.engine.url.database
        return Path(db) if self.dialect == "sqlite" and db and db != ":memory:" else None

    def _profile_cache_path(self) -> Path | None:
        p = self._sqlite_path()
        return p.with_suffix(".profile.json") if p else None

    def _file_stamp(self) -> str:
        st = self._sqlite_path().stat()
        return f"{st.st_mtime_ns}:{st.st_size}"

    def _build_profile(self) -> dict[str, dict[str, str]]:
        insp = inspect(self.engine)
        q = self.engine.dialect.identifier_preparer.quote
        maxv = config.HINT_MAX_VALUES
        prof: dict[str, dict[str, str]] = {}
        with self.engine.connect() as conn:
            for table in self.table_names():
                t = q(table)
                nrows = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                label_done = False  # first text column of a small lookup table, e.g. CategoryName
                for c in insp.get_columns(table):
                    col, typ = q(c["name"]), str(c["type"]).upper()
                    if "BLOB" in typ or "BINARY" in typ:
                        continue
                    hint = None
                    if "DATE" in typ or "TIME" in typ:
                        lo, hi = conn.execute(text(f"SELECT MIN({col}), MAX({col}) FROM {t}")).one()
                        if lo is not None:
                            has_null = conn.execute(text(
                                f"SELECT 1 FROM {t} WHERE {col} IS NULL LIMIT 1")).first()
                            hint = f"[{lo} .. {hi}{', has NULLs' if has_null else ''}]"
                    elif any(k in typ for k in ("CHAR", "TEXT", "CLOB", "STRING")) or not typ:
                        vals = conn.execute(text(
                            f"SELECT DISTINCT {col} FROM {t} WHERE {col} IS NOT NULL LIMIT {maxv + 1}"
                        )).scalars().all()
                        # Categorical = few values relative to rows. Lookup-table labels count too;
                        # unique-per-row columns (phones, addresses) are skipped as noise.
                        nonnull = conn.execute(text(f"SELECT COUNT({col}) FROM {t}")).scalar()
                        categorical = len(vals) * 2 <= nonnull
                        lookup_label = nrows <= maxv and not label_done and c["name"] not in (
                            insp.get_pk_constraint(table).get("constrained_columns") or [])
                        if (0 < len(vals) <= maxv and all(len(str(v)) <= 30 for v in vals)
                                and (categorical or lookup_label)):
                            hint = "{" + ", ".join(repr(v) for v in sorted(vals, key=str)) + "}"
                        label_done = label_done or lookup_label
                    if hint:
                        prof.setdefault(table, {})[c["name"]] = hint
        return prof

    def execute(self, sql: str, commit: bool) -> QueryResult:
        """Run `sql`. Reads are always rolled back, so nothing persists unless commit=True."""
        with self.engine.connect() as conn:
            cursor = conn.execute(text(sql))
            result = QueryResult(rowcount=cursor.rowcount)
            if cursor.returns_rows:
                result.columns = list(cursor.keys())
                fetched = cursor.fetchmany(config.MAX_RESULT_ROWS + 1)
                result.truncated = len(fetched) > config.MAX_RESULT_ROWS
                result.rows = [tuple(r) for r in fetched[: config.MAX_RESULT_ROWS]]
            if commit:
                conn.commit()
            else:
                conn.rollback()
            return result
