import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import MetaData, Table, create_engine, inspect, text
from sqlalchemy.schema import CreateTable

from . import config


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    rowcount: int = 0          # rows affected, for writes
    truncated: bool = False    # True if more than MAX_RESULT_ROWS were available


class ReadOnlyViolation(Exception):
    """A statement tried to write inside a read-only connection."""


class Database:
    """Thin wrapper over a SQLAlchemy engine: schema text for prompts, profiling, execution."""

    def __init__(self, name: str):
        if name not in config.DATABASES:
            raise ValueError(f"Unknown database '{name}'. Options: {', '.join(config.DATABASES)}")
        self.name = name
        self.engine = create_engine(config.DATABASES[name])
        self._compact: dict[bool, dict[str, str]] = {}

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    def table_names(self) -> list[str]:
        hidden = {t.lower() for t in config.HIDDEN_TABLES}
        return sorted(t for t in inspect(self.engine).get_table_names() if t.lower() not in hidden)

    def columns(self) -> dict[str, list[str]]:
        """{table: [column names]} for every visible table (cached). Feeds the security layer."""
        if not hasattr(self, "_columns"):
            insp = inspect(self.engine)
            self._columns = {t: [c["name"] for c in insp.get_columns(t)] for t in self.table_names()}
        return self._columns

    def schema(self, tables: list[str] | None = None) -> str:
        """CREATE TABLE statements plus a few sample rows per table (only the tables asked for)."""
        known = set(self.table_names())
        tables = tables or sorted(known)
        missing = [t for t in tables if t not in known]
        if missing:
            raise ValueError(f"Unknown table(s): {', '.join(missing)}")
        q = self.engine.dialect.identifier_preparer.quote
        parts = []
        with self.engine.connect() as conn:
            for t in tables:
                table = Table(t, MetaData(), autoload_with=conn)
                ddl = str(CreateTable(table).compile(self.engine)).strip()
                res = conn.execute(text(f"SELECT * FROM {q(t)} LIMIT {config.SAMPLE_ROWS}"))
                cols = list(res.keys())
                rows = ["\t".join(str(v)[:100] for v in r) for r in res]
                sample = f"/*\n{len(rows)} rows from {t} table:\n" + "\t".join(cols) + "\n" + "\n".join(rows) + "\n*/"
                parts.append(f"{ddl}\n\n{sample}")
        return "\n\n".join(parts)

    def compact_lines(self, hints: bool = False) -> dict[str, str]:
        """{table: one-line summary} with columns, types, PK and FK markers (cached).

        hints=True appends value hints from profile(): {'a','b'} for low-cardinality text
        columns, [min .. max] for date columns, so the model gets formats/codes right.
        """
        if hints in self._compact:
            return self._compact[hints]
        insp = inspect(self.engine)
        profile = self.profile() if hints else {}
        lines = {}
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
            lines[table] = f"{table}({', '.join(cols)})"
        self._compact[hints] = lines
        return lines

    def compact_schema(self, hints: bool = False, tables: list[str] | None = None) -> str:
        lines = self.compact_lines(hints)
        return "\n".join(lines[t] for t in (tables or lines))

    def foreign_keys(self) -> dict[str, set[str]]:
        """Undirected FK adjacency: {table: {tables it references or is referenced by}}."""
        insp = inspect(self.engine)
        graph: dict[str, set[str]] = {t: set() for t in self.table_names()}
        for t in graph:
            for fk in insp.get_foreign_keys(t):
                ref = fk["referred_table"]
                if ref in graph and ref != t:
                    graph[t].add(ref)
                    graph[ref].add(t)
        return graph

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

    def execute_readonly(self, sql: str) -> QueryResult:
        """Run `sql` where the database itself refuses writes (defense in depth on top of
        the sqlglot classifier). Raises ReadOnlyViolation if the statement tried to write."""
        try:
            if self.dialect == "sqlite" and self._sqlite_path():
                if not hasattr(self, "_ro_engine"):
                    uri = f"sqlite:///file:{self._sqlite_path().as_posix()}?mode=ro&uri=true"
                    self._ro_engine = create_engine(uri)
                return self._run(self._ro_engine, sql, commit=False)
            if self.dialect in ("postgresql", "mysql", "mariadb"):
                return self._run(self.engine, sql, commit=False, setup="SET TRANSACTION READ ONLY")
            # Other dialects: no read-only switch wired up yet; rely on classifier + rollback.
            return self._run(self.engine, sql, commit=False)
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ("readonly", "read-only", "read only")):
                raise ReadOnlyViolation(str(e)) from e
            raise

    def execute(self, sql: str, commit: bool) -> QueryResult:
        """Run `sql`. Reads are always rolled back, so nothing persists unless commit=True."""
        return self._run(self.engine, sql, commit)

    @staticmethod
    def _run(engine, sql: str, commit: bool, setup: str | None = None) -> QueryResult:
        with engine.connect() as conn:
            if setup:
                conn.execute(text(setup))  # e.g. SET TRANSACTION READ ONLY: must come first
            clear = _apply_timeout(conn, engine.dialect.name)
            try:
                cursor = conn.execute(text(sql))
            except Exception as e:
                if "interrupted" in str(e).lower():
                    raise TimeoutError(f"query timed out after {config.QUERY_TIMEOUT_S}s") from e
                raise
            finally:
                clear()
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


def _apply_timeout(conn, dialect: str):
    """Cap statement run time. Returns a function that removes the cap."""
    limit = config.QUERY_TIMEOUT_S
    if not limit:
        return lambda: None
    if dialect == "sqlite":
        raw = conn.connection.driver_connection
        deadline = time.monotonic() + limit
        raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        return lambda: raw.set_progress_handler(None, 0)
    if dialect == "postgresql":
        conn.execute(text(f"SET LOCAL statement_timeout = {int(limit * 1000)}"))
    elif dialect in ("mysql", "mariadb"):
        conn.execute(text(f"SET SESSION max_execution_time = {int(limit * 1000)}"))
    return lambda: None
