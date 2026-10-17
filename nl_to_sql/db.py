from dataclasses import dataclass, field

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

    def compact_schema(self) -> str:
        """One line per table: columns with types, PK and FK markers. Much cheaper than schema()."""
        insp = inspect(self.engine)
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
                cols.append(desc)
            lines.append(f"{table}({', '.join(cols)})")
        return "\n".join(lines)

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
