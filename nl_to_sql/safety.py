"""Classify SQL as read / write / unknown by parsing it with sqlglot (not regex)."""
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

QueryKind = Literal["read", "write", "unknown"]

# Any of these nodes anywhere in the tree makes the statement a write,
# which also catches data-modifying CTEs like `WITH x AS (DELETE ...) SELECT ...`.
_WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,
    exp.Create, exp.Drop, exp.Alter, exp.TruncateTable,
)

# sqlglot dialect names keyed by SQLAlchemy dialect names where they differ.
_DIALECTS = {"postgresql": "postgres", "mssql": "tsql"}


def to_sqlglot_dialect(sqlalchemy_dialect: str) -> str:
    return _DIALECTS.get(sqlalchemy_dialect, sqlalchemy_dialect)


def classify(sql: str, dialect: str = "sqlite") -> QueryKind:
    """Return the most dangerous kind across all statements in `sql`."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError:
        return "unknown"
    if not statements:
        return "unknown"

    kinds = set()
    for stmt in statements:
        if any(stmt.find_all(*_WRITE_NODES)):
            kinds.add("write")
        elif isinstance(stmt, exp.Query):  # SELECT, UNION, WITH ... SELECT
            kinds.add("read")
        else:  # PRAGMA, ATTACH, SET, GRANT, raw Commands, ...
            kinds.add("unknown")

    for kind in ("write", "unknown"):
        if kind in kinds:
            return kind
    return "read"
