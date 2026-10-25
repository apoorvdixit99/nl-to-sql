"""Query execution shared by both architectures (single agent and multi-agent).

    SQL -> SQLGuard (deterministic security) -> read-only run for reads
        -> permission policy -> EXPLAIN check for writes -> [approval] -> run + commit

The approval step itself lives in the caller (LangGraph interrupt), so this module stays
free of graph details.
"""
from dataclasses import dataclass, field

from . import config
from .db import Database, QueryResult, ReadOnlyViolation
from .safety import to_sqlglot_dialect
from .security import SQLGuard


@dataclass
class Outcome:
    status: str                  # ok | blocked | denied | error | needs_approval
    kind: str = "unknown"
    result: QueryResult | None = None
    message: str = ""            # what the model is told
    security: list[str] = field(default_factory=list)
    affected: int | None = None  # rows an UPDATE/DELETE would change (shown at approval)


class Executor:
    def __init__(self, db: Database, permissions: dict[str, str] | None = None):
        self.db = db
        self.permissions = permissions or config.PERMISSIONS
        self.guard = SQLGuard(db.columns(), to_sqlglot_dialect(db.dialect))

    def run(self, sql: str) -> Outcome:
        """Everything up to (not including) approval. Reads finish here."""
        verdict = self.guard.check(sql)
        if not verdict.allowed:
            return Outcome("blocked", verdict.kind, security=[f"blocked: {verdict.message()}"],
                           message=(f"Blocked by the security policy, query not run: {verdict.message()}.\n"
                                    "Rewrite the query to avoid this."))
        security = [f"warning: {w}" for w in verdict.warnings]
        kind = verdict.kind
        if kind == "read" and self.permissions.get("read", "auto") != "auto":
            # Unusual policy (reads need approval or are denied): don't run anything yet.
            if self.permissions["read"] == "deny":
                return Outcome("denied", kind, message="Blocked: read queries are not allowed.", security=security)
            return Outcome("needs_approval", kind, security=security)
        # Reads run on a read-only connection, so a misclassified write still can't change data.
        # Query-shaped SQL the parser couldn't classify gets the same treatment.
        if kind == "read" or (kind == "unknown" and sql.lstrip().lower().startswith(("select", "with"))):
            try:
                res = self.db.execute_readonly(sql)
                return Outcome("ok", "read", res, format_rows(res), security)
            except ReadOnlyViolation:
                kind = "write"  # it tried to write: fall through to the write policy
            except Exception as e:
                return Outcome("error", kind, message=f"Error: {e}\nFix the query and try again.",
                               security=security)
        policy = self.permissions.get(kind, "confirm")
        if policy == "deny":
            return Outcome("denied", kind, message=f"Blocked: {kind} queries are not allowed.", security=security)
        # Validate before asking the user, so they never approve a query that can't run.
        # EXPLAIN plans without executing on SQLite, Postgres and MySQL.
        try:
            self.db.execute(f"EXPLAIN {sql}", commit=False)
        except Exception as e:
            return Outcome("error", kind, message=f"Error (query not run): {e}\nFix the query and try again.",
                           security=security)
        # Blast radius: count what an UPDATE/DELETE would touch before anyone approves it.
        affected = self.affected_rows(sql)
        if affected:
            changed, total = affected
            if total >= config.BLAST_RADIUS_MIN_ROWS and changed > total * config.MAX_WRITE_FRACTION:
                why = (f"this would change {changed} of {total} rows (over "
                       f"{config.MAX_WRITE_FRACTION:.0%}); narrow the WHERE clause")
                return Outcome("blocked", kind, security=security + [f"blocked: {why}"],
                               message=f"Blocked by the security policy, query not run: {why}.")
            security = security + [f"info: would change {changed} of {total} rows"]
        if policy == "confirm":
            return Outcome("needs_approval", kind, security=security, affected=affected[0] if affected else None)
        return self.run_write(sql, kind, security)

    def affected_rows(self, sql: str) -> tuple[int, int] | None:
        """(rows matched, rows in table) for a single-table UPDATE/DELETE, counted read-only."""
        import sqlglot
        from sqlglot import exp
        try:
            stmt = sqlglot.parse_one(sql, read=self.guard.dialect)
        except Exception:
            return None
        if not isinstance(stmt, (exp.Update, exp.Delete)):
            return None
        table = stmt.this if isinstance(stmt, exp.Update) else stmt.find(exp.Table)
        if not isinstance(table, exp.Table):
            return None
        name = table.sql(dialect=self.guard.dialect)
        where = stmt.args.get("where")
        cond = f" WHERE {where.this.sql(dialect=self.guard.dialect)}" if where else ""
        try:
            changed = self.db.execute_readonly(f"SELECT COUNT(*) FROM {name}{cond}").rows[0][0]
            total = self.db.execute_readonly(f"SELECT COUNT(*) FROM {name}").rows[0][0]
        except Exception:
            return None
        return int(changed), int(total)

    def run_write(self, sql: str, kind: str = "write", security: list[str] | None = None) -> Outcome:
        """Run an approved statement (an approved read still runs read-only)."""
        try:
            res = self.db.execute_readonly(sql) if kind == "read" else self.db.execute(sql, commit=True)
        except Exception as e:
            return Outcome("error", kind, message=f"Error: {e}\nFix the query and try again.",
                           security=security or [])
        return Outcome("ok", kind, res, format_rows(res), security or [])


def format_rows(res: QueryResult) -> str:
    """Result as compact text for the model (first TOOL_RESULT_ROWS rows)."""
    if not res.columns:
        return f"OK, {res.rowcount} row(s) affected."
    shown = res.rows[: config.TOOL_RESULT_ROWS]
    lines = ["\t".join(res.columns)] + ["\t".join(map(str, r)) for r in shown]
    extra = len(res.rows) - len(shown)
    if extra > 0 or res.truncated:
        lines.append(f"... ({extra}{'+' if res.truncated else ''} more rows not shown)")
    return f"{len(res.rows)}{'+' if res.truncated else ''} row(s):\n" + "\n".join(lines)
