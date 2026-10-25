"""Deterministic security layer (v7). Pure rules, no LLM: the same SQL always gets the same verdict.

Every statement goes through SQLGuard.check() before it reaches the database, whichever
agent architecture produced it. It sits in front of the other defenses:

    SQLGuard (this file)  ->  permission policy (auto/confirm/deny)  ->  approval (interrupt)
    ->  read-only connection for reads  ->  statement timeout + row cap

Hard violations block the statement and the reason goes back to the model so it can
rewrite it. Warnings let it run but are passed on to the validator agent.

What it catches:
- stacked queries (`SELECT 1; DROP TABLE x`), over-long SQL
- DDL / admin commands (DROP, ALTER, TRUNCATE, GRANT, ATTACH, PRAGMA writes, VACUUM, COPY, LOAD DATA)
- tables outside the agent's schema (system catalogs like sqlite_master, pg_shadow, mysql.user,
  information_schema, and any HIDDEN_TABLES)
- sensitive columns (password, ssn, token, ...) by name, including through SELECT *
- dangerous functions: file access, extensions, sleep/benchmark (time-based injection), dblink
- classic injection shapes: always-true OR conditions, UNION-appended reads of other tables
  (blocked by the table allowlist), INTO OUTFILE/DUMPFILE
- UPDATE/DELETE with no WHERE clause (mass changes)
"""
import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from . import config
from .safety import QueryKind, classify

# Functions that touch the file system, load code, sleep, or reach other servers.
BLOCKED_FUNCTIONS = {
    # SQLite
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zipfile", "sqlar_compress",
    # Postgres
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "lo_import", "lo_export",
    "lo_get", "lo_put", "dblink", "dblink_exec", "dblink_connect", "pg_sleep", "pg_sleep_for",
    "pg_sleep_until", "pg_terminate_backend", "pg_cancel_backend", "set_config", "pg_reload_conf",
    "pg_rotate_logfile", "pg_advisory_lock", "nextval", "setval", "query_to_xml", "copy",
    # MySQL
    "load_file", "sleep", "benchmark", "sys_exec", "sys_eval", "get_lock",
    # SQL Server
    "xp_cmdshell", "openrowset", "opendatasource", "waitfor",
}

# Text-level patterns for things sqlglot may not turn into nodes (or can't parse at all).
TEXT_PATTERNS = [
    (re.compile(r"\binto\s+(out|dump)file\b", re.I), "writes a file (INTO OUTFILE/DUMPFILE)"),
    (re.compile(r"\bload\s+data\b", re.I), "LOAD DATA reads server files"),
    (re.compile(r"\bcopy\b[\s\S]*\b(program|to|from)\b\s*'", re.I), "COPY reads/writes server files"),
    (re.compile(r"\battach\s+(database\s+)?'", re.I), "ATTACH opens another database file"),
    (re.compile(r"\bvacuum\s+into\b", re.I), "VACUUM INTO writes a database file"),
    (re.compile(r"\bwaitfor\s+delay\b", re.I), "WAITFOR DELAY (time-based injection)"),
    (re.compile(r"\bxp_\w+", re.I), "extended stored procedure"),
]

# Read-only PRAGMAs are fine for introspection; everything else is an admin command.
SAFE_PRAGMAS = re.compile(r"^\s*pragma\s+(table_info|table_xinfo|index_list|index_info|foreign_key_list)\s*\(",
                          re.I)

DDL_NODES = (exp.Create, exp.Drop, exp.Alter, exp.TruncateTable, exp.Grant, exp.Revoke)


@dataclass
class SecurityVerdict:
    allowed: bool
    kind: QueryKind
    violations: list[str] = field(default_factory=list)  # blocking
    warnings: list[str] = field(default_factory=list)    # allowed, but worth a second look

    def message(self) -> str:
        return "; ".join(self.violations or self.warnings)


class SQLGuard:
    def __init__(self, tables: dict[str, list[str]], dialect: str = "sqlite"):
        """tables: {table: [columns]} the agent may use (already excludes HIDDEN_TABLES)."""
        self.dialect = dialect
        self.tables = {t.lower(): [c.lower() for c in cols] for t, cols in tables.items()}
        self.sensitive = [re.compile(p, re.I) for p in config.SENSITIVE_COLUMN_PATTERNS]

    def _is_sensitive(self, column: str) -> bool:
        return any(p.search(column) for p in self.sensitive)

    def check(self, sql: str) -> SecurityVerdict:
        v = SecurityVerdict(allowed=True, kind=classify(sql, self.dialect))
        text = _strip_strings(sql)

        if len(sql) > config.MAX_SQL_CHARS:
            v.violations.append(f"SQL is longer than {config.MAX_SQL_CHARS} characters")
        for pattern, why in TEXT_PATTERNS:
            if pattern.search(text):
                v.violations.append(why)
        if re.search(r"--|/\*", text):
            v.warnings.append("SQL contains a comment, which can hide part of a statement")
        if re.search(r"\b0x[0-9a-f]{8,}\b|\bchar\s*\(\s*\d+\s*(,\s*\d+\s*){3,}\)", text, re.I):
            v.warnings.append("encoded literal (hex/CHAR codes), a common obfuscation trick")

        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except ParseError:
            statements = None
        if statements is None:
            # Can't build a tree. Only query-shaped text may continue (it will run read-only).
            if not sql.lstrip().lower().startswith(("select", "with")):
                v.violations.append("could not parse the statement")
            else:
                v.warnings.append("could not fully parse the query; it will only run read-only")
                v.violations += self._text_only_checks(text)
            return self._finish(v)

        if len(statements) > 1:
            v.violations.append("multiple statements in one call (stacked queries) are not allowed")
        for stmt in statements:
            v.violations += self._check_statement(stmt, sql, v)
        return self._finish(v)

    def _finish(self, v: SecurityVerdict) -> SecurityVerdict:
        v.violations = list(dict.fromkeys(v.violations))  # dedupe, keep order
        v.allowed = not v.violations
        return v

    def _check_statement(self, stmt: exp.Expression, raw: str, v: SecurityVerdict) -> list[str]:
        out = []
        # Admin commands / DDL.
        if isinstance(stmt, exp.Command) or isinstance(stmt, exp.Pragma):
            if not SAFE_PRAGMAS.match(raw):
                out.append(f"admin command '{stmt.sql(dialect=self.dialect)[:40]}' is not allowed")
            return out
        if any(stmt.find_all(*DDL_NODES)) and not config.ALLOW_DDL:
            out.append("schema changes (CREATE/DROP/ALTER/TRUNCATE/GRANT) are not allowed")

        # Tables must be ones the agent was given. Blocks catalogs and hidden tables.
        ctes = {c.alias.lower() for c in stmt.find_all(exp.CTE)}
        # The table a CREATE makes doesn't exist yet (only reachable when ALLOW_DDL is on).
        created = {c.this.find(exp.Table).name.lower() for c in stmt.find_all(exp.Create)
                   if c.this is not None and c.this.find(exp.Table) is not None}
        used = []
        for t in stmt.find_all(exp.Table):
            name = t.name.lower()
            if not name or name in ctes or name in created:
                continue
            if t.args.get("db") or t.args.get("catalog"):
                schema_name = (t.args.get("db") or t.args.get("catalog")).name.lower()
                if schema_name in ("information_schema", "pg_catalog", "mysql", "sys", "performance_schema"):
                    out.append(f"system catalog '{schema_name}.{t.name}' is not allowed")
                    continue
            if name not in self.tables:
                out.append(f"table '{t.name}' is not available")
            else:
                used.append(name)

        # Sensitive columns, by name or through SELECT *.
        for col in stmt.find_all(exp.Column):
            if col.name and self._is_sensitive(col.name):
                out.append(f"column '{col.name}' is sensitive and can't be queried")
        if any(isinstance(s, exp.Star) for s in stmt.find_all(exp.Star)):
            for t in used:
                hidden = [c for c in self.tables[t] if self._is_sensitive(c)]
                if hidden:
                    out.append(f"SELECT * would expose sensitive columns of '{t}' ({', '.join(hidden)}); "
                               "list the columns you need")

        # Dangerous functions.
        for fn in stmt.find_all(exp.Func):
            name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
            if name in BLOCKED_FUNCTIONS:
                out.append(f"function {name}() is not allowed")

        # Always-true conditions: `... OR 1=1`, `OR 'a'='a'`, `OR TRUE`.
        for node in stmt.find_all(exp.Or):
            if _tautology(node.left) or _tautology(node.right):
                out.append("always-true OR condition (classic SQL injection pattern)")
                break
        for where in stmt.find_all(exp.Where):
            if _tautology(where.this):
                v.warnings.append("WHERE condition is always true")

        # Mass changes.
        for node in stmt.find_all(exp.Update, exp.Delete):
            if not node.args.get("where") and not config.ALLOW_UNBOUNDED_WRITES:
                out.append(f"{node.key.upper()} without a WHERE clause would change every row")

        # Cartesian products: comma joins or JOINs with no condition, and no WHERE at all.
        for select in stmt.find_all(exp.Select):
            joins = select.args.get("joins") or []
            # (SQLite's parser turns `FROM a, b` into a CROSS join, so CROSS counts as loose too.)
            loose = [j for j in joins if not j.args.get("on") and not j.args.get("using")]
            if loose and not select.args.get("where"):
                v.warnings.append("join without a condition (possible cartesian product)")
        return out

    def _text_only_checks(self, text: str) -> list[str]:
        out = []
        for word in re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", text, re.I):
            if word.lower() in BLOCKED_FUNCTIONS:
                out.append(f"function {word.lower()}() is not allowed")
        for word in re.findall(r"\b(?:from|join)\s+[\"`\[]?([a-z_][\w.]*)", text, re.I):
            if word.lower().split(".")[-1] not in self.tables:
                out.append(f"table '{word}' is not available")
        if re.search(r"\bor\s+(1\s*=\s*1|'(\w*)'\s*=\s*'\2'|true)\b", text, re.I):
            out.append("always-true OR condition (classic SQL injection pattern)")
        return out


def _strip_strings(sql: str) -> str:
    """Blank out quoted string literals so text patterns don't fire on data values."""
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def _tautology(node: exp.Expression | None) -> bool:
    if node is None:
        return False
    node = node.unnest() if isinstance(node, exp.Paren) else node
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Literal) and node.is_number:
        return float(node.this) != 0
    if isinstance(node, (exp.EQ, exp.GTE, exp.LTE)):
        left, right = node.left, node.right
        if isinstance(left, exp.Literal) and isinstance(right, exp.Literal):
            return left.this == right.this
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            return left.sql() == right.sql()  # x = x
    return False
