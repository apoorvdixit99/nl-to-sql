"""Adversarial tests for the deterministic security layer. No LLM, no database writes."""
import pytest

from nl_to_sql import config
from nl_to_sql.security import SQLGuard

TABLES = {
    "Artist": ["ArtistId", "Name"],
    "Album": ["AlbumId", "Title", "ArtistId"],
    "Invoice": ["InvoiceId", "CustomerId", "Total", "InvoiceDate"],
    "Customer": ["CustomerId", "FirstName", "LastName", "Email", "Country"],
    "Users": ["UserId", "Login", "PasswordHash", "ApiKey"],
}


@pytest.fixture
def guard():
    return SQLGuard(TABLES, "sqlite")


BLOCKED = [
    # stacked queries / piggybacking
    ("SELECT * FROM Artist; DROP TABLE Artist", "multiple statements"),
    ("SELECT 1; DELETE FROM Invoice WHERE 1=1", "multiple statements"),
    # DDL and admin commands
    ("DROP TABLE Artist", "schema changes"),
    ("ALTER TABLE Artist ADD COLUMN x INT", "schema changes"),
    ("CREATE TABLE stolen AS SELECT * FROM Customer", "schema changes"),
    ("ATTACH DATABASE '/tmp/evil.db' AS evil", "ATTACH"),
    ("PRAGMA writable_schema = 1", "admin command"),
    ("VACUUM INTO '/tmp/copy.db'", "VACUUM INTO"),
    # system catalogs / tables outside the schema
    ("SELECT sql FROM sqlite_master", "not available"),
    ("SELECT name FROM Artist UNION SELECT sql FROM sqlite_schema", "not available"),
    ("SELECT table_name FROM information_schema.tables", "system catalog"),
    ("SELECT usename, passwd FROM pg_shadow", "not available"),
    ("SELECT user, authentication_string FROM mysql.user", "system catalog"),
    # sensitive columns
    ("SELECT Login, PasswordHash FROM Users", "sensitive"),
    ("SELECT * FROM Users", "sensitive"),
    ("SELECT u.ApiKey FROM Users u JOIN Customer c ON c.CustomerId = u.UserId", "sensitive"),
    # dangerous functions
    ("SELECT load_extension('/tmp/evil.so')", "load_extension"),
    ("SELECT readfile('/etc/passwd')", "readfile"),
    ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
    ("SELECT pg_sleep(10)", "pg_sleep"),
    ("SELECT CASE WHEN 1=1 THEN sleep(5) END FROM Artist", "sleep"),
    ("SELECT benchmark(10000000, md5('x'))", "benchmark"),
    ("SELECT load_file('/etc/passwd')", "load_file"),
    # injection shapes
    ("SELECT * FROM Customer WHERE Country = 'x' OR 1=1", "always-true"),
    ("SELECT * FROM Customer WHERE Email = '' OR 'a'='a'", "always-true"),
    ("SELECT * FROM Customer WHERE CustomerId = 5 OR TRUE", "always-true"),
    ("SELECT Name FROM Artist INTO OUTFILE '/tmp/x.txt'", "OUTFILE"),
    # mass changes
    ("DELETE FROM Invoice", "without a WHERE"),
    ("UPDATE Customer SET Email = 'x'", "without a WHERE"),
    # unparseable non-queries
    ("EXEC xp_cmdshell 'dir'", None),
]

ALLOWED = [
    "SELECT Name FROM Artist ORDER BY Name LIMIT 10",
    "SELECT a.Name, COUNT(*) FROM Artist a JOIN Album al ON al.ArtistId = a.ArtistId GROUP BY a.ArtistId",
    "WITH t AS (SELECT CustomerId, SUM(Total) AS s FROM Invoice GROUP BY 1) SELECT * FROM t WHERE s > 40",
    "SELECT FirstName FROM Customer WHERE Email = 'john; DROP TABLE x --@example.com'",  # data, not code
    "SELECT * FROM Customer WHERE Country = 'USA' OR Country = 'Canada'",
    "UPDATE Customer SET Email = 'new@example.com' WHERE CustomerId = 3",  # write: allowed, needs approval
    "SELECT UserId, Login FROM Users",  # non-sensitive columns of a table with sensitive ones
    "PRAGMA table_info(Artist)",
]


@pytest.mark.parametrize("sql,reason", BLOCKED)
def test_blocked(guard, sql, reason):
    v = guard.check(sql)
    assert not v.allowed, sql
    if reason:
        assert reason.lower() in v.message().lower(), (sql, v.violations)


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(guard, sql):
    v = guard.check(sql)
    assert v.allowed, (sql, v.violations)


def test_write_kind_reported(guard):
    assert guard.check("UPDATE Customer SET Email = 'a' WHERE CustomerId = 1").kind == "write"
    assert guard.check("SELECT 1 FROM Artist").kind == "read"


def test_warnings_dont_block(guard):
    v = guard.check("SELECT Name FROM Artist -- just a comment")
    assert v.allowed and any("comment" in w for w in v.warnings)
    v = guard.check("SELECT * FROM Artist, Album")
    assert v.allowed and any("cartesian" in w for w in v.warnings)


def test_hidden_tables_and_config(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_UNBOUNDED_WRITES", True)
    assert SQLGuard(TABLES).check("DELETE FROM Invoice").allowed
    monkeypatch.setattr(config, "ALLOW_DDL", True)
    assert SQLGuard(TABLES).check("CREATE TABLE t2 (a INT)").allowed


def test_unparseable_select_only_runs_with_text_checks(guard):
    tricky = "SELECT AVG(Total) FROM (SELECT Total FROM Invoice ORDER BY Total LIMIT 2 - (SELECT COUNT(*) FROM Invoice) % 2)"
    assert guard.check(tricky).allowed
    evil = "SELECT readfile('/etc/passwd') FROM Invoice LIMIT 2 - (SELECT COUNT(*) FROM Invoice) % 2"
    assert not guard.check(evil).allowed


def test_timeout_stops_runaway_query(tmp_path, monkeypatch):
    import shutil
    from nl_to_sql.db import Database
    monkeypatch.setattr(config, "QUERY_TIMEOUT_S", 0.5)
    path = tmp_path / "c.db"
    shutil.copy(config.ROOT / "data" / "chinook.db", path)
    monkeypatch.setitem(config.DATABASES, "t", f"sqlite:///{path}")
    db = Database("t")
    with pytest.raises(TimeoutError):
        db.execute_readonly("SELECT COUNT(*) FROM Track a, Track b, Track c")
    assert db.execute_readonly("SELECT COUNT(*) FROM Track").rows == [(3503,)]


def test_tautological_where_on_writes_is_blocked(guard):
    for sql in ("DELETE FROM Invoice WHERE 1 = 1", "UPDATE Customer SET Email = 'x' WHERE TRUE",
                "DELETE FROM Invoice WHERE (1)"):
        v = guard.check(sql)
        assert not v.allowed and "always-true WHERE" in v.message(), sql


def test_count_star_is_not_select_star(guard):
    assert guard.check("SELECT COUNT(*) FROM Users WHERE Login = 'a'").allowed
    assert not guard.check("SELECT u.* FROM Users u").allowed


def test_blast_radius_blocks_disguised_mass_writes(tmp_path, monkeypatch):
    import shutil
    from nl_to_sql.db import Database
    from nl_to_sql.executor import Executor
    path = tmp_path / "c.db"
    shutil.copy(config.ROOT / "data" / "chinook.db", path)
    monkeypatch.setitem(config.DATABASES, "br", f"sqlite:///{path}")
    ex = Executor(Database("br"))
    out = ex.run("DELETE FROM Invoice WHERE InvoiceId > 0")      # syntax looks bounded; data says otherwise
    assert out.status == "blocked" and "412 of 412" in out.message
    out = ex.run("DELETE FROM Invoice WHERE InvoiceId = 7")
    assert out.status == "needs_approval" and out.affected == 1
