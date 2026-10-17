"""Offline tests for the v2 agent. A scripted fake LLM stands in for Claude: no API calls."""
import shutil

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from nl_to_sql import config
from nl_to_sql.agent import Pending, SQLAgent
from nl_to_sql.db import Database
from nl_to_sql.safety import classify


class ScriptedLLM:
    """Returns pre-written AIMessages in order and records what it was sent."""

    def __init__(self, *replies: AIMessage):
        self.replies = list(replies)
        self.seen: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


_n = 0


def call(sql: str, final: bool = True, text: str = "") -> AIMessage:
    global _n
    _n += 1
    return AIMessage(content=text, tool_calls=[
        {"name": "run_query", "args": {"sql": sql, "final": final}, "id": f"t{_n}"}])


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A throwaway copy of Chinook so write tests can't touch the real file."""
    path = tmp_path / "chinook.db"
    shutil.copy(config.ROOT / "data" / "chinook.db", path)
    monkeypatch.setitem(config.DATABASES, "test", f"sqlite:///{path}")
    return Database("test")


def artist_count(db: Database) -> int:
    return db.execute("SELECT count(*) FROM Artist", commit=False).rows[0][0]


def test_single_final_read_is_one_call(db):
    llm = ScriptedLLM(call("SELECT Name FROM Artist ORDER BY ArtistId LIMIT 2", text="Assuming ID order."))
    a = SQLAgent(db, llm=llm).start("first two artists")
    assert a.status == "ok" and a.llm_calls == 1
    assert a.result.rows == [("AC/DC",), ("Accept",)]
    assert a.note == "Assuming ID order."


def test_error_is_fed_back_and_fixed(db):
    llm = ScriptedLLM(call("SELECT Nme FROM Artist"), call("SELECT Name FROM Artist LIMIT 1"))
    a = SQLAgent(db, llm=llm).start("an artist")
    assert a.status == "ok" and a.llm_calls == 2
    tool_msg = llm.seen[1][-1]
    assert isinstance(tool_msg, ToolMessage) and "no such column" in tool_msg.content


def test_exploratory_query_then_final(db):
    llm = ScriptedLLM(call("SELECT min(InvoiceDate) FROM Invoice", final=False),
                      call("SELECT count(*) FROM Invoice"))
    a = SQLAgent(db, llm=llm).start("how many invoices")
    assert a.status == "ok" and a.llm_calls == 2 and a.result.rows == [(412,)]
    assert "row(s)" in llm.seen[1][-1].content  # model saw the exploratory rows


def test_empty_final_gets_one_more_look(db):
    llm = ScriptedLLM(call("SELECT * FROM Artist WHERE Name = 'Nobody'"),
                      AIMessage(content="There is no such artist."))
    a = SQLAgent(db, llm=llm).start("find Nobody")
    assert a.status == "ok" and a.llm_calls == 2
    assert a.result.rows == [] and a.note == "There is no such artist."


def test_write_declined_changes_nothing(db):
    before = artist_count(db)
    agent = SQLAgent(db, llm=ScriptedLLM(call("DELETE FROM Artist WHERE ArtistId = 275")))
    p = agent.start("delete artist 275")
    assert isinstance(p, Pending) and p.kind == "write" and "DELETE" in p.sql
    a = agent.resume(False)
    assert a.status == "declined" and artist_count(db) == before


def test_write_approved_commits(db):
    before = artist_count(db)
    agent = SQLAgent(db, llm=ScriptedLLM(call("DELETE FROM Artist WHERE ArtistId = 275")))
    assert isinstance(agent.start("delete artist 275"), Pending)
    a = agent.resume(True)
    assert a.status == "ok" and a.result.rowcount == 1 and artist_count(db) == before - 1


def test_deny_policy_blocks_without_asking(db):
    perms = {**config.PERMISSIONS, "write": "deny"}
    a = SQLAgent(db, perms, llm=ScriptedLLM(call("DROP TABLE Artist"))).start("drop it")
    assert a.status == "denied" and artist_count(db) > 0


def test_cannot_answer(db):
    llm = ScriptedLLM(AIMessage(content="CANNOT_ANSWER: no weather data."))
    a = SQLAgent(db, llm=llm).start("weather in Paris?")
    assert a.status == "cannot_answer" and a.note == "no weather data."


def test_call_budget_stops_loops(db):
    llm = ScriptedLLM(*[call("SELECT broken FROM nowhere") for _ in range(20)])
    a = SQLAgent(db, llm=llm).start("loop forever")
    assert a.status == "gave_up" and a.llm_calls == config.MAX_LLM_CALLS


def test_system_prompt_has_compact_schema(db):
    llm = ScriptedLLM(call("SELECT 1"))
    SQLAgent(db, llm=llm).start("x")
    system_text = llm.seen[0][0].content[0]["text"]
    assert "Album(AlbumId INTEGER PK" in system_text and "->Artist.ArtistId" in system_text


@pytest.mark.parametrize("sql,kind", [
    ("SELECT * FROM Artist", "read"),
    ("WITH t AS (SELECT 1) SELECT * FROM t", "read"),
    ("SELECT 1; DELETE FROM Artist", "write"),
    ("UPDATE Artist SET Name = 'x'", "write"),
    ("PRAGMA table_info(Artist)", "unknown"),
    ("SELEC oops", "unknown"),
])
def test_classify(sql, kind):
    assert classify(sql) == kind


def test_broken_write_is_rejected_before_asking_user(db):
    llm = ScriptedLLM(call("UPDATE Artistt SET Name = 'x'"), AIMessage(content="CANNOT_ANSWER: gave up"))
    out = SQLAgent(db, llm=llm).start("rename")
    assert not isinstance(out, Pending)  # never reached the approval step
    assert "query not run" in llm.seen[1][-1].content


def test_value_hints_and_few_shot_toggle(db, monkeypatch):
    monkeypatch.setattr(config, "VALUE_HINTS", True)
    monkeypatch.setattr(config, "FEW_SHOT", False)
    llm = ScriptedLLM(call("SELECT 1"))
    SQLAgent(db, llm=llm).start("x")
    text = llm.seen[0][0].content[0]["text"]
    assert "InvoiceDate DATETIME [2009-01-01" in text and "Worked examples" not in text

    monkeypatch.setattr(config, "VALUE_HINTS", False)
    llm = ScriptedLLM(call("SELECT 1"))
    SQLAgent(db, llm=llm).start("x")
    assert "[2009-01-01" not in llm.seen[0][0].content[0]["text"]


# ---- v4: retrieval on the 274-table testbed ----------------------------------

@pytest.fixture
def enterprise():
    if not (config.ROOT / "data" / "enterprise.db").exists():
        pytest.skip("run scripts/build_enterprise_db.py first")
    return Database("enterprise")


def test_auto_mode_picks_full_for_small_and_retrieve_for_large(db, enterprise):
    assert SQLAgent(db, llm=ScriptedLLM()).schema_mode == "full"
    assert SQLAgent(enterprise, llm=ScriptedLLM()).schema_mode == "retrieve"


def test_retrieved_tables_go_in_user_message(enterprise):
    llm = ScriptedLLM(call('SELECT COUNT(*) FROM "music_Track"'))
    a = SQLAgent(enterprise, llm=llm).start("Music store: How many tracks are in the Grunge playlist?")
    system_text = llm.seen[0][0].content[0]["text"]
    user_text = llm.seen[0][1].content
    assert "All table names:" in system_text and "music_Track(" not in system_text
    assert "music_PlaylistTrack(" in user_text and "music_Playlist(" in user_text
    assert a.status == "ok" and "music_PlaylistTrack" in a.context_tables
    assert len(a.context_tables) <= config.RETRIEVE_MAX_TABLES


def test_search_tables_tool(enterprise):
    search = AIMessage(content="", tool_calls=[
        {"name": "search_tables", "args": {"query": "order details discount"}, "id": "s1"}])
    llm = ScriptedLLM(search, call("SELECT 1"))
    SQLAgent(enterprise, llm=llm).start("Northwind shop: anything")
    assert '"shop_Order Details"(' in llm.seen[1][-1].content or "shop_Order Details(" in llm.seen[1][-1].content


def test_unparseable_select_runs_readonly_without_approval(db):
    median = ("SELECT AVG(Total) FROM (SELECT Total FROM Invoice ORDER BY Total "
              "LIMIT 2 - (SELECT COUNT(*) FROM Invoice) % 2 OFFSET ((SELECT COUNT(*) FROM Invoice) - 1) / 2)")
    assert classify(median) == "unknown"
    a = SQLAgent(db, llm=ScriptedLLM(call(median))).start("median invoice total")
    assert a.status == "ok" and a.kind == "read" and a.result.rows == [(3.96,)]


def test_unknown_non_query_still_needs_approval(db):
    out = SQLAgent(db, llm=ScriptedLLM(call("PRAGMA user_version = 7"))).start("set version")
    assert isinstance(out, Pending) and out.kind == "unknown"


# ---- v5: answer check and self-consistency -----------------------------------

def test_red_flag_triggers_review_and_fix(db, monkeypatch):
    monkeypatch.setattr(config, "VERIFY", "flags")
    yoy = ("WITH y AS (SELECT strftime('%Y', InvoiceDate) AS yr, SUM(Total) AS rev FROM Invoice GROUP BY yr) "
           "SELECT yr, ROUND(100.0 * (rev - LAG(rev) OVER (ORDER BY yr)) / LAG(rev) OVER (ORDER BY yr), 1) FROM y")
    fixed = f"SELECT * FROM ({yoy}) WHERE yr > '2009'"
    llm = ScriptedLLM(call(yoy), AIMessage(content="ISSUE: the first year has a NULL change; drop it."), call(fixed))
    a = SQLAgent(db, llm=llm).start("year-over-year change for each year after the first")
    assert a.status == "ok" and a.llm_calls == 3 and len(a.result.rows) == 4
    assert "NULL" in a.review
    assert "reviewer" in llm.seen[2][-1].content  # the fix request reached the agent


def test_clean_result_skips_review(db, monkeypatch):
    monkeypatch.setattr(config, "VERIFY", "flags")
    a = SQLAgent(db, llm=ScriptedLLM(call("SELECT COUNT(*) FROM Track"))).start("how many tracks")
    assert a.llm_calls == 1 and a.review == ""


def test_review_ok_keeps_answer(db, monkeypatch):
    monkeypatch.setattr(config, "VERIFY", "always")
    llm = ScriptedLLM(call("SELECT COUNT(*) FROM Track"), AIMessage(content="OK"))
    a = SQLAgent(db, llm=llm).start("how many tracks")
    assert a.status == "ok" and a.llm_calls == 2 and a.review == "OK" and a.result.rows == [(3503,)]


def test_self_consistency_majority_vote(db, monkeypatch):
    monkeypatch.setattr(config, "CANDIDATES", 3)
    monkeypatch.setattr(config, "VERIFY", "off")
    llm = ScriptedLLM(call("SELECT COUNT(*) FROM Album"),
                      call("SELECT COUNT(*) FROM Track"),
                      call("SELECT COUNT(TrackId) AS n FROM Track"))
    a = SQLAgent(db, llm=llm).start("how many tracks")
    assert a.result.rows == [(3503,)] and a.candidates_agreed == "2/3" and a.llm_calls == 3


def test_red_flags_rules():
    from nl_to_sql.db import QueryResult
    from nl_to_sql.verify import red_flags
    R = lambda cols, rows: QueryResult(columns=cols, rows=rows)
    assert red_flags("which artist has the most albums", R(["a", "n"], [("x", 3), ("y", 2)]))
    assert not red_flags("which artist has the most albums", R(["a", "n"], [("x", 3), ("y", 3)]))  # tie
    assert not red_flags("revenue for each year", R(["y", "r"], [("2009", 1.0), ("2010", 2.0)]))
    assert red_flags("x", R(["a"], [(None,)]))
    assert red_flags("percentage of sales by genre", R(["g", "p"], [("Rock", 135.0)]))


def test_prose_answer_after_exploratory_query_keeps_rows(db):
    llm = ScriptedLLM(call("SELECT COUNT(*) FROM Track", final=False),
                      AIMessage(content="There are 3503 tracks."))
    a = SQLAgent(db, llm=llm).start("how many tracks")
    assert a.status == "ok" and a.result.rows == [(3503,)] and a.sql == "SELECT COUNT(*) FROM Track"
    assert a.note == "There are 3503 tracks."


# ---- v6: memory, configuration, other dialects --------------------------------

def test_memory_passes_earlier_turn_to_follow_up(db):
    llm = ScriptedLLM(call("SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY 1"),
                      call("SELECT BillingCountry, strftime('%Y', InvoiceDate), SUM(Total) FROM Invoice GROUP BY 1, 2"))
    agent = SQLAgent(db, llm=llm, memory=True)
    agent.start("revenue by country")
    agent.start("now split that by year")
    follow_up = llm.seen[1][-1].content
    assert "Earlier in this conversation" in follow_up and "revenue by country" in follow_up
    assert "GROUP BY 1" in follow_up and follow_up.endswith("Question: now split that by year")
    agent.reset()
    assert agent.history == []


def test_no_memory_by_default(db):
    llm = ScriptedLLM(call("SELECT 1"), call("SELECT 2"))
    agent = SQLAgent(db, llm=llm)
    agent.start("a")
    agent.start("b")
    assert llm.seen[1][-1].content == "b"


def test_memory_keeps_last_n_turns(db, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_TURNS", 2)
    agent = SQLAgent(db, llm=ScriptedLLM(*[call(f"SELECT {i}") for i in range(4)]), memory=True)
    for i in range(4):
        agent.start(f"q{i}")
    assert [q for q, _, _ in agent.history] == ["q2", "q3"]


def test_databases_from_environment(monkeypatch):
    import importlib
    monkeypatch.setenv("NL2SQL_DB_WAREHOUSE", "postgresql+psycopg://u:p@localhost/wh")
    importlib.reload(config)
    try:
        assert config.DATABASES["warehouse"] == "postgresql+psycopg://u:p@localhost/wh"
        assert "chinook" in config.DATABASES
    finally:
        monkeypatch.delenv("NL2SQL_DB_WAREHOUSE")
        importlib.reload(config)


def test_postgres_reads_use_read_only_transaction(db, monkeypatch):
    """No Postgres server needed: check the read-only switch is issued for that dialect."""
    from nl_to_sql.db import QueryResult, ReadOnlyViolation
    calls = []

    def fake_run(engine, sql, commit, setup=None):
        calls.append((sql, setup))
        if sql.startswith("DELETE"):
            raise RuntimeError("cannot execute DELETE in a read-only transaction")
        return QueryResult(columns=["n"], rows=[(1,)])

    monkeypatch.setattr(type(db), "dialect", property(lambda self: "postgresql"))
    monkeypatch.setattr(db, "_run", fake_run)
    assert db.execute_readonly("SELECT 1").rows == [(1,)]
    assert calls[-1] == ("SELECT 1", "SET TRANSACTION READ ONLY")
    with pytest.raises(ReadOnlyViolation):
        db.execute_readonly("DELETE FROM t")


@pytest.mark.parametrize("dialect,sql,kind", [
    ("postgres", "SELECT * FROM t FOR UPDATE", "read"),
    ("postgres", "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d", "write"),
    ("mysql", "SELECT `a` FROM `t` LIMIT 5", "read"),
    ("mysql", "REPLACE INTO t VALUES (1)", "unknown"),  # sqlglot can't parse it; unknown still needs approval
])
def test_classify_other_dialects(dialect, sql, kind):
    assert classify(sql, dialect) == kind
