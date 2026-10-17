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
