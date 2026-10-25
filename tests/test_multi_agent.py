"""Three-agent architecture. One scripted fake LLM plays all agents in order: no API calls."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from nl_to_sql import config
from nl_to_sql.agent import Pending, SQLAgent
from nl_to_sql.multi_agent import MultiAgentSQL, make_agent

from .test_agent import ScriptedLLM, artist_count, call, db  # noqa: F401  (db fixture)

_n = 0


def tool(name: str, **args) -> AIMessage:
    global _n
    _n += 1
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"m{_n}"}])


def brief(tables, columns=(), joins=(), values=(), notes="") -> AIMessage:
    return tool("submit_brief", tables=list(tables), columns=list(columns), joins=list(joins),
                values=list(values), notes=notes)


APPROVE = AIMessage(content="APPROVE")


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(config, "SCHEMA_AGENT", "llm")
    monkeypatch.setattr(config, "MULTI_VALIDATOR", "always")


def test_three_agents_answer_a_read(db):
    llm = ScriptedLLM(
        brief(["Track", "Genre"], ["Genre.Name"], ["Track.GenreId = Genre.GenreId"], ["Genre.Name = 'Rock'"]),
        call("SELECT COUNT(*) FROM Track t JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = 'Rock'"),
        APPROVE)
    a = MultiAgentSQL(db, llm=llm).start("how many rock tracks")
    assert a.status == "ok" and a.result.rows == [(1297,)]
    assert a.llm_calls == 3 and set(a.agents) == {"schema", "sql", "validator"}
    assert a.review == "APPROVE" and "Genre.Name = 'Rock'" in a.brief
    sql_prompt = llm.seen[1][-1].content            # what the SQL writer saw
    assert "Schema analyst's brief" in sql_prompt and "Track(" in sql_prompt and "Genre(" in sql_prompt
    assert "Album(" not in sql_prompt                # only the brief's tables
    validator_prompt = llm.seen[2][-1].content
    assert "how many rock tracks" in validator_prompt and "1297" in validator_prompt


def test_validator_rejection_sends_query_back_once(db):
    llm = ScriptedLLM(
        brief(["Invoice"]),
        call("SELECT strftime('%Y', InvoiceDate) y, SUM(Total) FROM Invoice GROUP BY y"),
        AIMessage(content="REJECT: the question asks only for 2010"),
        call("SELECT SUM(Total) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2010'"))
    a = MultiAgentSQL(db, llm=llm).start("revenue in 2010")
    assert a.status == "ok" and a.result.rows == [(481.45,)] and a.llm_calls == 4
    assert "validator rejected" in llm.seen[3][-1].content


def test_schema_agent_explores_values_before_briefing(db):
    llm = ScriptedLLM(
        tool("sample_values", table="genre", column="name", contains="roc"),
        brief(["Genre"], values=["Genre.Name = 'Rock'"]),
        call("SELECT GenreId FROM Genre WHERE Name = 'Rock'"),
        APPROVE)
    a = MultiAgentSQL(db, llm=llm).start("id of the rock genre")
    sampled = llm.seen[1][-1].content
    assert "'Rock'" in sampled and "'Rock And Roll'" in sampled   # case-insensitive names, LIKE filter
    assert a.status == "ok" and a.agents["schema"]["calls"] == 2


def test_sample_values_refuses_sensitive_columns(db, monkeypatch):
    monkeypatch.setattr(config, "SENSITIVE_COLUMN_PATTERNS", [r"email"])
    llm = ScriptedLLM(tool("sample_values", table="Customer", column="Email"), brief(["Customer"]),
                      call("SELECT COUNT(*) FROM Customer"), APPROVE)
    MultiAgentSQL(db, llm=llm).start("how many customers")
    assert "sensitive" in llm.seen[1][-1].content


def test_write_is_validated_before_the_user_is_asked(db):
    llm = ScriptedLLM(brief(["Artist"]), call("DELETE FROM Artist WHERE ArtistId = 275"), APPROVE)
    agent = MultiAgentSQL(db, llm=llm)
    p = agent.start("delete artist 275")
    assert isinstance(p, Pending) and "DELETE" in p.sql
    assert "MODIFY data" in llm.seen[2][-1].content   # validator ran before the approval prompt
    a = agent.resume(True)
    assert a.status == "ok" and a.result.rowcount == 1 and artist_count(db) == 274
    assert len(llm.replies) == 0                       # resuming didn't re-run the validator


def test_declined_write_changes_nothing(db):
    agent = MultiAgentSQL(db, llm=ScriptedLLM(brief(["Artist"]), call("DELETE FROM Artist WHERE ArtistId = 275"),
                                              APPROVE))
    assert isinstance(agent.start("delete artist 275"), Pending)
    assert agent.resume(False).status == "declined" and artist_count(db) == 275


def test_validator_can_stop_a_bad_write(db):
    llm = ScriptedLLM(brief(["Artist"]), call("DELETE FROM Artist WHERE ArtistId = 1"),
                      AIMessage(content="REJECT: the user meant the last artist added, not AC/DC"),
                      AIMessage(content="CANNOT_ANSWER: unclear which artist"))
    out = MultiAgentSQL(db, llm=llm).start("delete that artist")
    assert not isinstance(out, Pending) and out.status == "cannot_answer" and artist_count(db) == 275


def test_security_layer_still_applies(db):
    llm = ScriptedLLM(brief(["Artist"]), call("SELECT Name FROM Artist; DROP TABLE Artist"),
                      call("SELECT Name FROM Artist LIMIT 1"), APPROVE)
    a = MultiAgentSQL(db, llm=llm).start("an artist")
    assert a.status == "ok" and any("multiple statements" in s for s in a.security)
    assert artist_count(db) == 275


def test_retrieval_only_schema_mode_skips_the_analyst_call(db, monkeypatch):
    monkeypatch.setattr(config, "SCHEMA_AGENT", "retrieval")
    llm = ScriptedLLM(call("SELECT COUNT(*) FROM Track"), APPROVE)
    a = MultiAgentSQL(db, llm=llm).start("how many tracks")
    assert a.status == "ok" and a.llm_calls == 2 and "schema" not in a.agents


def test_flag_gated_validator_skips_clean_results(db, monkeypatch):
    monkeypatch.setattr(config, "MULTI_VALIDATOR", "flags")
    llm = ScriptedLLM(brief(["Track"]), call("SELECT COUNT(*) FROM Track"))
    a = MultiAgentSQL(db, llm=llm).start("how many tracks")
    assert a.status == "ok" and a.llm_calls == 2 and "validator" not in a.agents


def test_analyst_without_brief_falls_back(db):
    llm = ScriptedLLM(AIMessage(content="Use the Track table."), call("SELECT COUNT(*) FROM Track"), APPROVE)
    a = MultiAgentSQL(db, llm=llm).start("how many tracks")
    assert a.status == "ok" and "Use the Track table." in a.brief


def test_cannot_answer(db):
    llm = ScriptedLLM(brief([], notes="no weather data"), AIMessage(content="CANNOT_ANSWER: no weather data"))
    a = MultiAgentSQL(db, llm=llm).start("weather?")
    assert a.status == "cannot_answer"


def test_memory_reaches_both_agents(db):
    llm = ScriptedLLM(brief(["Invoice"]), call("SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY 1"), APPROVE,
                      brief(["Invoice"]), call("SELECT BillingCountry, COUNT(*) FROM Invoice GROUP BY 1"), APPROVE)
    agent = MultiAgentSQL(db, llm=llm, memory=True)
    agent.start("revenue by country")
    agent.start("and the invoice count?")
    assert "Earlier in this conversation" in llm.seen[3][-1].content   # analyst
    assert "revenue by country" in llm.seen[4][-1].content             # writer


def test_make_agent_follows_config(db, monkeypatch):
    monkeypatch.setattr(config, "ARCHITECTURE", "multi")
    assert isinstance(make_agent(db, llm=ScriptedLLM()), MultiAgentSQL)
    monkeypatch.setattr(config, "ARCHITECTURE", "single")
    agent = make_agent(db, llm=ScriptedLLM())
    assert isinstance(agent, SQLAgent) and not isinstance(agent, MultiAgentSQL)


def test_retrieve_mode_gives_analyst_candidates(monkeypatch):
    from nl_to_sql.db import Database
    if not (config.ROOT / "data" / "enterprise.db").exists():
        pytest.skip("run scripts/build_enterprise_db.py first")
    monkeypatch.setattr(config, "RETRIEVAL", "bm25")
    ent = Database("enterprise")
    llm = ScriptedLLM(brief(["music_Track"]), call('SELECT COUNT(*) FROM "music_Track"'), APPROVE)
    a = MultiAgentSQL(ent, llm=llm).start("Music store: how many tracks?")
    first = llm.seen[0][1]
    assert isinstance(first, HumanMessage) and "Candidate tables:" in first.content
    assert a.status == "ok" and a.result.rows == [(3503,)]


def test_blast_radius_blocks_before_validator_or_user(db):
    llm = ScriptedLLM(brief(["Artist"]), call("DELETE FROM Artist WHERE ArtistId > 0"),
                      AIMessage(content="CANNOT_ANSWER: blocked"))
    out = MultiAgentSQL(db, llm=llm).start("delete artists")
    assert not isinstance(out, Pending) and artist_count(db) == 275
    assert any("275 of 275" in s for s in out.security) and "validator" not in out.agents
