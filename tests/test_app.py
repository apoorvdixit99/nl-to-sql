"""Headless Streamlit tests (AppTest). The LLM is replaced with a scripted fake: no API calls."""
import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from nl_to_sql import agent as agent_module
from nl_to_sql import config

from .test_agent import ScriptedLLM, call


@pytest.fixture
def fake_llm(monkeypatch):
    """Every ChatAnthropic built by the app returns this script's replies in order."""
    script = ScriptedLLM()
    monkeypatch.setattr(agent_module, "ChatAnthropic", lambda **kw: script)
    monkeypatch.setattr(config, "VERIFY", "off")
    return script


def app() -> AppTest:
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_question_shows_rows(fake_llm):
    fake_llm.replies += [call("SELECT Name FROM Artist ORDER BY ArtistId LIMIT 3", text="By ID.")]
    at = app()
    at.chat_input[0].set_value("first three artists").run()
    assert not at.exception
    df = at.dataframe[0].value
    assert list(df["Name"]) == ["AC/DC", "Accept", "Aerosmith"]
    assert any("1 model call" in c.value for c in at.caption)


def test_write_needs_button_approval_and_decline_runs_nothing(fake_llm):
    fake_llm.replies += [call("DELETE FROM Artist WHERE ArtistId = 1")]
    at = app()
    at.chat_input[0].set_value("delete artist 1").run()
    assert "needs your approval" in at.warning[0].value
    assert at.chat_input[0].disabled
    decline = [b for b in at.button if b.label == "Decline"][0]
    decline.click().run()
    assert any("Nothing was run" in i.value for i in at.info)
    from nl_to_sql.db import Database
    assert Database("chinook").execute("SELECT COUNT(*) FROM Artist WHERE ArtistId = 1", False).rows == [(1,)]


def test_cannot_answer_message(fake_llm):
    fake_llm.replies += [AIMessage(content="CANNOT_ANSWER: no weather data here.")]
    at = app()
    at.chat_input[0].set_value("weather?").run()
    assert any("Can't answer" in w.value for w in at.warning)
