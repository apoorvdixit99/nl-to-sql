"""Descriptions + hybrid retrieval. Scripted LLM and a fake embedder: no API calls, no model load."""
import json
import zlib
import shutil

import numpy as np
import pytest
from langchain_core.messages import AIMessage

from nl_to_sql import config, descriptions, retrieval
from nl_to_sql.db import Database
from nl_to_sql.retrieval import TableIndex

from .test_agent import ScriptedLLM


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "chinook.db"
    shutil.copy(config.ROOT / "data" / "chinook.db", path)
    monkeypatch.setitem(config.DATABASES, "tdesc", f"sqlite:///{path}")
    return Database("tdesc")


def reply(tables: dict) -> AIMessage:
    return AIMessage(content="```json\n" + json.dumps(tables) + "\n```",
                     usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})


def test_generate_save_load_roundtrip(db):
    names = db.table_names()
    llm = ScriptedLLM(reply({t: {"description": f"about {t}", "columns": {}} for t in names}))
    tables, usage = descriptions.generate(db, llm, log=lambda *_: None)
    assert usage["calls"] == 1 and tables["Track"]["description"] == "about Track"
    descriptions.save(db, tables, "llm")
    assert descriptions.load(db)["Track"]["description"] == "about Track"


def test_schema_change_invalidates_generated_descriptions(db):
    descriptions.save(db, {"Track": {"description": "x", "columns": {}}}, "llm")
    db.execute("CREATE TABLE NewOne (id INTEGER)", commit=True)
    fresh = Database("tdesc")
    assert descriptions.load(fresh) == {}


def test_ensure_respects_auto_describe(db, monkeypatch):
    llm = ScriptedLLM(reply({"Track": {"description": "songs", "columns": {}}}))
    monkeypatch.setattr(config, "AUTO_DESCRIBE", False)
    assert descriptions.ensure(db, llm) == {} and len(llm.replies) == 1  # no call made
    monkeypatch.setattr(config, "AUTO_DESCRIBE", True)
    assert descriptions.ensure(db, llm)["Track"]["description"] == "songs"


def test_descriptions_appear_in_schema_lines(db):
    db.descriptions = {"Track": {"description": "one row per song",
                                 "columns": {"Milliseconds": "track length in ms"}}}
    line = db.compact_lines(hints=False)["Track"]
    assert '"track length in ms"' in line and line.endswith("-- one row per song")


def _fake_embed(texts):
    """Bag-of-words hashing: deterministic, and similar texts get similar vectors."""
    out = np.zeros((len(texts), 1024), dtype="float32")
    for i, t in enumerate(texts):
        for w in retrieval.tokenize(t):
            out[i, zlib.crc32(w.encode()) % 1024] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_vector_search_uses_column_descriptions(db, monkeypatch):
    monkeypatch.setattr(retrieval, "embed", _fake_embed)
    monkeypatch.setattr(retrieval, "embed_cached", lambda db, kind, texts: _fake_embed(texts))
    monkeypatch.setattr(config, "RETRIEVAL", "vector")
    db.descriptions = {"Track": {"description": "", "columns": {"Milliseconds": "song duration length"}}}
    idx = TableIndex(db)
    assert idx.search("song duration", k=1) == ["Track"]
    assert "Track.Milliseconds" in idx.relevant_columns("song duration length", ["Track"])


def test_hybrid_fuses_both_rankings(db, monkeypatch):
    monkeypatch.setattr(retrieval, "embed", _fake_embed)
    monkeypatch.setattr(retrieval, "embed_cached", lambda db, kind, texts: _fake_embed(texts))
    monkeypatch.setattr(config, "RETRIEVAL", "hybrid")
    idx = TableIndex(db)
    top = idx.search("invoice line unit price", k=3)
    assert "InvoiceLine" in top
