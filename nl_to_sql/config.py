import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- LLM ---
# Light model by default to conserve credits; set NL2SQL_MODEL=claude-opus-5-5 for harder work.
MODEL = os.getenv("NL2SQL_MODEL", "claude-haiku-5-5")
EFFORT = os.getenv("NL2SQL_EFFORT", "low")  # low | medium | high | xhigh | max
MAX_TOKENS = 16000
# USD per 1M tokens (input, output), for cost estimates in evals.
PRICING = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# --- Agent (v2) ---
MAX_LLM_CALLS = 6        # per question; caps self-correction loops (and spend)
TOOL_RESULT_ROWS = 20    # rows of a result shown back to the model

# --- Accuracy features (v3) ---
VALUE_HINTS = True       # add distinct values / date ranges to the schema prompt
HINT_MAX_VALUES = 12     # list a text column's values only if it has at most this many
FEW_SHOT = False         # add worked examples from nl_to_sql/examples/<db>.json

# --- Large schemas (v4) ---
# "full": whole compact schema in the prompt. "retrieve": only tables relevant to the question.
# "auto": retrieve when the compact schema is longer than FULL_SCHEMA_MAX_CHARS.
SCHEMA_MODE = "auto"
FULL_SCHEMA_MAX_CHARS = 20000   # ~5k tokens
RETRIEVE_TOP_K = 6              # BM25 hits before FK expansion
RETRIEVE_MAX_TABLES = 14        # after FK expansion

# --- Advanced accuracy (v5) ---
# Answer check: "flags" = an LLM reviews the result only when free red-flag rules fire
# (NULLs, several rows for a single-answer question, duplicate rows, odd percentages);
# "always" = review every read result; "off" = never.
VERIFY = "flags"
PLAN = False             # ask for a short plan before multi-step queries (output tokens only)
CANDIDATES = 1           # self-consistency: >1 runs N independent attempts and votes (N x cost)

# --- Conversation (v6) ---
MEMORY_TURNS = 3         # earlier turns (question + SQL + result peek) passed to follow-ups

# --- Databases ---
# Any SQLAlchemy URL works here, e.g. "postgresql+psycopg://user:pw@host/db".
DATABASES = {
    "chinook": f"sqlite:///{ROOT / 'data' / 'chinook.db'}",
    "northwind": f"sqlite:///{ROOT / 'data' / 'northwind.db'}",
    # Large-schema testbed (274 tables): build with scripts/build_enterprise_db.py
    "enterprise": f"sqlite:///{ROOT / 'data' / 'enterprise.db'}",
}
# Add more without editing code: NL2SQL_DB_<NAME>=<SQLAlchemy URL>, e.g.
#   NL2SQL_DB_SALES=postgresql+psycopg://user:pw@host:5432/sales   (pip install "psycopg[binary]")
#   NL2SQL_DB_SHOP=mysql+pymysql://user:pw@host/shop                (pip install pymysql)
DATABASES.update({k.removeprefix("NL2SQL_DB_").lower(): v
                  for k, v in os.environ.items() if k.startswith("NL2SQL_DB_") and v})
DEFAULT_DB = os.getenv("NL2SQL_DEFAULT_DB", "chinook")

# Sample rows per table included in the schema prompt.
SAMPLE_ROWS = 3
# Hard cap on rows fetched from a result set.
MAX_RESULT_ROWS = 200

# --- Permission policy ---
# Per query kind: "auto" (run without asking), "confirm" (ask first), "deny" (never run).
# "unknown" covers SQL we couldn't parse/classify; treat it like a write.
PERMISSIONS = {
    "read": "auto",
    "write": "confirm",
    "unknown": "confirm",
}

# --- Security layer (v7, nl_to_sql/security.py) ---
# Deterministic checks on every statement before it runs. Violations block; warnings go
# to the validator agent.
SENSITIVE_COLUMN_PATTERNS = [r"passw(or)?d", r"\bssn\b", r"social_?security", r"secret", r"api_?key",
                             r"(access|auth|refresh)_?token", r"credit_?card", r"card_?number", r"\bcvv\b"]
HIDDEN_TABLES: list[str] = []   # tables the agents never see or query, e.g. ["users_auth"]
ALLOW_DDL = False               # CREATE/DROP/ALTER/TRUNCATE/GRANT
ALLOW_UNBOUNDED_WRITES = False  # UPDATE/DELETE without WHERE
MAX_SQL_CHARS = 20000
QUERY_TIMEOUT_S = 15            # per statement (SQLite progress handler / Postgres statement_timeout / MySQL)

# --- Descriptions + vector search (v7) ---
AUTO_DESCRIBE = True            # generate table/column descriptions once per DB if none exist (cached)
DESCRIPTIONS_IN_PROMPT = True   # add them to the schema the model sees
RETRIEVAL = "hybrid"            # "bm25" | "vector" | "hybrid" (Reciprocal Rank Fusion of both)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # local ONNX model via fastembed (~65 MB, CPU, no API calls)
RRF_K = 60
BM25_WEIGHT = 1.0               # RRF weight of the BM25 ranking
VECTOR_WEIGHT = 0.5             # RRF weight of each vector ranking (table docs, column docs)

# --- Architecture (v7) ---
ARCHITECTURE = os.getenv("NL2SQL_ARCHITECTURE", "single")  # "single" (SQLAgent) | "multi" (3 agents)
SCHEMA_AGENT = "llm"            # multi: "llm" = schema analyst agent; "retrieval" = no model call, brief from retrieval
SCHEMA_AGENT_MAX_CALLS = 3      # analyst exploration budget before it must hand over
MULTI_VALIDATOR = "always"      # multi: "always" | "flags" (only on red flags / security warnings) | "off"
MULTI_VALIDATION_ROUNDS = 1     # how many times the validator may send a final query back
MULTI_MAX_LLM_CALLS = 10        # all three agents together, per question

PROFILE_SAMPLE_ROWS = 200_000  # value hints on bigger tables are computed from a sample
BLAST_RADIUS_MIN_ROWS = 20      # tables smaller than this skip the blast-radius check
MAX_WRITE_FRACTION = 0.5        # block UPDATE/DELETE that would change more than this share of a table

# --- Answer shape (v7) ---
# "exact": only the columns asked for, in the order asked, duplicates kept unless the question asks
#          for distinct values (what Spider/BIRD expect, and cleaner for users).
# "helpful": the model may add context columns (IDs next to names, the value used for ranking).
ANSWER_STYLE = "helpful"     # "exact" is expected to help on Spider/BIRD but is not yet measured (see STAGES v7)
