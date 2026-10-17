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

# --- Databases ---
# Any SQLAlchemy URL works here, e.g. "postgresql+psycopg://user:pw@host/db".
DATABASES = {
    "chinook": f"sqlite:///{ROOT / 'data' / 'chinook.db'}",
    "northwind": f"sqlite:///{ROOT / 'data' / 'northwind.db'}",
    # Large-schema testbed (274 tables): build with scripts/build_enterprise_db.py
    "enterprise": f"sqlite:///{ROOT / 'data' / 'enterprise.db'}",
}
DEFAULT_DB = "chinook"

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
