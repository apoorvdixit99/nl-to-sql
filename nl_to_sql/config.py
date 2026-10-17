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

# --- Agent (v2) ---
MAX_LLM_CALLS = 6        # per question; caps self-correction loops (and spend)
TOOL_RESULT_ROWS = 20    # rows of a result shown back to the model

# --- Databases ---
# Any SQLAlchemy URL works here, e.g. "postgresql+psycopg://user:pw@host/db".
DATABASES = {
    "chinook": f"sqlite:///{ROOT / 'data' / 'chinook.db'}",
    "northwind": f"sqlite:///{ROOT / 'data' / 'northwind.db'}",
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
