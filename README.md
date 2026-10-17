# nl-to-sql

A natural-language → SQL agent built in stages (v1 to v6) with LangGraph and Claude.
Read-only queries run automatically. Writes need your approval.
[docs/STAGES.md](docs/STAGES.md) records what each stage added and what the evals showed. There is one git commit per stage.

## Quick start

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_data.py        # Chinook + Northwind (SQLite)
.venv\Scripts\python scripts\build_enterprise_db.py  # optional: 274-table large-schema testbed
copy .env.example .env                               # then set ANTHROPIC_API_KEY

.venv\Scripts\python cli.py --db chinook             # terminal REPL
.venv\Scripts\streamlit run app.py                   # web UI
```

## How a question flows

```
question
  │
  ├─ schema context ── small DB: compact schema + value hints in the (cached) system prompt
  │                    large DB: BM25 table retrieval + FK expansion → user message
  │                    + memory: last 3 turns (question, SQL, result peek)
  ▼
LangGraph:  agent ──tool calls──▶ tools ──final result──▶ verify? ──▶ END
              ▲                     │  describe_tables / search_tables / run_query
              └──── errors, empty ──┘
                    results, reviewer
                    issues

run_query:  sqlglot classify ─▶ read: run on a READ-ONLY connection
                               └▶ write/unknown: EXPLAIN check ─▶ interrupt() for approval ─▶ run + commit
```

- **One model call per typical question.** The schema is already in the prompt, and a successful `run_query(final=true)` ends the run without asking the model to summarize.
- **Self-correction.** SQL errors, empty results and reviewer issues go back to the model, capped at `MAX_LLM_CALLS`.
- **Answer check.** Free red-flag rules run on every result; an LLM review runs only when one fires (`VERIFY`).
- **UI-agnostic core.** `SQLAgent.start(q)` returns `Answer | Pending`, and `resume(approved)` continues. The CLI and Streamlit app both use this.

## Layout

| Path | What |
|---|---|
| `nl_to_sql/agent.py` | LangGraph agent: tools, approval, retrieval mode, verify node, memory, self-consistency |
| `nl_to_sql/chain.py` | v1 single-shot chain (kept for comparison: `cli.py --v1`). Also defines `Answer` |
| `nl_to_sql/db.py` | SQLAlchemy wrapper: compact schema, value profiling (cached), FK graph, read-only execution |
| `nl_to_sql/safety.py` | sqlglot read/write/unknown classification |
| `nl_to_sql/retrieval.py` | BM25 table index + FK expansion (no LLM calls, no dependencies) |
| `nl_to_sql/verify.py` | Red-flag rules + reviewer prompt |
| `nl_to_sql/compare.py` | Result-set comparison (used by the eval grader and the self-consistency vote) |
| `nl_to_sql/evaluation.py`, `evals/dataset.json` | 53-question execution-accuracy eval + grader. Results go to `evals/results/` |
| `scripts/` | Data download, testbed build, eval runner, free retrieval-recall check, DB copier |
| `tests/` | 50 offline tests (scripted fake LLM: no API calls), including headless Streamlit tests |

## Configuration (`nl_to_sql/config.py`, overridable via `.env`)

| Setting | Default | Notes |
|---|---|---|
| `NL2SQL_MODEL` / `NL2SQL_EFFORT` | `claude-haiku-5-5` / `low` | Kept light to save credits. Use `claude-opus-5-5` for harder work |
| `PERMISSIONS` | read `auto`, write `confirm`, unknown `confirm` | Values: `auto` / `confirm` / `deny` |
| `NL2SQL_DB_<NAME>` | – | Add any SQLAlchemy URL (Postgres needs `psycopg[binary]`, MySQL needs `pymysql`) |
| `VALUE_HINTS` | on | Distinct values / date ranges in the schema |
| `SCHEMA_MODE` | `auto` | Retrieval when the compact schema is over 20k characters |
| `VERIFY` | `flags` | `always` / `off` |
| `PLAN`, `FEW_SHOT`, `CANDIDATES` | off / off / 1 | Implemented and measured; no gain at current difficulty (see STAGES) |
| `MEMORY_TURNS` | 3 | Turns carried into follow-ups (CLI and app) |

## Evals and tests

```powershell
.venv\Scripts\python -m pytest                                  # free
.venv\Scripts\python scripts\run_eval.py --check-gold           # free: validate the reference queries
.venv\Scripts\python scripts\retrieval_recall.py                # free: retrieval coverage
.venv\Scripts\python scripts\run_eval.py                        # live: ~55 calls, ~$0.01 on Haiku
.venv\Scripts\python scripts\run_eval.py --db enterprise        # live: large-schema variant
.venv\Scripts\python scripts\run_eval.py --set VERIFY=always PLAN=true --label my-experiment
```

Latest results: 53/53 on Chinook+Northwind ($0.011) and 53/53 on the 274-table testbed ($0.018).
