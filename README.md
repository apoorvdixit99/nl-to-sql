# nl-to-sql

A natural-language → SQL agent built in stages (v1 to v7) with LangGraph and Claude.
It comes as either one agent or a team of three, both behind a deterministic security layer.
Read-only queries run automatically. Writes need your approval.
[docs/STAGES.md](docs/STAGES.md) records what each stage added and what the evals showed. There is one git commit per stage.

## Quick start

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_data.py        # Chinook + Northwind (SQLite)
.venv\Scripts\python scripts\build_enterprise_db.py  # optional: 274-table large-schema testbed
copy .env.example .env                               # then set ANTHROPIC_API_KEY

.venv\Scripts\python cli.py --db chinook --arch multi   # terminal REPL (single | multi)
.venv\Scripts\streamlit run app.py                      # web UI (architecture toggle in the sidebar)
```

## How a question flows

```
question ─▶ schema context: compact schema + value hints + table/column descriptions
            (large DBs: hybrid BM25 + vector retrieval, FK expansion)  + memory of the last 3 turns

ARCHITECTURE="single"                       ARCHITECTURE="multi"
  agent ⇄ tools ─▶ verify? ─▶ END             1. schema analyst ⇄ search/describe/sample_values ─▶ brief
                                              2. SQL writer ⇄ executor
                                              3. validator: checks SQL + result (and writes, BEFORE approval)

every statement:  SQLGuard (deterministic security) ─▶ permission policy ─▶ EXPLAIN + blast-radius check
                  ─▶ [validator] ─▶ [user approval] ─▶ read-only connection for reads ─▶ timeout + row cap
```

- **One model call per typical question for the single agent, about three for the team.** The schema is already in the prompt, and a successful `run_query(final=true)` ends the run without asking the model to summarize.
- **Self-correction.** SQL errors, empty results, security blocks and validator rejections go back to the writer, with a budget per question.
- **Security never depends on the model.** `security.py` blocks the following whatever the agents decide:
  - stacked queries, DDL and admin commands;
  - system catalogs and hidden tables;
  - sensitive columns;
  - dangerous functions and injection shapes;
  - mass writes, including disguised ones (`WHERE 1=1`, or a WHERE that matches more than 50% of the rows).
- **UI-agnostic core.** `start(q)` returns `Answer | Pending`, and `resume(approved)` continues. Both architectures share this API, so the CLI, the app and the eval runners all work with either.

## Layout

| Path | What |
|---|---|
| `nl_to_sql/agent.py` | Single LangGraph agent: tools, approval, retrieval mode, verify node, memory, self-consistency |
| `nl_to_sql/multi_agent.py` | Three-agent team (schema analyst, SQL writer, validator); `make_agent()` picks the architecture |
| `nl_to_sql/security.py` | Deterministic security layer (`SQLGuard`) |
| `nl_to_sql/executor.py` | Shared execution path: guard → read-only run / write policy → EXPLAIN → blast radius |
| `nl_to_sql/descriptions.py` | Table/column descriptions: curated > provided (BIRD) > generated once by the LLM, cached |
| `nl_to_sql/retrieval.py` | BM25 + local vector search (fastembed), RRF fusion, FK expansion |
| `nl_to_sql/db.py` | SQLAlchemy wrapper: compact schema, value profiling (cached), FK graph, read-only execution, timeouts |
| `nl_to_sql/benchmarks.py` | Spider 1.0 / BIRD / Spider 2.0-lite loaders and official-style metrics |
| `nl_to_sql/chain.py` | v1 single-shot chain (`cli.py --v1`). Also defines `Answer` |
| `nl_to_sql/verify.py`, `compare.py`, `safety.py` | Red-flag rules + reviewer prompts; result comparison; read/write classification |
| `nl_to_sql/evaluation.py`, `evals/dataset.json` | In-house 53-question eval + grader. Results go to `evals/results/` |
| `scripts/` | Data download/prep, testbed build, description generation, eval/benchmark runners, red team, recall checks |
| `tests/` | 126 offline tests (scripted fake LLM: no API calls), including adversarial security tests and headless Streamlit tests |

## Configuration (`nl_to_sql/config.py`; some settings can also be set in `.env`)

| Setting | Default | Notes |
|---|---|---|
| `NL2SQL_MODEL` / `NL2SQL_EFFORT` | `claude-haiku-5-5` / `low` | Kept light to save credits. Use `claude-opus-5-5` for harder work |
| `NL2SQL_ARCHITECTURE` | `single` | `multi` = the three-agent team |
| `SCHEMA_AGENT`, `MULTI_VALIDATOR` | `llm`, `always` | multi: `retrieval` skips the analyst call; validator `flags` / `off` save calls |
| `PERMISSIONS` | read `auto`, write `confirm`, unknown `confirm` | Values: `auto` / `confirm` / `deny` |
| `SENSITIVE_COLUMN_PATTERNS`, `HIDDEN_TABLES` | password, ssn, token… / none | Never queryable. Hidden tables are invisible to the agents |
| `ALLOW_DDL`, `ALLOW_UNBOUNDED_WRITES` | off | |
| `MAX_WRITE_FRACTION`, `QUERY_TIMEOUT_S` | 0.5, 15 s | Blast-radius limit; per-statement timeout |
| `AUTO_DESCRIBE`, `DESCRIPTIONS_IN_PROMPT` | on, on | One-time description generation per database (cached) |
| `RETRIEVAL` | `hybrid` | `bm25` / `vector` / `hybrid` (used when the schema is too big for the prompt) |
| `ANSWER_STYLE` | `exact` | `exact` = only the requested columns; `helpful` = context columns allowed |
| `NL2SQL_DB_<NAME>` | – | Add any SQLAlchemy URL (Postgres needs `psycopg[binary]`, MySQL needs `pymysql`) |
| `VALUE_HINTS`, `SCHEMA_MODE`, `VERIFY`, `MEMORY_TURNS` | on, `auto`, `flags`, 3 | See STAGES v3–v6 |

## Evals, benchmarks and tests

```powershell
.venv\Scripts\python -m pytest                                        # free
.venv\Scripts\python scripts\run_eval.py --runner multi               # in-house 53 questions (live)
.venv\Scripts\python scripts\prepare_benchmarks.py                    # extract Spider/BIRD/Spider 2.0 (see script for sources)
.venv\Scripts\python scripts\run_benchmark.py bird --arch multi       # live, seeded sample (default 100 / 100 / 30)
.venv\Scripts\python scripts\run_benchmark.py spider --check-gold     # free
.venv\Scripts\python scripts\retrieval_benchmark.py                   # free: BM25 vs vector vs hybrid table ranking
.venv\Scripts\python scripts\red_team.py                              # live: 10 adversarial prompts per architecture
.venv\Scripts\python scripts\describe_schema.py enterprise --force     # (re)generate descriptions
```

Results are in [docs/STAGES.md](docs/STAGES.md) (v7).
