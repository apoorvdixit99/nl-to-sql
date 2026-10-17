# Stage log

What each stage added, why, and what the evals said. Each stage is one git commit.
All eval numbers are single runs on `claude-haiku-5-5` at effort `low`, so ±1 question is noise.

## v1: single-shot chain (`nl_to_sql/chain.py`)
- Full schema (CREATE TABLE + 3 sample rows per table) in the system prompt → one LLM call → SQL parsed from a fenced block → safety check → execute.
- `safety.py` classifies SQL by parsing it with sqlglot: `read` / `write` / `unknown`. Write nodes anywhere in the tree count, so `WITH x AS (DELETE ...)` and `SELECT 1; DROP ...` are writes.
- The permission policy lives in config (`read: auto`, `write: confirm`, `unknown: confirm`). Reads are always rolled back.
- Structured output wasn't used: LangChain's `with_structured_output` forces a tool call, which Opus 5.5 rejects.

## v2: LangGraph agent (`nl_to_sql/agent.py`)
- Graph: `agent ⇄ tools`, with tools `describe_tables` and `run_query(sql, final)`.
- Errors go back to the model so it can fix the query, up to `MAX_LLM_CALLS` (6) per question.
- Writes pause via `interrupt()`. The public API is `start(question)` → `Answer | Pending` and `resume(approved)`, so the CLI and a future Streamlit UI drive it the same way.
- Designed for few API calls:
  - The compact schema (one line per table, about 70% smaller than v1's) is in the prompt, so there's no discovery round trip.
  - A successful `run_query(final=true)` ends the run without asking the model to summarize the result.
  - A final query that returns 0 rows gets one more look.
- Tests use a scripted fake LLM, so they cost nothing.

## v3: accuracy foundations
- **Eval set** (`evals/dataset.json`, 43 questions across both DBs, graded easy/medium/hard):
  - Graded on results, not SQL text. The grader ignores row and column order, allows extra columns, rounds floats to 2 decimals, and accepts a name split across first/last columns.
  - `alt_gold_sql` covers ties. The reference answers were checked for ties before use.
  - `nw14` checks that the model says "no data" (this Northwind copy has no 1997 orders).
  - Run with `scripts/run_eval.py`. `--check-gold` is free; `--set KEY=VALUE` turns features on and off for comparison runs.
- **Value hints** (`Database.profile()`, cached next to the .db file, built with plain SQL, no LLM calls):
  - Columns that look categorical get `{'0','1'}`-style value lists.
  - Date columns get `[min .. max, has NULLs]`.
  - Columns with a unique value per row (phones, addresses) are skipped as noise.
- **Prompt fixes**:
  - "Which X has the most…" means the top row, plus any ties.
  - Follow the question's exact wording (e.g. "after the first", "rounded to 1 decimal").
  - Use the listed values exactly.
- **Write validation**: `EXPLAIN` runs before the approval prompt, so you're never asked to approve a query that can't run.
- **Few-shot examples**: implemented (`nl_to_sql/examples/`), but off by default because they added tokens without improving accuracy.

| Run | Accuracy (43) | Calls | Input tok | Cost |
|---|---|---|---|---|
| v1 chain | 42 | 43 | ~200k | ~$0.026 |
| v2 agent | 39 | 49 | ~95k | ~$0.016 |
| v3 prompt only | 42 | 48 | 102k | $0.016 |
| **v3 prompt + hints** | **43** | 44 | 111k | $0.017 |
| v3 + few-shot | 42 | 44 | 124k | $0.018 |

Takeaway: Haiku 5.5 now saturates this set, so later stages need a harder benchmark.

## v4: large schemas (`nl_to_sql/retrieval.py`)
- **Testbed** (`scripts/build_enterprise_db.py`, no LLM calls), 274 tables in total:
  - Chinook's tables copied as `music_*` and Northwind's as `shop_*`, with constraints and data kept (table names rewritten with sqlglot).
  - 250 generated tables across about 30 domains, including deliberately confusing names (`rail_track_maintenance`, `hr_employee_shifts`, `events_playlist_requests`).
  - Eval questions are remapped automatically. Table names in the reference queries are rewritten with sqlglot, and each question gets a "Music store:" / "Northwind shop:" prefix. A test checks the remapped reference answers match the originals.
- **Retriever** (no LLM calls, no new dependencies):
  - BM25 over one document per table: the table name counted three times, plus column names, value hints and FK targets.
  - Takes the top 6 hits, then adds bridge tables (linked to two or more hits) and FK neighbours of the top 3 hits, capped at 14 tables.
- **Agent integration** (`SCHEMA_MODE=auto`): if the compact schema is over about 20k characters, the agent switches to retrieve mode:
  - The system prompt (cached) holds only the instructions and all table names.
  - The retrieved tables go in the user message.
  - A `search_tables` tool lets the agent recover when retrieval misses a table.
- **Free recall check** (`scripts/retrieval_recall.py`): covers every needed table for 42/42 questions (40/42 without the domain prefix), averaging 8.7 of 274 tables.
- `langchain-community`'s deprecated `SQLDatabase` was replaced with plain SQLAlchemy inspection, which only loads the tables it needs.

| Run (enterprise, 274 tables) | Accuracy | Input tok (cached) | Cost |
|---|---|---|---|
| **retrieve** | **42/43** | 201k (136k) | **$0.014** |
| full schema | 41/43 | 1.41M (1.38M) | $0.025 (~$0.14 without caching) |

Remaining misses:
- `ch17`: Haiku keeps the first year's NULL row despite "after the first". This is a target for v5's answer check.
- `nw09`: an ambiguous wording, now accepted both ways.

## v5: advanced accuracy (`nl_to_sql/verify.py`, `compare.py`)
- **10 harder questions** (53 in total), for example:
  - second-ranked genre per year (`RANK` within groups),
  - median computed without a `MEDIAN` function,
  - year-over-year deltas per product or employee,
  - per-order value thresholds,
  - late-delivery percentage per shipper,
  - running totals.
- **Read-only execution.** Reads, and `SELECT`/`WITH` statements sqlglot can't classify, run on a read-only connection (SQLite `mode=ro`, `SET TRANSACTION READ ONLY` on Postgres/MySQL):
  - A misclassified write can't change data.
  - This fixed a false approval prompt for a median query using `LIMIT <expression>`.
  - Statements that aren't queries (PRAGMA, ATTACH) still ask for approval.
- **Answer check** (`VERIFY="flags"`, the default):
  - Free rules check every result: NULLs, several rows for a single-answer question (ties allowed), duplicate rows, percentages outside 0–100.
  - An LLM review runs only when a rule fires, and gets one chance to send the agent back to fix the query. `always` and `off` are available.
- **Plan rule** (`PLAN`, off): a 1–3 line plan before multi-step queries. Costs only output tokens.
- **Self-consistency** (`CANDIDATES=N`, off): N independent attempts, with results grouped by the eval's comparison logic; the largest group wins.
- **Fallback answer:** if the agent ends with prose after an exploratory query, the rows of its last successful read are returned. This fixed a real failure on `ch07`.
- `results_match` moved to `compare.py`, so the grader and the voting share it.

| Run (53 questions) | Accuracy | Calls | Cost |
|---|---|---|---|
| verify off (= v4 behaviour) | 52 | 54 | $0.011 |
| **verify=flags (default)** | 52* | 56 | $0.011 |
| PLAN=true | 53 | 55 | $0.011 |
| CANDIDATES=3 (27 hard only) | 27/27 | 82 | $0.021 |

\* Its one miss (`ch07`, prose instead of a final query) is fixed by the fallback above. A re-run of `ch07`, `ch17` and `ch19` passed all three.

Findings:
- The answer check fired on 1 of 53 questions (a legitimate tie, judged OK), so it adds almost no cost.
- Self-consistency attempts agreed 3/3 on 25 of 27 hard questions. There's too little diversity to vote on (Haiku 5.5 doesn't accept temperature changes), so it isn't worth 2–3× the cost here.
- At this difficulty Haiku 5.5 is at the eval's ceiling, so differences between these features are within noise.

## v6: production extras
- **Streamlit UI** (`app.py`):
  - Chat history with result tables, SQL in an expander, and model calls/tokens shown per answer.
  - A running cost estimate in the sidebar.
  - A database picker, a memory toggle and a "New conversation" button.
  - Writes show the SQL with **Approve / Decline** buttons. This is the same `Pending` → `resume()` flow the CLI uses.
  - Tested headlessly with Streamlit's `AppTest` and a scripted fake LLM.
- **Conversation memory** (`SQLAgent(memory=True)`, on in the CLI and app, off in evals):
  - The last `MEMORY_TURNS` turns go into the next user message as question + SQL + a peek at the result, not whole transcripts. That keeps token use flat, and each question still runs in a fresh LangGraph thread.
  - Retrieval searches with the previous question too, so follow-ups get the same tables.
  - The CLI has `/reset`.
- **Other databases:**
  - `NL2SQL_DB_<NAME>=<url>` adds any SQLAlchemy URL.
  - Postgres/MySQL reads run under `SET TRANSACTION READ ONLY`.
  - `scripts/copy_database.py` copies a database into any SQLAlchemy URL.
  - Dialect handling is covered by offline tests (classification, the read-only switch).
  - **Not verified against a live Postgres/MySQL server.** None is installed here. An embedded-Postgres attempt (PGlite over its socket server) was too unstable on Windows: the server crashed on disconnect, psycopg segfaulted and pg8000 failed the handshake.

| Final regression | Accuracy | Calls | Cost |
|---|---|---|---|
| Chinook + Northwind (53) | **53/53** | 55 | $0.011 |
| Enterprise, 274 tables (53) | **53/53** | 58 | $0.018 |

Live memory check: "Top 3 genres by number of tracks" → "Now show the average track length for those genres". The follow-up reused the top-3 query as a CTE, in 1 call.
