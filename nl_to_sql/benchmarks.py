"""v7: harder benchmarks - Spider 1.0 (dev), BIRD (dev) and Spider 2.0-lite (local SQLite subset).

Data comes from scripts/prepare_benchmarks.py. Each benchmark is scored with its own
official-style metric, plus our lenient grader (compare.results_match) for reference:

- Spider 1.0: execution match in the style of the official test-suite evaluator: same rows
  (as a multiset, or in order when the gold query has ORDER BY), columns may be permuted.
  Run on the original dev databases only, not the extra test-suite database variants.
- BIRD:       the official EX metric: set(predicted rows) == set(gold rows).
- Spider 2.0: the official spider2-lite comparison (reimplemented from its evaluate.py):
  every gold column (or the task's condition_cols) must appear among the predicted columns,
  numbers within 1e-2, optional order-insensitivity, several acceptable gold CSVs per task.

Questions are sampled deterministically (seeded), stratified by difficulty where the
benchmark provides one, so runs are comparable and cheap.
"""
import io
import json
import math
import random
import time
from collections import Counter, defaultdict
from itertools import permutations

from . import config, descriptions
from .compare import results_match
from .db import Database, QueryResult

BENCH_DIR = config.ROOT / "data" / "benchmarks"
BENCHMARKS = ("spider", "bird", "spider2")
DEFAULT_SAMPLE = {"spider": 100, "bird": 100, "spider2": 30}


# ---- items ---------------------------------------------------------------------------

def db_name(bench: str, db_id: str) -> str:
    return f"{bench}__{db_id}"


def register(bench: str, db_id: str) -> str:
    name = db_name(bench, db_id)
    config.DATABASES[name] = f"sqlite:///{(BENCH_DIR / bench / 'db' / f'{db_id}.sqlite').as_posix()}"
    return name


def load_items(bench: str, sample: int | None = None, seed: int = 7, ids: list[str] | None = None) -> list[dict]:
    path = BENCH_DIR / bench / "items.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/prepare_benchmarks.py {bench}")
    items = json.loads(path.read_text(encoding="utf-8"))
    if ids:
        items = [i for i in items if i["id"] in ids]
    elif sample and sample < len(items):
        rng = random.Random(seed)
        levels = defaultdict(list)
        for i in items:
            levels[i.get("level") or "all"].append(i)
        picked = []
        for lvl, group in sorted(levels.items()):  # proportional stratified sample
            k = max(1, round(sample * len(group) / len(items)))
            picked += rng.sample(group, min(k, len(group)))
        if len(picked) < sample:  # rounding left a gap: top up from what's left
            chosen = {i["id"] for i in picked}
            rest = [i for i in items if i["id"] not in chosen]
            picked += rng.sample(rest, min(sample - len(picked), len(rest)))
        items = sorted(picked[:sample], key=lambda i: i["id"])
    for i in items:
        i["bench"], i["db"] = bench, register(bench, i["db_id"])
        i.setdefault("level", "")
    return items


def agent_question(item: dict) -> str:
    """The text the agent sees: BIRD adds its evidence hint, Spider 2.0 its reference document."""
    q = item["question"]
    if item["bench"] == "bird" and item.get("evidence"):
        return f"{q}\nHint: {item['evidence']}"
    if item["bench"] == "spider2" and item.get("evidence"):
        return f"{q}\n\nReference document:\n{item['evidence'][:6000]}"
    return q


# ---- gold results (cached) ---------------------------------------------------------------

def gold_result(item: dict) -> QueryResult | list | None:
    """Spider/BIRD: QueryResult of the gold SQL. Spider 2.0: list of gold DataFrames."""
    if item["bench"] == "spider2":
        import pandas as pd
        return [pd.read_csv(BENCH_DIR / p) for p in item["gold_csvs"]]
    cache = BENCH_DIR / item["bench"] / "gold_cache" / f"{item['id']}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return QueryResult(columns=d["columns"], rows=[tuple(r) for r in d["rows"]])
    db = Database(item["db"])
    old = config.QUERY_TIMEOUT_S
    config.QUERY_TIMEOUT_S = 120  # gold queries get longer than model queries
    try:
        res = db.execute_readonly(item["gold_sql"])
    except Exception as e:
        print(f"  gold failed for {item['id']}: {e}")
        return None
    finally:
        config.QUERY_TIMEOUT_S = old
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"columns": res.columns, "rows": [list(r) for r in res.rows]}, default=str),
                     encoding="utf-8")
    return res


# ---- metrics ---------------------------------------------------------------------------------

def spider_ex(gold: QueryResult, pred: QueryResult, order_matters: bool) -> bool:
    """Test-suite-style execution match (exact values, column permutation allowed)."""
    g, p = [tuple(r) for r in gold.rows], [tuple(r) for r in pred.rows]
    if not g and not p:
        return True
    if len(g) != len(p) or not p or len(g[0]) != len(p[0]):
        return False
    n = len(g[0])
    gcols = [Counter(r[i] for r in g) for i in range(n)]
    pcols = [Counter(r[i] for r in p) for i in range(n)]
    candidates = [[j for j in range(n) if pcols[j] == gcols[i]] for i in range(n)]
    if any(not c for c in candidates):
        return False
    tried = 0
    for perm in permutations(range(n)):
        if any(perm[i] not in candidates[i] for i in range(n)):
            continue
        tried += 1
        if tried > 5000:
            break
        p2 = [tuple(r[perm[i]] for i in range(n)) for r in p]
        if (p2 == g) if order_matters else (Counter(p2) == Counter(g)):
            return True
    return False


def bird_ex(gold: QueryResult, pred: QueryResult) -> bool:
    """BIRD's official metric."""
    return set(map(tuple, pred.rows)) == set(map(tuple, gold.rows))


def spider2_match(pred: QueryResult, golds: list, condition_cols, ignore_order: bool) -> bool:
    """Reimplementation of spider2-lite evaluate.py compare_(multi_)pandas_table."""
    import pandas as pd
    if not pred.columns:
        return False
    buf = io.StringIO()
    pd.DataFrame(pred.rows, columns=[str(c) for c in pred.columns]).to_csv(buf, index=False)
    buf.seek(0)
    pred_df = pd.read_csv(buf)  # same CSV round trip the official harness applies to predictions

    if condition_cols in (None, [], [[]], [None]):
        conds = [[] for _ in golds]
    elif len(golds) > 1 and not all(isinstance(c, list) for c in condition_cols):
        conds = [condition_cols for _ in golds]
    else:
        conds = condition_cols if len(golds) > 1 else [condition_cols]

    def norm(x):
        return 0 if pd.isna(x) else x

    def vec_match(v1, v2):
        v1, v2 = [norm(x) for x in v1], [norm(x) for x in v2]
        if ignore_order:
            key = lambda x: (x is None, str(x), isinstance(x, (int, float)))  # noqa: E731
            v1, v2 = sorted(v1, key=key), sorted(v2, key=key)
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=1e-2):
                    return False
            elif a != b:
                return False
        return True

    pred_vecs = pred_df.transpose().values.tolist()
    for gold, cond in zip(golds, conds):
        cols = cond if isinstance(cond, list) else [cond]
        g = gold.iloc[:, cols] if cols else gold
        if all(any(vec_match(gv, pv) for pv in pred_vecs) for gv in g.transpose().values.tolist()):
            return True
    return False


def grade(item: dict, answer, gold) -> tuple[bool, bool]:
    """(official-style pass, lenient pass)."""
    res = answer.result if answer.status == "ok" else None
    if res is None or gold is None:
        return False, False
    if item["bench"] == "spider2":
        ok = spider2_match(res, gold, item.get("condition_cols"), item.get("ignore_order", False))
        return ok, ok
    lenient = results_match(gold, res)
    if item["bench"] == "bird":
        return bird_ex(gold, res), lenient
    return spider_ex(gold, res, "order by" in item["gold_sql"].lower()), lenient


# ---- runner ------------------------------------------------------------------------------------

def run(bench: str, items: list[dict], make_runner, label: str, llm_for_descriptions=None) -> dict:
    """make_runner(db) -> object with .ask(question) -> Answer (writes auto-declined)."""
    old_rows = config.MAX_RESULT_ROWS
    config.MAX_RESULT_ROWS = 100_000  # compare full result sets
    dbs, runners, rows = {}, {}, []
    desc_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    try:
        for n, item in enumerate(items, 1):
            if item["db"] not in runners:
                db = Database(item["db"])
                found = descriptions.load(db)
                if not found and config.AUTO_DESCRIBE and llm_for_descriptions is not None:
                    found, u = descriptions.generate(db, llm_for_descriptions)
                    descriptions.save(db, found, "llm")
                    for k in desc_usage:
                        desc_usage[k] += u[k]
                db.descriptions = found
                dbs[item["db"]], runners[item["db"]] = db, make_runner(db)
            gold = gold_result(item)
            t0 = time.time()
            try:
                answer = runners[item["db"]].ask(agent_question(item))
            except Exception as e:  # keep going: one broken item shouldn't sink the run
                from .chain import Answer
                answer = Answer(question=item["question"], status="error", error=f"{type(e).__name__}: {e}")
            official, lenient = grade(item, answer, gold)
            rows.append({
                "id": item["id"], "db": item["db_id"], "level": item["level"], "pass": official,
                "lenient_pass": lenient, "status": answer.status, "sql": answer.sql, "error": answer.error,
                "llm_calls": answer.llm_calls, "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens, "cache_read_tokens": answer.cache_read_tokens,
                "agents": answer.agents, "security": answer.security, "review": answer.review,
                "seconds": round(time.time() - t0, 1), "gold_ok": gold is not None,
            })
            print(f"  [{n}/{len(items)}] {'PASS' if official else 'FAIL'} {item['id']:<12} {answer.status:<13} "
                  f"calls={answer.llm_calls} tok={answer.input_tokens}/{answer.output_tokens}", flush=True)
    finally:
        config.MAX_RESULT_ROWS = old_rows
    return summarize(bench, rows, label, desc_usage)


def summarize(bench: str, rows: list[dict], label: str, desc_usage: dict) -> dict:
    pin, pout = config.PRICING.get(config.MODEL, (0, 0))

    def cost(tin, tcache, tout):
        return (tin - tcache) / 1e6 * pin + tcache / 1e6 * pin * 0.1 + tout / 1e6 * pout

    by_level = {}
    for lvl in sorted({r["level"] for r in rows if r["level"]}):
        sub = [r for r in rows if r["level"] == lvl]
        by_level[lvl] = f"{sum(r['pass'] for r in sub)}/{len(sub)}"
    tin = sum(r["input_tokens"] for r in rows)
    tcache = sum(r["cache_read_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    n = len(rows)
    summary = {
        "benchmark": bench, "label": label, "model": config.MODEL, "effort": config.EFFORT,
        "architecture": config.ARCHITECTURE, "n": n,
        "accuracy": round(100 * sum(r["pass"] for r in rows) / max(n, 1), 1),
        "lenient_accuracy": round(100 * sum(r["lenient_pass"] for r in rows) / max(n, 1), 1),
        "by_level": by_level,
        "llm_calls": sum(r["llm_calls"] for r in rows),
        "calls_per_question": round(sum(r["llm_calls"] for r in rows) / max(n, 1), 2),
        "input_tokens": tin, "cache_read_tokens": tcache, "output_tokens": tout,
        "est_cost_usd": round(cost(tin, tcache, tout), 4),
        "description_generation": {**desc_usage, "est_cost_usd": round(
            cost(desc_usage["input_tokens"], 0, desc_usage["output_tokens"]), 4)},
        "statuses": dict(Counter(r["status"] for r in rows)),
        "gold_missing": [r["id"] for r in rows if not r["gold_ok"]],
        "config": {"retrieval": config.RETRIEVAL, "schema_mode": config.SCHEMA_MODE,
                   "value_hints": config.VALUE_HINTS, "descriptions_in_prompt": config.DESCRIPTIONS_IN_PROMPT,
                   "verify": config.VERIFY, "schema_agent": config.SCHEMA_AGENT,
                   "multi_validator": config.MULTI_VALIDATOR},
        "rows": rows,
    }
    out = config.ROOT / "evals" / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}_bench-{bench}_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["path"] = str(out)
    return summary


def gold_tables(sql: str) -> set[str]:
    """Tables a gold query reads (for the free retrieval-recall check)."""
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes}
