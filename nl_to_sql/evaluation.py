"""Execution-accuracy evals: run each question, compare the result set to the gold query's.

Matching is deliberately lenient about presentation and strict about data:
- row order is ignored (LIMIT already fixes *which* rows for top-N questions),
- column order and names are ignored, and extra predicted columns are allowed,
- floats are compared at 2 decimals,
- a gold "First Last" column can be matched by two adjacent predicted name columns.
"""
import json
import time
from collections import Counter
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Callable

from . import config
from .db import Database, QueryResult

DATASET = config.ROOT / "evals" / "dataset.json"
RESULTS_DIR = config.ROOT / "evals" / "results"


def load_dataset(db: str | None = None, ids: list[str] | None = None) -> list[dict]:
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    return [i for i in items if (not db or i["db"] == db) and (not ids or i["id"] in ids)]


# ---- result matching -------------------------------------------------------

def _norm(v):
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, float):
        v = round(v, 2)
        return int(v) if v.is_integer() else v
    if isinstance(v, str):
        s = v.strip()
        try:  # numbers stored as text, e.g. "3.0"
            return _norm(float(s)) if s.replace(".", "", 1).lstrip("-").isdigit() else s
        except ValueError:
            return s
    return v


def _columns(rows: list[tuple], n: int) -> list[list]:
    return [[_norm(r[i]) for r in rows] for i in range(n)]


def results_match(gold: QueryResult, pred: QueryResult) -> bool:
    if len(gold.rows) != len(pred.rows):
        return False
    if not gold.rows:
        return True
    gcols = _columns(gold.rows, len(gold.columns))
    pcols = _columns(pred.rows, len(pred.columns))
    # Derived candidates: "First Last" from pairs of text columns.
    for a, b in combinations(range(len(pcols)), 2):
        if all(isinstance(x, str) for x in pcols[a] + pcols[b]):
            pcols.append([f"{x} {y}" for x, y in zip(pcols[a], pcols[b])])
            pcols.append([f"{y} {x}" for x, y in zip(pcols[a], pcols[b])])

    gold_rows = Counter(zip(*gcols))
    candidates = [[j for j, pc in enumerate(pcols) if Counter(pc) == Counter(gc)] for gc in gcols]

    def assign(i: int, chosen: list[int]) -> bool:
        if i == len(gcols):
            return Counter(zip(*(pcols[j] for j in chosen))) == gold_rows
        return any(assign(i + 1, chosen + [j]) for j in candidates[i] if j not in chosen)

    return assign(0, [])


def grade_against(golds: list[QueryResult], pred: QueryResult) -> bool:
    return any(results_match(g, pred) for g in golds)


def grade(item: dict, answer, golds: list[QueryResult]) -> bool:
    if item.get("expect") == "cannot_answer":
        # Right outcome: say it can't be answered, or return no rows (with an explanation).
        return answer.status == "cannot_answer" or (
            answer.status == "ok" and (answer.result is None or not answer.result.rows))
    if answer.status != "ok" or answer.result is None or not golds:
        return False
    return grade_against(golds, answer.result)


# ---- runner ----------------------------------------------------------------

def gold_results(db: Database, item: dict) -> list[QueryResult]:
    """The gold query plus any alt_gold_sql (e.g. a variant that returns all tied rows)."""
    sqls = ([item["gold_sql"]] if item.get("gold_sql") else []) + item.get("alt_gold_sql", [])
    return [db.execute(sql, commit=False) for sql in sqls]


def run(items: list[dict], make_runner: Callable[[Database], object], label: str) -> dict:
    """make_runner(db) returns an object with .ask(question) -> Answer (writes auto-declined)."""
    dbs: dict[str, Database] = {}
    runners: dict[str, object] = {}
    rows = []
    for item in items:
        name = item["db"]
        if name not in dbs:
            dbs[name] = Database(name)
            runners[name] = make_runner(dbs[name])
        t0 = time.time()
        answer = runners[name].ask(item["question"])
        secs = time.time() - t0
        passed = grade(item, answer, gold_results(dbs[name], item))
        rows.append({
            "id": item["id"], "level": item["level"], "pass": passed,
            "status": answer.status, "sql": answer.sql, "note": answer.note, "error": answer.error,
            "llm_calls": answer.llm_calls, "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens, "seconds": round(secs, 1),
        })
        print(f"  {'PASS' if passed else 'FAIL'}  {item['id']}  {answer.status:<13} "
              f"calls={answer.llm_calls} tok={answer.input_tokens}/{answer.output_tokens}")
    return summarize(rows, label)


def summarize(rows: list[dict], label: str) -> dict:
    by_level = {}
    for lvl in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["level"] == lvl]
        if sub:
            by_level[lvl] = f"{sum(r['pass'] for r in sub)}/{len(sub)}"
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    pin, pout = config.PRICING.get(config.MODEL, (0, 0))
    summary = {
        "label": label, "model": config.MODEL, "effort": config.EFFORT,
        "features": {"value_hints": config.VALUE_HINTS, "few_shot": config.FEW_SHOT},
        "accuracy": f"{sum(r['pass'] for r in rows)}/{len(rows)}",
        "by_level": by_level,
        "llm_calls": sum(r["llm_calls"] for r in rows),
        "input_tokens": tin, "output_tokens": tout,
        "est_cost_usd": round(tin / 1e6 * pin + tout / 1e6 * pout, 4),
        "failed": [r["id"] for r in rows if not r["pass"]],
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["path"] = str(out)
    return summary
