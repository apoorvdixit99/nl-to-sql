"""Free (no LLM calls) table-ranking check on Spider / BIRD / Spider 2.0 gold queries.

For each question, rank the database's tables with BM25, vector search and hybrid, and
report how often every gold table is in the top k (before FK expansion). Uses whatever
descriptions already exist for each database; never generates new ones.

  python scripts/retrieval_benchmark.py [--sample 200]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nl_to_sql import benchmarks, config, descriptions  # noqa: E402
from nl_to_sql.db import Database  # noqa: E402
from nl_to_sql.retrieval import TableIndex  # noqa: E402


def gold_sql_tables(item: dict) -> set[str]:
    if item.get("gold_sql"):
        return benchmarks.gold_tables(item["gold_sql"])
    p = benchmarks.BENCH_DIR / "spider2_raw" / "repo" / "spider2-lite" / "evaluation_suite" / "gold" / "sql" / f"{item['id']}.sql"
    return benchmarks.gold_tables(p.read_text(encoding="utf-8")) if p.exists() else set()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--benches", nargs="*", default=list(benchmarks.BENCHMARKS))
    args = p.parse_args()
    report = {}
    for bench in args.benches:
        items = [i for i in benchmarks.load_items(bench, args.sample) if gold_sql_tables(i)]
        indexes = {}
        stats = {m: {3: 0, 5: 0} for m in ("bm25", "vector", "hybrid")}
        with_desc = set()
        for it in items:
            if it["db"] not in indexes:
                db = Database(it["db"])
                db.descriptions = descriptions.load(db)
                if db.descriptions:
                    with_desc.add(it["db"])
                indexes[it["db"]] = TableIndex(db)
            idx = indexes[it["db"]]
            gold = gold_sql_tables(it)
            q = benchmarks.agent_question(it)
            for mode in stats:
                config.RETRIEVAL = mode
                ranked = [t.lower() for t in idx.ranked(q)]
                for k in (3, 5):
                    stats[mode][k] += gold <= set(ranked[:k])
        n = len(items)
        report[bench] = {"questions": n, "dbs_with_descriptions": f"{len(with_desc)}/{len(indexes)}",
                         **{m: {f"top{k}": round(100 * v / n, 1) for k, v in s.items()} for m, s in stats.items()}}
        print(bench, json.dumps(report[bench]))
    out = config.ROOT / "evals" / "results" / "retrieval_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
