"""Free (no LLM) retrieval check: does the retriever return every table each gold query uses?

  python scripts/retrieval_recall.py [--strip-prefix] [--k 6] [--cap 14]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

from nl_to_sql import config, descriptions  # noqa: E402
from nl_to_sql.db import Database  # noqa: E402
from nl_to_sql.evaluation import load_dataset  # noqa: E402
from nl_to_sql.retrieval import TableIndex  # noqa: E402


def gold_tables(sql: str) -> set[str]:
    tree = sqlglot.parse_one(sql, read="sqlite")
    ctes = {c.alias for c in tree.find_all(exp.CTE)}
    return {t.name for t in tree.find_all(exp.Table) if t.name not in ctes}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strip-prefix", action="store_true", help="drop the 'Music store:' style hint")
    p.add_argument("--k", type=int, default=config.RETRIEVE_TOP_K)
    p.add_argument("--cap", type=int, default=config.RETRIEVE_MAX_TABLES)
    p.add_argument("--retrieval", choices=["bm25", "vector", "hybrid"], default=config.RETRIEVAL)
    p.add_argument("--no-descriptions", action="store_true")
    args = p.parse_args()
    config.RETRIEVE_TOP_K, config.RETRIEVE_MAX_TABLES = args.k, args.cap
    config.RETRIEVAL = args.retrieval

    db = Database("enterprise")
    if not args.no_descriptions:
        db.descriptions = descriptions.load(db)  # never generates here: this script stays free
    idx = TableIndex(db)
    items = [i for i in load_dataset("enterprise") if i.get("gold_sql")]
    covered, sizes = 0, []
    for it in items:
        q = it["question"].split(": ", 1)[1] if args.strip_prefix else it["question"]
        got = set(idx.select(q))
        sizes.append(len(got))
        missing = gold_tables(it["gold_sql"]) - got
        covered += not missing
        if missing:
            print(f"  MISS {it['id']}: {sorted(missing)}")
    print(f"recall {covered}/{len(items)} · avg {sum(sizes) / len(sizes):.1f} of "
          f"{len(idx.lines)} tables · k={args.k} cap={args.cap} · {args.retrieval} · "
          f"descriptions={'yes' if db.descriptions else 'no'}")


if __name__ == "__main__":
    main()
