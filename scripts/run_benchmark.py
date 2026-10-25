"""Run Spider 1.0 / BIRD / Spider 2.0-lite (local SQLite) on a seeded sample.

  python scripts/run_benchmark.py spider --arch single              # 100 dev questions
  python scripts/run_benchmark.py bird --arch multi --sample 50
  python scripts/run_benchmark.py spider2 --arch multi --sample 30
  python scripts/run_benchmark.py spider --check-gold                # free: run gold SQL only
  python scripts/run_benchmark.py bird --set MULTI_VALIDATOR=flags RETRIEVAL=bm25

Live runs cost API calls; the summary prints calls, tokens and estimated cost.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_anthropic import ChatAnthropic  # noqa: E402

from nl_to_sql import benchmarks, config  # noqa: E402
from nl_to_sql.multi_agent import make_agent  # noqa: E402
from scripts.run_eval import AgentAdapter, apply_overrides  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bench", choices=benchmarks.BENCHMARKS)
    p.add_argument("--arch", choices=["single", "multi"], default=config.ARCHITECTURE)
    p.add_argument("--sample", type=int, help="questions to run (seeded); default per benchmark")
    p.add_argument("--all", action="store_true", help="run every question (expensive)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--label")
    p.add_argument("--check-gold", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    apply_overrides(args.set)
    config.ARCHITECTURE = args.arch

    sample = None if args.all else (args.sample or benchmarks.DEFAULT_SAMPLE[args.bench])
    items = benchmarks.load_items(args.bench, sample, args.seed, args.ids)
    if args.check_gold:
        missing = 0
        for it in items:
            g = benchmarks.gold_result(it)
            missing += g is None
            size = len(g) if isinstance(g, list) else (len(g.rows) if g else "-")
            print(f"{it['id']}: {'MISSING' if g is None else 'ok'} ({size})")
        print(f"{len(items) - missing}/{len(items)} gold results available")
        return

    label = args.label or args.arch
    print(f"{args.bench}: {len(items)} questions · arch={args.arch} · model={config.MODEL} · "
          f"retrieval={config.RETRIEVAL}")
    describer = ChatAnthropic(model=config.MODEL, max_tokens=config.MAX_TOKENS,
                              output_config={"effort": config.EFFORT})
    summary = benchmarks.run(args.bench, items, lambda db: AgentAdapter(make_agent(db)), label, describer)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
