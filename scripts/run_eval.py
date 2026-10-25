"""Run the eval set.

  python scripts/run_eval.py --check-gold          # free: run gold SQL only
  python scripts/run_eval.py --runner v2           # live: costs API calls
  python scripts/run_eval.py --runner v2 --ids ch10 nw11
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nl_to_sql import config  # noqa: E402
from nl_to_sql.agent import Pending, SQLAgent  # noqa: E402
from nl_to_sql.chain import NLToSQL  # noqa: E402
from nl_to_sql.db import Database  # noqa: E402
from nl_to_sql.multi_agent import MultiAgentSQL  # noqa: E402
from nl_to_sql.evaluation import gold_results, load_dataset, run  # noqa: E402


class AgentAdapter:
    def __init__(self, agent: SQLAgent):
        self.agent = agent

    def ask(self, question: str):
        out = self.agent.start(question)
        while isinstance(out, Pending):  # evals never write
            out = self.agent.resume(False)
        return out


class ChainAdapter:
    def __init__(self, chain: NLToSQL):
        self.chain = chain

    def ask(self, question: str):
        return self.chain.ask(question, approve=lambda sql, kind: False)


def _agent_without_hints(db):
    config.VALUE_HINTS = False  # approximates v2 (v3 also tightened the prompt)
    return AgentAdapter(SQLAgent(db))


RUNNERS = {
    "multi": lambda db: AgentAdapter(MultiAgentSQL(db)),
    "v1": lambda db: ChainAdapter(NLToSQL(db)),
    "v2": _agent_without_hints,
    "v3": lambda db: AgentAdapter(SQLAgent(db)),
}


def check_gold(items: list[dict]) -> None:
    for item in items:
        results = gold_results(Database(item["db"]), item)
        if not results:
            print(f"{item['id']}: expect={item['expect']}")
        for i, res in enumerate(results):
            print(f"{item['id']}{'' if i == 0 else f' (alt {i})'}: {len(res.rows)} rows {res.rows[:3]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runner", choices=list(RUNNERS), default="v3")
    p.add_argument("--db")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--check-gold", action="store_true")
    p.add_argument("--label", help="name for the results file")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config for this run, e.g. --set FEW_SHOT=true VALUE_HINTS=false")
    args = p.parse_args()
    for kv in args.set:
        key, val = kv.split("=", 1)
        old = getattr(config, key)
        setattr(config, key, val.lower() in ("1", "true", "yes") if isinstance(old, bool)
                else type(old)(val))

    items = load_dataset(args.db, args.ids)
    if args.check_gold:
        return check_gold(items)

    label = args.label or args.runner
    print(f"Running {len(items)} questions · runner={args.runner} · model={config.MODEL} · "
          f"hints={config.VALUE_HINTS} few_shot={config.FEW_SHOT}")
    summary = run(items, RUNNERS[args.runner], label)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
