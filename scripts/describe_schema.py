"""Generate (or regenerate) table/column descriptions for a database. One-time, cached.

  python scripts/describe_schema.py enterprise [--force]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_anthropic import ChatAnthropic  # noqa: E402

from nl_to_sql import config, descriptions  # noqa: E402
from nl_to_sql.db import Database  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("db", choices=list(config.DATABASES))
    p.add_argument("--force", action="store_true", help="regenerate even if descriptions exist")
    args = p.parse_args()
    db = Database(args.db)
    if descriptions.load(db) and not args.force:
        print(f"'{args.db}' already has descriptions (use --force to regenerate)")
        return
    llm = ChatAnthropic(model=config.MODEL, max_tokens=config.MAX_TOKENS,
                        output_config={"effort": config.EFFORT})
    tables, usage = descriptions.generate(db, llm)
    path = descriptions.save(db, tables, "llm")
    pin, pout = config.PRICING.get(config.MODEL, (0, 0))
    cost = usage["input_tokens"] / 1e6 * pin + usage["output_tokens"] / 1e6 * pout
    described = sum(1 for t in tables.values() if t["description"])
    print(f"wrote {path.name}: {described}/{len(tables)} tables described · {usage['calls']} calls · "
          f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out · ~${cost:.4f}")


if __name__ == "__main__":
    main()
