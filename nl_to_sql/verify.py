"""v5: answer checking.

red_flags() is free (plain Python over the result). The LLM review in the agent's
"verify" node only runs when a flag fires (VERIFY="flags", the default) or on every
answer (VERIFY="always"), so most questions still cost a single model call.
"""
import re

from .db import QueryResult

_SUPERLATIVE = re.compile(
    r"\b(most|least|highest|lowest|largest|smallest|biggest|best|worst|longest|shortest|greatest|fewest|top)\b", re.I)
_MANY = re.compile(r"\b(each|every|per|all|list|top \d+|ranking|by year|by month)\b", re.I)


def red_flags(question: str, res: QueryResult) -> list[str]:
    """Cheap signs that a result may not answer the question. Empty list = looks fine."""
    flags = []
    if not res.columns:
        return flags
    rows = res.rows
    if any(v is None for r in rows for v in r):
        flags.append("the result contains NULL values")
    if len(rows) > 1 and _SUPERLATIVE.search(question) and not _MANY.search(question):
        # Several rows for a single-answer question is fine only if they're tied.
        last = {r[-1] for r in rows}
        if len(last) > 1:
            flags.append(f"the question asks for a single top/bottom answer but {len(rows)} rows came back")
    if len(rows) != len(set(rows)):
        flags.append("the result has duplicate rows (possible join fan-out)")
    if re.search(r"\bpercent|percentage|share\b", question, re.I):
        nums = [v for r in rows for v in r if isinstance(v, (int, float))]
        if any(v < 0 or v > 100 for v in nums if v is not None) and not re.search(r"change|growth|increase", question, re.I):
            flags.append("a percentage outside 0-100")
    return flags


REVIEW_PROMPT = """You are reviewing whether a SQL result correctly and exactly answers a question.

Question: {question}

SQL:
{sql}

Result ({n} rows, first {shown} shown):
{table}

Automatic checks noticed: {flags}

Judge only against the question's wording. Extra helper columns are fine. A tie returning several rows is fine.
Reply with exactly one line: OK, or ISSUE: <what is wrong and how to fix it>."""


def review_prompt(question: str, sql: str, res: QueryResult, flags: list[str], max_rows: int = 15) -> str:
    shown = res.rows[:max_rows]
    table = "\n".join(["\t".join(res.columns)] + ["\t".join(map(str, r)) for r in shown])
    return REVIEW_PROMPT.format(question=question, sql=sql, n=len(res.rows), shown=len(shown),
                                table=table, flags="; ".join(flags) or "nothing")


def parse_review(text: str) -> str | None:
    """None if the reviewer accepted the result, otherwise the issue text."""
    t = text.strip()
    if t.upper().startswith("OK"):
        return None
    return t.split(":", 1)[1].strip() if t.upper().startswith("ISSUE") and ":" in t else t
