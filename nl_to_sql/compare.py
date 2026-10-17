"""Compare query results the way a person would: same data, any presentation.

- row order is ignored (LIMIT already fixes *which* rows for top-N questions),
- column order and names are ignored, and extra columns on the right-hand side are allowed,
- floats are compared at 2 decimals,
- a "First Last" column can be matched by two adjacent name columns.

Used by the eval grader and by self-consistency voting in the agent.
"""
from collections import Counter
from decimal import Decimal
from itertools import combinations

from .db import QueryResult


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
