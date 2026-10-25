"""Benchmark metrics (offline). Mirrors the official comparison rules."""
import pandas as pd
import pytest

from nl_to_sql import benchmarks
from nl_to_sql.benchmarks import bird_ex, spider2_match, spider_ex
from nl_to_sql.chain import Answer
from nl_to_sql.db import QueryResult


def R(cols, rows):
    return QueryResult(columns=cols, rows=rows)


def test_spider_ex_column_permutation_and_order():
    gold = R(["a", "b"], [("x", 1), ("y", 2)])
    assert spider_ex(gold, R(["b", "a"], [(1, "x"), (2, "y")]), order_matters=False)
    assert spider_ex(gold, R(["a", "b"], [("y", 2), ("x", 1)]), order_matters=False)
    assert not spider_ex(gold, R(["a", "b"], [("y", 2), ("x", 1)]), order_matters=True)
    assert not spider_ex(gold, R(["a"], [("x",), ("y",)]), order_matters=False)        # missing column
    assert not spider_ex(gold, R(["a", "b"], [("x", 1), ("x", 1)]), order_matters=False)  # multiset differs


def test_spider_ex_is_exact_unlike_lenient_grader():
    assert not spider_ex(R(["v"], [(469.58,)]), R(["v"], [(469.58000000000004,)]), False)


def test_bird_ex_is_set_equality():
    gold = R(["n"], [(1,), (2,)])
    assert bird_ex(gold, R(["n"], [(2,), (1,), (1,)]))   # sets: duplicates don't matter
    assert not bird_ex(gold, R(["n", "m"], [(1, 0), (2, 0)]))  # extra column changes the rows


def test_spider2_match_condition_cols_tolerance_and_alternatives():
    gold_a = pd.DataFrame({"name": ["x", "y"], "val": [1.004, 2.0]})
    gold_b = pd.DataFrame({"val": [9.0]})
    pred = R(["label", "name", "val"], [("L1", "x", 1.0), ("L2", "y", 2.0)])
    assert spider2_match(pred, [gold_a], [], ignore_order=False)                 # extra pred column ok, 1e-2 tol
    assert spider2_match(pred, [gold_b, gold_a], [[0], [1]], ignore_order=False)  # second gold matches on col 1
    assert not spider2_match(R(["val"], [(2.0,), (1.0,)]), [gold_a], [1], ignore_order=False)
    assert spider2_match(R(["val"], [(2.0,), (1.0,)]), [gold_a], [1], ignore_order=True)


def test_grade_requires_ok_status():
    item = {"bench": "bird", "gold_sql": "SELECT 1"}
    gold = R(["n"], [(1,)])
    assert benchmarks.grade(item, Answer(question="q", status="ok", result=R(["n"], [(1,)])), gold) == (True, True)
    assert benchmarks.grade(item, Answer(question="q", status="error"), gold) == (False, False)


def test_items_and_gold_load_when_prepared():
    path = benchmarks.BENCH_DIR / "spider" / "items.json"
    if not path.exists():
        pytest.skip("run scripts/prepare_benchmarks.py first")
    a = benchmarks.load_items("spider", sample=10, seed=1)
    b = benchmarks.load_items("spider", sample=10, seed=1)
    assert [i["id"] for i in a] == [i["id"] for i in b] and len(a) == 10   # deterministic sample
    assert benchmarks.gold_result(a[0]) is not None
