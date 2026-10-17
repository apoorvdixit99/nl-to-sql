from nl_to_sql.chain import Answer
from nl_to_sql.db import QueryResult
from nl_to_sql.compare import results_match
from nl_to_sql.evaluation import grade


def R(cols, rows):
    return QueryResult(columns=cols, rows=rows)


def test_row_and_column_order_ignored():
    gold = R(["a", "b"], [("x", 1), ("y", 2)])
    assert results_match(gold, R(["b", "a"], [(2, "y"), (1, "x")]))


def test_extra_predicted_columns_allowed():
    gold = R(["name"], [("Iron Maiden",)])
    assert results_match(gold, R(["id", "name", "n"], [(90, "Iron Maiden", 21)]))


def test_float_rounding_and_int_float_equivalence():
    assert results_match(R(["v"], [(469.58,)]), R(["v"], [(469.58000000000004,)]))
    assert results_match(R(["v"], [(81,)]), R(["v"], [(81.0,)]))


def test_split_name_columns_match_concatenated_gold():
    gold = R(["full"], [("Jane Peacock",)])
    assert results_match(gold, R(["first", "last"], [("Jane", "Peacock")]))


def test_wrong_values_or_row_count_fail():
    gold = R(["n"], [(8,)])
    assert not results_match(gold, R(["n"], [(0,)]))
    assert not results_match(gold, R(["n"], [(8,), (9,)]))


def test_columns_must_stay_aligned_across_rows():
    gold = R(["a", "b"], [("x", 1), ("y", 2)])
    assert not results_match(gold, R(["a", "b"], [("x", 2), ("y", 1)]))


def test_cannot_answer_expectation():
    item = {"expect": "cannot_answer"}
    assert grade(item, Answer(question="q", status="cannot_answer"), [])
    assert grade(item, Answer(question="q", status="ok", result=R(["n"], [])), [])
    assert not grade(item, Answer(question="q", status="ok", result=R(["n"], [(3,)])), [])


def test_alt_gold_accepts_tied_answers():
    from nl_to_sql.evaluation import grade_against
    golds = [R(["name"], [("Music",)]), R(["name"], [("Music",), ("Music",)])]
    assert grade_against(golds, R(["name", "n"], [("Music", 198), ("Music", 198)]))


def test_enterprise_gold_matches_original_gold():
    """Remapped gold SQL on the 274-table testbed returns the same rows as on the source DBs."""
    import pytest
    from nl_to_sql import config
    from nl_to_sql.db import Database
    from nl_to_sql.evaluation import gold_results, load_dataset
    if not (config.ROOT / "data" / "enterprise.db").exists():
        pytest.skip("run scripts/build_enterprise_db.py first")
    ent = Database("enterprise")
    originals = {i["id"]: i for i in load_dataset()}
    for item in load_dataset("enterprise"):
        orig = originals[item["id"].removeprefix("e-")]
        a = gold_results(Database(orig["db"]), orig)
        b = gold_results(ent, item)
        assert all(results_match(x, y) for x, y in zip(a, b)) and len(a) == len(b), item["id"]
