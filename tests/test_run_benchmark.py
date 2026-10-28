"""
The benchmark harness stops the whole run on exhausted credit.

Free and offline: the graph, dev set and gold results are replaced, so the
real run_benchmark() loop -- checkpointing, resume, scoring -- runs with no
LLM, database or network. The case under test is the night the OpenAI balance
ran out at question 93: the harness kept going, scoring each remaining
question as wrong after ~143s of retries.
"""

import csv

import pytest

from eval import run_benchmark as rb
from src.llm_providers import QuotaExhaustedError

N = 6


class FakeGraph:
    """Answers correctly, except it raises `fail_with` on the question indices in `fail_at`."""

    def __init__(self, fail_at=(), fail_with=None):
        self.fail_at, self.fail_with, self.invoked = set(fail_at), fail_with, []

    def invoke(self, state):
        idx = int(state["question"].split("#")[1])
        self.invoked.append(idx)
        if idx in self.fail_at:
            raise self.fail_with
        return {"sql_query": "SELECT 1", "execution_result": [(1,)], "trace": [], "retry_count": 0, "success": True}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    results = tmp_path / "results.csv"
    monkeypatch.setattr(rb, "RESULTS_PATH", str(results))
    monkeypatch.setattr(rb, "METADATA_PATH", str(tmp_path / "meta.json"))
    monkeypatch.setattr(rb, "load_dev_set", lambda path: [
        {"question_id": 100 + i, "db_id": "db", "question": f"question #{i}", "SQL": "SELECT 1",
         "evidence": "", "difficulty": "simple"}
        for i in range(N)
    ])
    monkeypatch.setattr(rb, "get_gold_result", lambda db_id, sql: [(1,)])

    def run(graph, resume=False):
        monkeypatch.setattr(rb, "build_graph", lambda: graph)
        rb.run_benchmark(resume=resume)

    def rows():
        if not results.exists():
            return []
        with open(results, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    return run, rows


def test_quota_exhaustion_stops_the_run(harness):
    run, rows = harness
    graph = FakeGraph(fail_at={3}, fail_with=QuotaExhaustedError("no credit remaining"))

    with pytest.raises(SystemExit) as exc:
        run(graph)

    assert exc.value.code == rb.EXIT_QUOTA_EXHAUSTED
    assert graph.invoked == [0, 1, 2, 3]  # stopped at once -- 4 and 5 never attempted
    saved = rows()
    assert [r["question_id"] for r in saved] == ["100", "101", "102"]  # the failing question is NOT recorded
    assert all(r["correct"] == "True" for r in saved)


def test_resume_after_topup_reruns_the_stopped_question(harness):
    run, rows = harness
    with pytest.raises(SystemExit):
        run(FakeGraph(fail_at={3}, fail_with=QuotaExhaustedError("no credit remaining")))

    resumed = FakeGraph()  # credit restored
    run(resumed, resume=True)

    assert resumed.invoked == [3, 4, 5]  # picks up exactly where it stopped
    saved = rows()
    assert [r["question_id"] for r in saved] == [str(100 + i) for i in range(N)]
    assert all(r["correct"] == "True" for r in saved) and all(not r["fatal_error"] for r in saved)


def test_quota_message_without_the_type_still_stops(harness):
    # Detection also matches the raw billing code, in case an error reaches
    # the harness without passing through _with_backoff's conversion.
    run, rows = harness
    raw = RuntimeError("Error code: 429 - {'error': {'code': 'credit_balance_exhausted'}}")
    with pytest.raises(SystemExit):
        run(FakeGraph(fail_at={1}, fail_with=raw))
    assert len(rows()) == 1


def test_an_ordinary_crash_is_still_a_counted_failure(harness):
    # Unchanged behaviour: any other exception is recorded as a fatal row,
    # scored incorrect, and the run carries on.
    run, rows = harness
    graph = FakeGraph(fail_at={2}, fail_with=RuntimeError("something else broke"))
    run(graph)
    saved = rows()
    assert len(saved) == N
    assert graph.invoked == list(range(N))
    assert saved[2]["correct"] == "False" and "something else broke" in saved[2]["fatal_error"]


# --- cost columns ---------------------------------------------------------------
#
# Each row records what the question cost. Unknown is an EMPTY cell, never 0, and
# every file written before these columns existed must still read and resume.

OLD_COLUMNS = [
    "index", "question_id", "db_id", "question", "evidence", "gold_sql", "predicted_sql",
    "correct", "difficulty", "retries", "error_classes_hit", "fatal_error",
    "schema_context_chars", "seconds",
]


# Everything after the original columns and before split.
COST_CELLS = rb.FIELDNAMES[len(OLD_COLUMNS):-1]


def _run(node, seconds, calls=0, prompt=0, completion=0, reasoning=0, cached=0, cost=0.0):
    return {
        "node": node, "retry_count": 0, "seconds": seconds, "llm_calls": calls,
        "prompt_tokens": prompt, "cached_prompt_tokens": cached,
        "completion_tokens": completion, "reasoning_tokens": reasoning, "cost_usd": cost,
    }


class MeteredGraph:
    """Answers every question correctly and reports the node_runs it is given."""

    def __init__(self, node_runs):
        self.node_runs = node_runs

    def invoke(self, state):
        return {
            "sql_query": "SELECT 1", "execution_result": [(1,)], "trace": [], "retry_count": 1,
            "success": True, "node_runs": self.node_runs,
        }


CORRECTION_LOOP = [
    _run("schema_retriever", 0.25),
    _run("query_planner", 2.0, calls=1, prompt=1000, completion=200, reasoning=100, cost=0.00065),
    _run("sql_generator", 1.5, calls=1, prompt=2000, completion=50, cached=500, cost=0.0006),
    _run("executor", 0.125),
    _run("classify_error", 0.0),
    _run("query_planner", 3.0, calls=1, prompt=1100, completion=210, reasoning=100, cost=0.0007),
    _run("sql_generator", 1.0, calls=1, prompt=2100, completion=40, cost=0.0005),
    _run("executor", 0.125),
]


def test_a_row_records_tokens_cost_and_per_node_latency(harness):
    run, rows = harness

    run(MeteredGraph(CORRECTION_LOOP))

    row = rows()[0]
    assert row["llm_calls"] == "4"
    assert row["prompt_tokens"] == str(1000 + 2000 + 1100 + 2100)
    assert row["cached_prompt_tokens"] == "500"
    assert row["completion_tokens"] == str(200 + 50 + 210 + 40)
    assert row["reasoning_tokens"] == "200"
    assert float(row["est_cost_usd"]) == pytest.approx(0.00065 + 0.0006 + 0.0007 + 0.0005)
    # A node the loop re-entered has its runs summed.
    assert float(row["planner_seconds"]) == pytest.approx(5.0)
    assert float(row["generator_seconds"]) == pytest.approx(2.5)
    assert float(row["executor_seconds"]) == pytest.approx(0.25)
    assert float(row["retriever_seconds"]) == pytest.approx(0.25)
    assert float(row["classifier_seconds"]) == 0.0


def test_a_question_with_no_llm_calls_costs_exactly_zero(harness):
    run, rows = harness

    run(MeteredGraph([_run("schema_retriever", 0.1)]))

    row = rows()[0]
    assert row["llm_calls"] == "0"
    assert row["est_cost_usd"] == "0.0"


def test_an_unpriced_call_leaves_the_cost_empty_but_keeps_the_tokens(harness):
    run, rows = harness

    run(MeteredGraph([_run("query_planner", 1.0, calls=1, prompt=900, completion=90, cost=None)]))

    row = rows()[0]
    assert row["prompt_tokens"] == "900"
    assert row["est_cost_usd"] == ""


def test_a_crashed_question_leaves_every_cost_cell_empty_not_zero(harness):
    run, rows = harness

    run(FakeGraph(fail_at={2}, fail_with=RuntimeError("something else broke")))

    crashed = rows()[2]
    assert crashed["correct"] == "False"
    for column in COST_CELLS:
        assert crashed[column] == "", column


def test_a_graph_state_without_node_runs_gives_empty_cells(harness):
    # What every older fake and any pre-metering graph returns.
    run, rows = harness

    run(FakeGraph())

    assert all(rows()[0][column] == "" for column in COST_CELLS)


def test_the_original_columns_keep_their_names_and_order():
    assert rb.FIELDNAMES[: len(OLD_COLUMNS)] == OLD_COLUMNS
    for committed in ("eval/results_500_baseline.csv", "eval/results_150_with_evidence.csv"):
        with open(committed, encoding="utf-8", newline="") as f:
            assert next(csv.reader(f)) == OLD_COLUMNS, committed


def test_resuming_a_file_without_the_cost_columns_keeps_them(harness, tmp_path):
    # A run started on the old format, interrupted, then resumed on the new code.
    run, rows = harness
    old_rows = [
        {"index": i, "question_id": 100 + i, "db_id": "db", "question": f"question #{i}", "evidence": "",
         "gold_sql": "SELECT 1", "predicted_sql": "SELECT 1", "correct": "True", "difficulty": "simple",
         "retries": 0, "error_classes_hit": "", "fatal_error": "", "schema_context_chars": 10, "seconds": 1.5}
        for i in range(2)
    ]
    with open(rb.RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OLD_COLUMNS)
        writer.writeheader()
        writer.writerows(old_rows)

    run(MeteredGraph(CORRECTION_LOOP), resume=True)

    saved = rows()
    assert [r["question_id"] for r in saved] == [str(100 + i) for i in range(N)]
    with open(rb.RESULTS_PATH, encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == rb.FIELDNAMES
    assert all(saved[i]["llm_calls"] == "" for i in range(2))  # old rows: unknown, not zero
    assert all(saved[i]["llm_calls"] == "4" for i in range(2, N))
    assert saved[0]["correct"] == "True" and saved[0]["seconds"] == "1.5"  # old values untouched


def test_run_metadata_records_the_prices_the_cost_used(harness):
    import json

    from src.pricing import PRICING_AS_OF

    run, _ = harness
    run(MeteredGraph(CORRECTION_LOOP))

    with open(rb.METADATA_PATH, encoding="utf-8") as f:
        assert json.load(f)["pricing_as_of"] == PRICING_AS_OF


def test_the_summary_averages_only_over_questions_with_a_known_cost(capsys):
    results = [
        {"correct": "True", "difficulty": "simple", "db_id": "a", "est_cost_usd": 0.5},
        {"correct": "False", "difficulty": "simple", "db_id": "a", "est_cost_usd": ""},   # crashed
        {"correct": "True", "difficulty": "simple", "db_id": "a", "est_cost_usd": "1.5"},  # read back from CSV
        {"correct": "True", "difficulty": "simple", "db_id": "a"},                          # older file
    ]

    rb.print_summary(results, total_elapsed=10.0, questions_run_now=0)

    out = capsys.readouterr().out
    assert "Estimated cost: $2.0000 over 2 of 4 questions with a known cost ($1.0000/question)" in out


def test_the_summary_prints_no_cost_line_when_nothing_has_one(capsys):
    results = [{"correct": "True", "difficulty": "simple", "db_id": "a", "est_cost_usd": ""}]

    rb.print_summary(results, total_elapsed=1.0, questions_run_now=0)

    assert "Estimated cost" not in capsys.readouterr().out


# --- splits ---------------------------------------------------------------------
#
# A run covers one committed split, chosen by question_id -- never "the first N
# questions of dev.json", which covered 4 of 11 databases.


@pytest.fixture
def splits(tmp_path, monkeypatch):
    """Calibration = question_ids 101 and 104; held-out = the other four of the harness's six."""
    import json

    for name, ids in (("calibration", [101, 104]), ("heldout", [100, 102, 103, 105])):
        (tmp_path / f"{name}_ids.json").write_text(json.dumps({"split": name, "question_ids": ids}), encoding="utf-8")
    monkeypatch.setattr(rb, "SPLITS_DIR", str(tmp_path))


def test_a_run_covers_only_its_split(harness, splits, monkeypatch):
    run, rows = harness
    graph = FakeGraph()
    monkeypatch.setattr(rb, "build_graph", lambda: graph)

    rb.run_benchmark(split="calibration")

    assert graph.invoked == [1, 4]
    assert [r["question_id"] for r in rows()] == ["101", "104"]


def test_the_default_split_is_every_question(harness):
    run, rows = harness
    run(FakeGraph())
    assert len(rows()) == N


def test_limit_caps_questions_inside_the_split(harness, splits, monkeypatch):
    run, rows = harness
    monkeypatch.setattr(rb, "build_graph", lambda: FakeGraph())

    rb.run_benchmark(limit=2, split="heldout")

    assert [r["question_id"] for r in rows()] == ["100", "102"]  # first two held-out, not first two overall


def test_an_unknown_split_stops_before_any_question_runs(harness, splits, monkeypatch):
    run, rows = harness
    graph = FakeGraph()
    monkeypatch.setattr(rb, "build_graph", lambda: graph)

    with pytest.raises(SystemExit) as exc:
        rb.run_benchmark(split="holdout")

    assert "holdout" in str(exc.value)
    assert graph.invoked == [] and rows() == []


def test_a_duplicated_question_id_keeps_both_rows_in_its_split(harness, splits, monkeypatch):
    run, rows = harness
    original = rb.load_dev_set

    def with_duplicate(path):
        examples = original(path)
        return examples + [dict(examples[1])]  # question_id 101 twice, like BIRD's 137 and 138

    monkeypatch.setattr(rb, "load_dev_set", with_duplicate)
    monkeypatch.setattr(rb, "build_graph", lambda: FakeGraph())

    rb.run_benchmark(split="calibration")

    assert [r["question_id"] for r in rows()] == ["101", "104", "101"]


# --- results files are never reused silently ------------------------------------


def test_an_existing_results_file_is_refused_without_resume(harness):
    run, rows = harness
    run(FakeGraph())
    before = rows()

    graph = FakeGraph()
    with pytest.raises(SystemExit) as exc:
        run(graph)

    assert "--resume" in str(exc.value)
    assert graph.invoked == [] and rows() == before  # nothing ran, nothing rewritten


def test_a_refused_run_leaves_the_existing_metadata_alone(harness):
    import os

    run, _ = harness
    run(FakeGraph())
    stamp = os.path.getmtime(rb.METADATA_PATH)

    with pytest.raises(SystemExit):
        run(FakeGraph())

    assert os.path.getmtime(rb.METADATA_PATH) == stamp


def test_default_paths_are_timestamped_per_run():
    assert rb.RUN_STAMP in rb.DEFAULT_RESULTS_PATH and rb.DEFAULT_RESULTS_PATH.endswith(".csv")
    assert rb.RUN_STAMP in rb.DEFAULT_METADATA_PATH and rb.DEFAULT_METADATA_PATH.endswith(".json")


def test_resume_matches_by_question_id_not_position(harness):
    run, rows = harness
    with pytest.raises(SystemExit):
        run(FakeGraph(fail_at={3}, fail_with=QuotaExhaustedError("no credit remaining")))
    saved = rows()
    # Reorder the file: positional resume would now skip the wrong questions.
    with open(rb.RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rb.FIELDNAMES, restval="")
        writer.writeheader()
        writer.writerows(reversed(saved))

    resumed = FakeGraph()
    run(resumed, resume=True)

    assert resumed.invoked == [3, 4, 5]
    assert sorted(r["question_id"] for r in rows()) == [str(100 + i) for i in range(N)]


def test_resume_refuses_a_file_from_a_different_set_of_questions(harness, splits, monkeypatch):
    run, rows = harness
    monkeypatch.setattr(rb, "build_graph", lambda: FakeGraph())
    rb.run_benchmark(split="heldout")
    before = rows()

    graph = FakeGraph()
    monkeypatch.setattr(rb, "build_graph", lambda: graph)
    with pytest.raises(SystemExit) as exc:
        rb.run_benchmark(split="calibration", resume=True)  # question_ids 100, 102, 103, 105 are not calibration

    assert "outside split 'calibration'" in str(exc.value)
    assert graph.invoked == [] and rows() == before


def test_resume_with_a_duplicated_question_id_does_each_row_once(harness, monkeypatch):
    run, rows = harness
    original = rb.load_dev_set
    monkeypatch.setattr(rb, "load_dev_set", lambda path: original(path)[:2] + [dict(original(path)[1])])  # 100, 101, 101

    graph = FakeGraph(fail_at={1}, fail_with=QuotaExhaustedError("no credit remaining"))
    with pytest.raises(SystemExit):
        run(graph)
    assert [r["question_id"] for r in rows()] == ["100"]

    run(FakeGraph(), resume=True)

    assert [r["question_id"] for r in rows()] == ["100", "101", "101"]


# --- the split is recorded with every run ---------------------------------------


def test_every_row_records_the_split_it_came_from(harness, splits, monkeypatch):
    monkeypatch.setattr(rb, "build_graph", lambda: FakeGraph())
    run, rows = harness

    rb.run_benchmark(split="heldout")

    assert {r["split"] for r in rows()} == {"heldout"}


def test_split_is_the_last_column_and_the_others_keep_their_order():
    assert rb.FIELDNAMES[-1] == "split"
    assert rb.FIELDNAMES[: len(OLD_COLUMNS)] == OLD_COLUMNS


def test_run_metadata_records_the_split_and_the_hash_of_its_file(harness, splits, monkeypatch):
    import hashlib
    import json

    monkeypatch.setattr(rb, "build_graph", lambda: FakeGraph())
    rb.run_benchmark(split="calibration")

    with open(rb.METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    with open(f"{rb.SPLITS_DIR}/calibration_ids.json", "rb") as f:
        assert metadata["split_file_sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert metadata["split"] == "calibration" and metadata["total_questions"] == 2


def test_an_all_questions_run_records_split_all_with_no_file_hash(harness):
    import json

    run, rows = harness
    run(FakeGraph())

    with open(rb.METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["split"] == "all" and metadata["split_file_sha256"] is None
    assert {r["split"] for r in rows()} == {"all"}
