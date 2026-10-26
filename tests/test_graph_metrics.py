"""
Per-node metering must record what each node run cost without changing a single
answer the graph gives.

Free and offline: the real graph, agents and executor run, but every LLM call is a
script (recording usage the way the provider layer does), the schema store is
fake, and the database is a throwaway SQLite file. Each scenario is also run on a
graph built WITHOUT the wrapper, and the two final states must match apart from
the new `node_runs` key.
"""

import json
import sqlite3

import pytest

from src import graph as graph_module
from src import tracing, usage
from src.agents import error_classifier, query_planner, schema_retriever, sql_generator
from src.agents.state import initial_state
from src.config import settings
from src.graph import build_graph
from src.llm_providers import QuotaExhaustedError

DB_ID = "metrics_db"
GOOD_SQL = "SELECT a FROM t"


@pytest.fixture(autouse=True)
def _environment(monkeypatch, tmp_path):
    db_dir = tmp_path / "dev_databases" / DB_ID
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / f"{DB_ID}.sqlite")
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(settings, "benchmark_data_path", str(tmp_path))

    class Store:
        def retrieve_relevant_tables(self, db_id, question, top_k=6):
            return [{"table_name": "t", "schema_text": "Table: t\nColumns: a"}]

    monkeypatch.setattr(schema_retriever, "store_for", lambda db_id: Store())
    monkeypatch.setattr(schema_retriever, "get_cached_retrieval", lambda *a: None)
    monkeypatch.setattr(schema_retriever, "set_cached_retrieval", lambda *a: None)
    monkeypatch.setattr(tracing, "_state", {"tracer": None, "path": None, "unavailable": False})


def _script(monkeypatch, planner=(), generator=(), classifier=()):
    """
    Fake LLM replies, consumed in order. Each reply is (text, prompt, completion,
    reasoning) or an Exception to raise; a returned reply is recorded as one
    billed call on openai / gpt-5-mini, as the provider layer would.
    """
    queues = {"planner": list(planner), "generator": list(generator), "classifier": list(classifier)}

    def fake(role):
        def generate_text(**kwargs):
            reply = queues[role].pop(0)
            if isinstance(reply, Exception):
                raise reply
            text, prompt, completion, reasoning = reply
            usage.record(
                "openai", "gpt-5-mini",
                prompt_tokens=prompt, completion_tokens=completion, reasoning_tokens=reasoning,
            )
            return text
        return generate_text

    monkeypatch.setattr(query_planner, "generate_text", fake("planner"))
    monkeypatch.setattr(sql_generator, "generate_text", fake("generator"))
    monkeypatch.setattr(error_classifier, "generate_text", fake("classifier"))


PLAN = json.dumps({"tables": ["t"]})


def _happy(monkeypatch):
    _script(
        monkeypatch,
        planner=[(PLAN, 1000, 200, 100)],
        generator=[(GOOD_SQL, 2000, 50, 0)],
    )


def _schema_error_then_recover(monkeypatch):
    # The first query names a column that doesn't exist: the rule-based classifier
    # (no LLM) sends it back to the retriever, then planner and generator run again.
    _script(
        monkeypatch,
        planner=[(PLAN, 1000, 200, 100), (PLAN, 1100, 210, 100)],
        generator=[("SELECT nope FROM t", 2000, 50, 0), (GOOD_SQL, 2100, 40, 0)],
    )


def _llm_classified_then_recover(monkeypatch):
    # "no such function" matches no rule, so the classifier asks the model.
    _script(
        monkeypatch,
        planner=[(PLAN, 1000, 200, 100)],
        generator=[("SELECT nofn(a) FROM t", 2000, 50, 0), (GOOD_SQL, 2100, 40, 0)],
        classifier=[("SYNTAX_ERROR", 300, 20, 18)],
    )


SCENARIOS = {
    "happy": _happy,
    "schema_error_then_recover": _schema_error_then_recover,
    "llm_classified_then_recover": _llm_classified_then_recover,
}


def _run(monkeypatch, scenario, instrumented=True):
    SCENARIOS[scenario](monkeypatch)
    if not instrumented:
        monkeypatch.setattr(graph_module, "_timed", lambda name, fn: fn)
    return build_graph().invoke(initial_state(DB_ID, "How many rows?"))


def _without_node_runs(state):
    return {k: v for k, v in state.items() if k != "node_runs"}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_wrapper_changes_no_answer(monkeypatch, scenario):
    metered = _run(monkeypatch, scenario)
    reference = _run(monkeypatch, scenario, instrumented=False)

    assert "node_runs" not in reference or reference["node_runs"] == []
    assert _without_node_runs(metered) == _without_node_runs(reference)
    assert metered["success"] is True
    assert metered["sql_query"] == GOOD_SQL
    assert metered["execution_result"] == [(1,), (2,)]


def test_a_happy_path_records_each_node_once_with_its_tokens_and_cost(monkeypatch):
    final = _run(monkeypatch, "happy")

    runs = final["node_runs"]
    assert [r["node"] for r in runs] == ["schema_retriever", "query_planner", "sql_generator", "executor"]
    by_node = {r["node"]: r for r in runs}

    planner = by_node["query_planner"]
    assert (planner["llm_calls"], planner["prompt_tokens"], planner["completion_tokens"]) == (1, 1000, 200)
    assert planner["reasoning_tokens"] == 100
    assert planner["cost_usd"] == pytest.approx(1000 * 0.25 / 1e6 + 200 * 2.00 / 1e6)

    generator = by_node["sql_generator"]
    assert (generator["llm_calls"], generator["prompt_tokens"]) == (1, 2000)
    assert generator["cost_usd"] == pytest.approx(2000 * 0.25 / 1e6 + 50 * 2.00 / 1e6)

    # Nodes that make no LLM call cost exactly zero, not unknown.
    for name in ("schema_retriever", "executor"):
        assert by_node[name]["llm_calls"] == 0
        assert by_node[name]["cost_usd"] == 0.0

    for run in runs:
        assert run["seconds"] >= 0
        assert run["retry_count"] == 0


def test_a_node_the_correction_loop_re_enters_gets_one_entry_per_run(monkeypatch):
    final = _run(monkeypatch, "schema_error_then_recover")

    runs = final["node_runs"]
    assert [r["node"] for r in runs] == [
        "schema_retriever", "query_planner", "sql_generator", "executor",  # first pass fails
        "classify_error",                                                   # rule-based, no LLM
        "schema_retriever", "query_planner", "sql_generator", "executor",  # second pass
    ]
    assert [r["retry_count"] for r in runs] == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    # The visit order metering saw is the one the trace recorded (the trace calls
    # the classifier node "error_classifier", the graph calls it "classify_error").
    assert [r["node"].replace("classify_error", "error_classifier") for r in runs] == [
        t["node"] for t in final["trace"]
    ]
    classifier = runs[4]
    assert (classifier["llm_calls"], classifier["cost_usd"]) == (0, 0.0)


def test_totals_over_the_node_runs_equal_what_the_script_spent(monkeypatch):
    final = _run(monkeypatch, "schema_error_then_recover")

    runs = final["node_runs"]
    assert sum(r["llm_calls"] for r in runs) == 4
    assert sum(r["prompt_tokens"] for r in runs) == 1000 + 1100 + 2000 + 2100
    assert sum(r["completion_tokens"] for r in runs) == 200 + 210 + 50 + 40
    assert sum(r["reasoning_tokens"] for r in runs) == 100 + 100
    expected = (1000 + 1100 + 2000 + 2100) * 0.25 / 1e6 + (200 + 210 + 50 + 40) * 2.00 / 1e6
    assert sum(r["cost_usd"] for r in runs) == pytest.approx(expected)


def test_an_llm_classifier_call_is_metered_on_its_own_node(monkeypatch):
    final = _run(monkeypatch, "llm_classified_then_recover")

    classifier = next(r for r in final["node_runs"] if r["node"] == "classify_error")
    assert (classifier["llm_calls"], classifier["prompt_tokens"], classifier["completion_tokens"]) == (1, 300, 20)
    assert classifier["cost_usd"] == pytest.approx(300 * 0.25 / 1e6 + 20 * 2.00 / 1e6)


def test_a_call_to_an_unpriced_model_leaves_that_nodes_cost_unknown(monkeypatch):
    _happy(monkeypatch)

    def unpriced(**kwargs):
        usage.record("groq", "llama-3.3-70b-versatile", prompt_tokens=500, completion_tokens=50)
        return PLAN

    monkeypatch.setattr(query_planner, "generate_text", unpriced)

    final = build_graph().invoke(initial_state(DB_ID, "How many rows?"))

    by_node = {r["node"]: r for r in final["node_runs"]}
    assert by_node["query_planner"]["cost_usd"] is None
    assert by_node["query_planner"]["prompt_tokens"] == 500  # tokens are still known
    assert by_node["sql_generator"]["cost_usd"] is not None  # one node's gap doesn't hide the others


def test_a_failed_retrieval_ends_the_question_with_one_costless_entry(monkeypatch):
    class Failing:
        def retrieve_relevant_tables(self, *a, **kw):
            raise RuntimeError("hnsw segment missing")

    monkeypatch.setattr(schema_retriever, "store_for", lambda db_id: Failing())
    _script(monkeypatch)  # any LLM call would pop from an empty script and fail the test

    final = build_graph().invoke(initial_state(DB_ID, "How many rows?"))

    assert final["retrieval_failed"] is True
    assert [r["node"] for r in final["node_runs"]] == ["schema_retriever"]
    assert (final["node_runs"][0]["llm_calls"], final["node_runs"][0]["cost_usd"]) == (0, 0.0)


def test_quota_exhaustion_still_propagates_out_of_the_graph(monkeypatch):
    _script(monkeypatch, planner=[QuotaExhaustedError("no credit")])

    with pytest.raises(QuotaExhaustedError):
        build_graph().invoke(initial_state(DB_ID, "How many rows?"))


def test_any_other_node_exception_also_propagates_unchanged(monkeypatch):
    _script(monkeypatch, planner=[ValueError("planner exploded")])

    with pytest.raises(ValueError, match="planner exploded"):
        build_graph().invoke(initial_state(DB_ID, "How many rows?"))


def test_metering_does_not_leak_between_questions(monkeypatch):
    first = _run(monkeypatch, "happy")
    second = _run(monkeypatch, "happy")

    assert [r["llm_calls"] for r in first["node_runs"]] == [r["llm_calls"] for r in second["node_runs"]]
    assert sum(r["prompt_tokens"] for r in second["node_runs"]) == 3000


def test_hand_built_states_without_the_new_keys_still_work():
    # Existing tests and older callers build states without node_runs or retry_count.
    wrapped = graph_module._timed("probe", lambda state: {"sql_query": "SELECT 1"})

    update = wrapped({"db_id": "x"})

    assert update["sql_query"] == "SELECT 1"
    assert [r["node"] for r in update["node_runs"]] == ["probe"]
    assert update["node_runs"][0]["retry_count"] == 0


def test_the_wrapper_keeps_the_node_signature_langgraph_reads():
    from src.db.executor import executor_node

    wrapped = graph_module._timed("executor", executor_node)

    assert wrapped.__name__ == executor_node.__name__
    assert wrapped.__wrapped__ is executor_node


def test_enabled_tracing_writes_one_span_per_node_run(monkeypatch, tmp_path):
    span_file = tmp_path / "spans.jsonl"
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "trace_export_path", str(span_file))

    final = _run(monkeypatch, "schema_error_then_recover")

    spans = [json.loads(line) for line in span_file.read_text(encoding="utf-8").splitlines()]
    assert [s["attributes"]["node"] for s in spans] == [r["node"] for r in final["node_runs"]]
    planner_span = next(s for s in spans if s["attributes"]["node"] == "query_planner")
    assert planner_span["attributes"]["llm_calls"] == 1
    assert planner_span["attributes"]["prompt_tokens"] == 1000
    assert planner_span["attributes"]["db_id"] == DB_ID
    # Tracing changes no answer either.
    assert final["success"] is True and final["sql_query"] == GOOD_SQL


def test_tracing_off_writes_nothing(monkeypatch, tmp_path):
    span_file = tmp_path / "spans.jsonl"
    monkeypatch.setattr(settings, "tracing_enabled", False)
    monkeypatch.setattr(settings, "trace_export_path", str(span_file))

    _run(monkeypatch, "happy")

    assert not span_file.exists()
