"""
A failed schema retrieval must end the question, not crash the process.

Free and offline: the vector store and every LLM call are replaced, so no
index, Redis, API key or network is needed. The failure case is the one that
actually happened -- all 6 crashed questions in the 500-question baseline were
formula_1 questions that died on their first retrieval with Chroma's
"Error creating hnsw segment reader: Nothing found on disk".
"""

import pytest

from eval.metrics import execution_match
from src.agents import error_classifier, query_planner, schema_retriever, sql_generator
from src.agents.schema_retriever import schema_retriever_node
from src.agents.state import initial_state
from src.graph import build_graph, route_after_retrieval

HNSW = RuntimeError(
    "Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk"
)


class _FailingStore:
    def __init__(self, exc):
        self.exc = exc

    def retrieve_relevant_tables(self, *a, **kw):
        raise self.exc


class _WorkingStore:
    def retrieve_relevant_tables(self, db_id, question, top_k=6):
        return [{"table_name": "results", "schema_text": "Table: results\nColumns: constructorId"}]


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    # The Redis cache already swallows its own errors; bypass it so these
    # tests exercise the store call and nothing else.
    monkeypatch.setattr(schema_retriever, "get_cached_retrieval", lambda *a: None)
    monkeypatch.setattr(schema_retriever, "set_cached_retrieval", lambda *a: None)


def _store(monkeypatch, store):
    monkeypatch.setattr(schema_retriever, "store_for", lambda db_id: store)


@pytest.mark.parametrize(
    "exc",
    [HNSW, ValueError("Collection schema_formula_1 does not exist."), OSError("disk I/O error"), MemoryError()],
)
def test_retrieval_failure_is_contained(monkeypatch, exc):
    _store(monkeypatch, _FailingStore(exc))

    out = schema_retriever_node(initial_state("formula_1", "Which constructor won the most races?"))  # must not raise

    assert out["retrieval_failed"] is True
    assert out["success"] is False
    assert out["execution_result"] is None
    assert type(exc).__name__ in out["execution_error"]
    entry = out["trace"][-1]
    assert entry["node"] == "schema_retriever"
    assert entry["status"] == "failed"
    assert type(exc).__name__ in entry["retrieval_failure"]


def test_store_construction_failure_is_contained(monkeypatch):
    # store_for() can itself raise -- e.g. opening the upload store.
    def boom(db_id):
        raise HNSW
    monkeypatch.setattr(schema_retriever, "store_for", boom)
    out = schema_retriever_node(initial_state("formula_1", "q"))
    assert out["retrieval_failed"] is True


def test_success_path_returns_exactly_the_old_keys(monkeypatch):
    # A completed retrieval must look exactly as it did before this change:
    # same keys, no retrieval_failed, no failure fields in the trace.
    _store(monkeypatch, _WorkingStore())
    out = schema_retriever_node(initial_state("formula_1", "q"))
    assert set(out) == {"schema_context", "semantic_context", "trace"}
    assert out["schema_context"] == "Table: results\nColumns: constructorId"
    assert out["trace"][-1] == {
        "node": "schema_retriever",
        "retry_count": 0,
        "retrieved_tables": ["results"],
        "cache_hit": False,
    }


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"retrieval_failed": True}, "end"),
        ({"retrieval_failed": False}, "query_planner"),
        ({}, "query_planner"),  # hand-built state without the key: as before
    ],
)
def test_route_after_retrieval(state, expected):
    assert route_after_retrieval(state) == expected


def _forbid_llm(monkeypatch):
    def forbidden(**kw):
        raise AssertionError("no LLM call may happen after a failed retrieval")
    for module in (query_planner, sql_generator, error_classifier):
        monkeypatch.setattr(module, "generate_text", forbidden)


def test_graph_ends_before_any_llm_call(monkeypatch):
    _store(monkeypatch, _FailingStore(HNSW))
    _forbid_llm(monkeypatch)

    final = build_graph().invoke(initial_state("formula_1", "Which constructor won the most races?"))

    assert final["retrieval_failed"] is True
    assert final["success"] is False
    assert final["execution_result"] is None
    assert final["sql_query"] is None
    assert [t["node"] for t in final["trace"]] == ["schema_retriever"]


@pytest.mark.parametrize("gold", [[(6,)], [], [("Ferrari",), ("McLaren",)]])
def test_contained_failure_scores_incorrect_like_a_crash(monkeypatch, gold):
    # The benchmark scores a question with execution_match(execution_result,
    # gold). A crash was scored False; the contained failure must be too --
    # including against an EMPTY gold result, the one case where a sloppy
    # fallback (an empty list instead of None) could have scored correct.
    _store(monkeypatch, _FailingStore(HNSW))
    _forbid_llm(monkeypatch)
    final = build_graph().invoke(initial_state("formula_1", "q"))
    assert execution_match(final.get("execution_result"), gold) is False
    assert execution_match([], []) is True  # why None, not [], is the right sentinel


def test_failure_on_a_schema_error_retry_also_ends(monkeypatch):
    # The retriever is also the re-entry point after a SCHEMA_ERROR. A
    # failure there must end the question too, not loop.
    _store(monkeypatch, _FailingStore(HNSW))
    state = initial_state("formula_1", "q")
    state.update(retry_count=1, sql_query="SELECT wins FROM results", execution_error="no such column: wins")
    out = schema_retriever_node(state)
    assert route_after_retrieval({**state, **out}) == "end"


# --- unknown databases, against a REAL Chroma store ---------------------------
#
# Not stubbed: these build an actual PersistentClient in a temp directory,
# because the claim under test is about what lands on disk. Needs the ONNX
# embedding model in the local cache, as tests/test_uploads.py already does.

from src.retrieval.vector_store import SchemaVectorStore

_TABLES = {
    "results": {"columns": [
        {"name": "constructorId", "type": "INTEGER", "samples": ["6", "1"], "description": ""},
        {"name": "positionOrder", "type": "INTEGER", "samples": ["1", "2"], "description": ""},
    ]},
}


@pytest.fixture
def real_store(tmp_path):
    store = SchemaVectorStore(persist_dir=str(tmp_path / "chroma"))
    store.index_schema("known_db", _TABLES)
    return store


def _collections(store):
    return sorted(c.name for c in store.client.list_collections())


def test_known_database_still_retrieves(real_store):
    tables = real_store.retrieve_relevant_tables("known_db", "who won the most races?", top_k=3)
    assert [t["table_name"] for t in tables] == ["results"]


def test_unknown_database_raises_and_creates_nothing(real_store):
    before = _collections(real_store)
    with pytest.raises(Exception):
        real_store.retrieve_relevant_tables("no_such_db", "anything", top_k=3)
    assert _collections(real_store) == before == ["schema_known_db"]


def test_unknown_database_ends_question_without_writing_to_index(monkeypatch, real_store):
    # End to end: the real store, the real graph. The question must end as a
    # retrieval failure -- not continue to the planner with an empty schema --
    # and the index must hold exactly the collections it held before.
    monkeypatch.setattr(schema_retriever, "store_for", lambda db_id: real_store)
    _forbid_llm(monkeypatch)
    before = _collections(real_store)

    final = build_graph().invoke(initial_state("no_such_db", "Which constructor won the most races?"))

    assert final["retrieval_failed"] is True
    assert final["success"] is False
    assert final["execution_result"] is None
    assert "Schema retrieval failed" in final["execution_error"]
    assert [t["node"] for t in final["trace"]] == ["schema_retriever"]
    assert execution_match(final.get("execution_result"), []) is False
    assert _collections(real_store) == before


def test_repeated_unknown_names_leave_no_junk(real_store):
    # The accumulation the old get_or_create behaviour caused: every unknown
    # name used to leave one more empty collection behind.
    for name in ("typo_db", "formula1", "Formula_1 ", "nope"):
        with pytest.raises(Exception):
            real_store.retrieve_relevant_tables(name, "q", top_k=3)
    assert _collections(real_store) == ["schema_known_db"]
