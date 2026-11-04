"""
Schema Retriever Agent.

Entry point of the graph, and the re-entry point when the error classifier
flags a SCHEMA_ERROR -- in that case it retrieves a wider slice of the
schema on retry.
"""

from src import semantic_context, uploads
from src.config import settings
from src.retrieval.vector_store import SchemaVectorStore
from src.retrieval.cache import get_cached_retrieval, set_cached_retrieval
from src.agents.state import AgentState

_shipped_store = SchemaVectorStore()
_upload_store = None


def store_for(db_id: str) -> SchemaVectorStore:
    """
    Uploads live in their own Chroma directory (see src/uploads.py): the shipped
    index is a committed artifact the deployed app cannot rebuild, while upload
    segments are per-session garbage. Same reason the executor resolves upload
    paths separately -- one db_id namespace, two stores with different lifecycles.
    """
    global _upload_store
    if not uploads.is_upload(db_id):
        return _shipped_store
    if _upload_store is None:
        _upload_store = uploads.store()
    return _upload_store


def schema_retriever_node(state: AgentState) -> dict:
    # On a schema-error retry, widen the search since the first pass likely
    # missed the table/column the question actually needed.
    top_k = 6 if state["retry_count"] == 0 else 10

    relevant_tables = get_cached_retrieval(state["db_id"], state["question"], top_k)
    cache_hit = relevant_tables is not None

    if not cache_hit:
        # A failed retrieval must never fail the PROCESS. Unguarded, a Chroma
        # error -- the 6 historical crashes were all "Error creating hnsw
        # segment reader: Nothing found on disk" -- propagated out of
        # graph.invoke: a fatal_error row in the benchmark, a 500 in the API.
        #
        # The fallback ENDS the question rather than continuing with an empty
        # schema. Continuing would still spend planner and generator calls,
        # and could occasionally guess correct SQL with no schema at all --
        # turning a question that crashed into one scored correct, which
        # would move results rather than just stop the crash. Ending
        # guarantees the question scores exactly as a crash did: incorrect.
        try:
            relevant_tables = store_for(state["db_id"]).retrieve_relevant_tables(
                state["db_id"], state["question"], top_k=top_k
            )
        except Exception as e:  # noqa: BLE001 -- deliberately total; see above
            failure = f"{type(e).__name__}: {e}"[:300]
            return {
                "retrieval_failed": True,
                "success": False,
                "execution_result": None,
                "execution_error": f"Schema retrieval failed: {failure}",
                "trace": state["trace"] + [
                    {
                        "node": "schema_retriever",
                        "retry_count": state["retry_count"],
                        "status": "failed",
                        "retrieval_failure": failure,
                    }
                ],
            }
        set_cached_retrieval(state["db_id"], state["question"], top_k, relevant_tables)

    schema_text = "\n\n".join(t["schema_text"] for t in relevant_tables if t.get("schema_text"))
    table_names = [t["table_name"] for t in relevant_tables]
    notes = semantic_context.for_tables(state["db_id"], table_names) if settings.semantic_context_enabled else ""

    trace_entry = {
        "node": "schema_retriever",
        "retry_count": state["retry_count"],
        "retrieved_tables": table_names,
        "cache_hit": cache_hit,
    }

    return {
        "schema_context": schema_text,
        "semantic_context": notes,
        "trace": state["trace"] + [trace_entry],
    }