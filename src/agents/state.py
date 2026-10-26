"""
Shared state passed between every node in the LangGraph state machine.

Design note: `trace` accumulates one entry per node hop, so the full
correction-loop path (which agent fired, why, how many retries) can be
inspected after the fact -- this is what feeds the eval harness's failure
taxonomy and the Streamlit UI's "agent trace" panel.
"""

from typing import TypedDict, List, Optional, Literal, Any

ErrorClass = Literal[
    "SCHEMA_ERROR",
    "SYNTAX_ERROR",
    "LOGIC_ERROR",
    "TIMEOUT_ERROR",
    "UNKNOWN_ERROR",
]


class AgentState(TypedDict):
    db_id: str
    question: str
    evidence: Optional[str]  # BIRD-SQL's external-knowledge hint; see src/config.py's
                              # include_evidence_in_prompts for whether agents actually see it

    schema_context: Optional[str]
    plan: Optional[dict]
    sql_query: Optional[str]

    execution_result: Optional[List[Any]]
    execution_error: Optional[str]
    error_class: Optional[ErrorClass]

    retry_count: int
    max_retries: int

    # Set when schema retrieval itself fails (vector store unreadable, index
    # segment missing). Routes the graph straight to END: without a schema
    # there is nothing for the planner or generator to work from.
    retrieval_failed: bool

    trace: List[dict]
    # One entry per node RUN (the correction loop re-enters nodes), appended by
    # the wrapper in src/graph.py: node, retry_count, seconds, llm_calls, token
    # counts and cost_usd (None when a call's model has no listed price).
    node_runs: List[dict]
    success: bool
    final_answer: Optional[str]


def initial_state(db_id: str, question: str, evidence: Optional[str] = None, max_retries: int = 3) -> AgentState:
    # Strip at the boundary: a trailing space in db_id propagates into the
    # Chroma collection name ("schema_superhero ") and into the SQLite file
    # path, failing far downstream with errors that point at the wrong
    # culprit. Sanitising once here covers every caller -- API, UI, and the
    # benchmark harness -- rather than patching each consumer separately.
    return {
        "db_id": db_id.strip(),
        "question": question.strip(),
        "evidence": evidence.strip() if evidence else None,
        "schema_context": None,
        "plan": None,
        "sql_query": None,
        "execution_result": None,
        "execution_error": None,
        "error_class": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "retrieval_failed": False,
        "trace": [],
        "node_runs": [],
        "success": False,
        "final_answer": None,
    }