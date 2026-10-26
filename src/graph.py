"""
Builds the LangGraph state machine:

    schema_retriever -> query_planner -> sql_generator -> executor
                                                              |
                                          success ------------+------- failure
                                             |                          |
                                            END                 classify_error
                                                                        |
                                        +---------------+---------------+
                                        |               |               |
                                  SCHEMA_ERROR    SYNTAX_ERROR     LOGIC_ERROR
                                        |               |               |
                                schema_retriever  sql_generator   query_planner
                                        |               |               |
                                        +---------------+---------------+
                                                        |
                                                 (loops back to executor,
                                                  bounded by max_retries)

If schema retrieval itself fails (vector store unreadable), route_after_retrieval
ends the question at once, before any LLM call -- see schema_retriever.py.

The routing logic (route_after_classification) is the actual differentiator
of this project: most NL2SQL correction loops just re-prompt with the raw
error message. Routing to the specific agent responsible for that error
class is a more deliberate design and is what should come up in interviews.
"""

import functools
import time

from langgraph.graph import StateGraph, END

from src import tracing, usage
from src.agents.state import AgentState
from src.agents.schema_retriever import schema_retriever_node
from src.agents.query_planner import query_planner_node
from src.agents.sql_generator import sql_generator_node
from src.agents.error_classifier import error_classifier_node
from src.db.executor import executor_node
from src.pricing import calls_cost_usd


def _timed(name: str, node_fn):
    """
    Meter one node: wall time, and the LLM calls, tokens and cost made inside it.

    Applied in build_graph() rather than on the node functions, so the nodes and
    their direct-call tests are untouched, and the node's own update is returned
    as it was with one key added. A node can run several times per question (the
    correction loop re-enters it), so each run appends an entry to
    state["node_runs"]; totals are derived from that list afterwards.

    A node that raises records nothing: the question has no final state to read
    it from, and the exception propagates unchanged.
    """

    @functools.wraps(node_fn)
    def wrapper(state: AgentState) -> dict:
        retry_count = state.get("retry_count", 0)
        span_attributes = {"db_id": state.get("db_id", ""), "retry_count": retry_count}
        # One collector per node run, so nothing carries over between nodes or questions.
        with usage.collect() as collector, tracing.node_span(name, span_attributes) as span:
            started = time.perf_counter()
            update = node_fn(state)
            seconds = time.perf_counter() - started
            entry = {
                "node": name,
                "retry_count": retry_count,
                "seconds": round(seconds, 6),
                **collector.totals(),
                "cost_usd": calls_cost_usd(collector.calls),
            }
            span.set_attributes({k: v for k, v in entry.items() if v is not None})
        return {**update, "node_runs": state.get("node_runs", []) + [entry]}

    return wrapper


def route_after_retrieval(state: AgentState) -> str:
    # .get rather than [] so hand-built states (tests, older callers) without
    # the key behave exactly as before: straight on to the planner.
    if state.get("retrieval_failed"):
        return "end"
    return "query_planner"


def route_after_execution(state: AgentState) -> str:
    if state["success"]:
        return "end"
    if state["retry_count"] >= state["max_retries"]:
        return "end"
    return "classify_error"


def route_after_classification(state: AgentState) -> str:
    error_class = state["error_class"]
    if state["retry_count"] >= state["max_retries"]:
        return "end"
    if error_class == "SCHEMA_ERROR":
        return "schema_retriever"
    if error_class == "SYNTAX_ERROR":
        return "sql_generator"
    if error_class == "LOGIC_ERROR":
        return "query_planner"
    return "end"  # UNKNOWN_ERROR / TIMEOUT_ERROR: not worth retrying blindly


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("schema_retriever", _timed("schema_retriever", schema_retriever_node))
    graph.add_node("query_planner", _timed("query_planner", query_planner_node))
    graph.add_node("sql_generator", _timed("sql_generator", sql_generator_node))
    graph.add_node("executor", _timed("executor", executor_node))
    graph.add_node("classify_error", _timed("classify_error", error_classifier_node))

    graph.set_entry_point("schema_retriever")
    # Conditional rather than a plain edge only so a failed retrieval can end
    # the question. On every successful retrieval it returns "query_planner"
    # -- the same transition the plain edge made.
    graph.add_conditional_edges(
        "schema_retriever",
        route_after_retrieval,
        {"query_planner": "query_planner", "end": END},
    )
    graph.add_edge("query_planner", "sql_generator")
    graph.add_edge("sql_generator", "executor")

    graph.add_conditional_edges(
        "executor",
        route_after_execution,
        {"end": END, "classify_error": "classify_error"},
    )

    graph.add_conditional_edges(
        "classify_error",
        route_after_classification,
        {
            "schema_retriever": "schema_retriever",
            "sql_generator": "sql_generator",
            "query_planner": "query_planner",
            "end": END,
        },
    )

    return graph.compile()
