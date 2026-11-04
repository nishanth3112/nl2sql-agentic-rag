"""
Query Planner Agent.

Deliberately outputs a structured intermediate plan (tables, joins, filters,
aggregation) rather than jumping straight to SQL. This intermediate step is
what makes the correction loop useful: when the error classifier flags a
LOGIC_ERROR, the router sends the error back HERE (not to the SQL generator),
because the mistake is in the query's intent/structure, not its syntax.
"""

import json

from src.config import settings
from src.llm_providers import generate_text
from src.agents.state import AgentState

PLANNER_SYSTEM_PROMPT = """You are a SQL query planning agent. Given a natural \
language question and relevant database schema, produce a structured plan as \
JSON with these fields:
- tables: list of table names needed
- joins: list of {"left": str, "right": str, "on": str}
- filters: list of plain-language filter conditions
- aggregation: string (e.g. "COUNT", "AVG") or null
- group_by: list of column names
- order_by: list of {"column": str, "direction": "ASC"|"DESC"}
- notes: brief string flagging any ambiguity in the question

Respond with ONLY valid JSON. No prose, no markdown fences."""


def query_planner_node(state: AgentState) -> dict:
    error_context = ""
    if state.get("error_class") == "LOGIC_ERROR":
        error_context = (
            f"\n\nYour previous plan led to a logic error: {state.get('execution_error')}\n"
            f"Previous plan: {json.dumps(state.get('plan'))}\n"
            f"Revise the plan to fix the underlying logic (joins/aggregation/filters), not just the wording."
        )

    # Off by default -- see src/config.py's include_evidence_in_prompts. The
    # schema retriever agent's whole job is finding relevant context itself;
    # feeding it BIRD's answer-adjacent hints would make that job trivial
    # and understate what the retrieval step actually does.
    evidence_context = ""
    if settings.include_evidence_in_prompts and state.get("evidence"):
        evidence_context = f"\n\nHint: {state['evidence']}"

    notes = state.get("semantic_context")
    notes_context = f"\n\nDomain notes:\n{notes}" if notes else ""

    user_prompt = (
        f"Question: {state['question']}\n\n"
        f"Schema:\n{state['schema_context']}{notes_context}{evidence_context}{error_context}"
    )

    raw_text = generate_text(
        provider=settings.planner_provider,
        model=settings.planner_model,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=1000,
        json_mode=True,
    )

    try:
        plan = json.loads(raw_text)
    except json.JSONDecodeError:
        plan = {"raw": raw_text, "parse_error": True}

    trace_entry = {"node": "query_planner", "retry_count": state["retry_count"], "plan": plan}

    return {
        "plan": plan,
        "trace": state["trace"] + [trace_entry],
    }
