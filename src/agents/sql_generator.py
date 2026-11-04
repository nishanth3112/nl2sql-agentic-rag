"""
SQL Generator Agent.

Takes the planner's structured intent + schema context and writes SQLite.
This is also the re-entry point when the error classifier flags a
SYNTAX_ERROR -- the router sends it back HERE (not to the planner), since
the intent was fine and only the SQL surface syntax needs fixing.
"""

from src.config import settings
from src.llm_providers import generate_text
from src.agents.state import AgentState

GENERATOR_SYSTEM_PROMPT = """You are a SQLite query generation agent. Given a \
structured query plan and database schema, write a single valid SQLite query \
that implements the plan and answers the question.

Critical rules:
- SELECT ONLY the column(s) the question actually asks for. Do not add extra \
descriptive columns for context -- results are compared exactly against a \
reference answer, so an extra column makes a correct query score as wrong.
- Match the granularity the question asks for. If it asks for a month, return \
the month, not the full date.
- Study the sample values shown for each column and match their exact format. \
Do NOT assume a column named "Date" holds ISO dates -- if the samples show \
values like '201308', use string operations such as SUBSTR or LIKE, not \
strftime() or ISO-formatted BETWEEN ranges.

Respond with ONLY the SQL query -- no explanation, no markdown fences."""

def sql_generator_node(state: AgentState) -> dict:
    error_context = ""
    if state.get("error_class") == "SYNTAX_ERROR":
        error_context = (
            f"\n\nThe previous query failed with this SQLite error:\n{state.get('execution_error')}\n"
            f"Previous query: {state.get('sql_query')}\n"
            f"Fix the syntax while keeping the same intent."
        )

    notes = state.get("semantic_context")
    notes_context = f"Domain notes:\n{notes}\n\n" if notes else ""

    user_prompt = (
        f"Question: {state['question']}\n\n"
        f"Schema:\n{state['schema_context']}\n\n"
        f"{notes_context}"
        f"Plan:\n{state['plan']}{error_context}"
    )

    sql = generate_text(
        provider=settings.generator_provider,
        model=settings.generator_model,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=500,
    )
    sql = sql.strip()
    # Defensive cleanup in case the model wraps the query in fences anyway.
    sql = sql.replace("```sql", "").replace("```", "").strip()

    trace_entry = {"node": "sql_generator", "retry_count": state["retry_count"], "sql": sql}

    return {
        "sql_query": sql,
        "trace": state["trace"] + [trace_entry],
    }
