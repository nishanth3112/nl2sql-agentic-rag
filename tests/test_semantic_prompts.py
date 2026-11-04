"""
Semantic definitions reach the planner and generator only when switched on.

Free and offline: the store, cache and LLM calls are replaced. The flag-off
prompts are pinned byte for byte, since the Phase 2 baselines were measured
with exactly those prompts.
"""

import pytest

from src import semantic_context
from src.agents import query_planner, schema_retriever, sql_generator
from src.agents.state import initial_state
from src.config import settings


class _Store:
    def retrieve_relevant_tables(self, db_id, question, top_k=6):
        return [{"table_name": "results", "schema_text": "Table: results"}]


@pytest.fixture
def retrieve(monkeypatch):
    monkeypatch.setattr(schema_retriever, "get_cached_retrieval", lambda *a: None)
    monkeypatch.setattr(schema_retriever, "set_cached_retrieval", lambda *a: None)
    monkeypatch.setattr(schema_retriever, "store_for", lambda db_id: _Store())
    seen = []

    def fake_for_tables(db_id, tables):
        seen.append((db_id, list(tables)))
        return "- a note"

    monkeypatch.setattr(semantic_context, "for_tables", fake_for_tables)

    def run(enabled):
        monkeypatch.setattr(settings, "semantic_context_enabled", enabled)
        return schema_retriever.schema_retriever_node(initial_state("formula_1", "q")), seen

    return run


def test_flag_off_selects_nothing(retrieve):
    out, seen = retrieve(False)
    assert out["semantic_context"] == "" and seen == []


def test_flag_on_selects_for_the_retrieved_tables(retrieve):
    out, seen = retrieve(True)
    assert out["semantic_context"] == "- a note"
    assert seen == [("formula_1", ["results"])]


def prompts(monkeypatch, notes):
    captured = {}

    def fake(name, reply):
        def generate_text(**kw):
            captured[name] = kw["user_prompt"]
            return reply
        return generate_text

    monkeypatch.setattr(query_planner, "generate_text", fake("planner", "{}"))
    monkeypatch.setattr(sql_generator, "generate_text", fake("generator", "SELECT 1"))
    state = {**initial_state("db", "How many?"), "schema_context": "Table: t", "semantic_context": notes}
    query_planner.query_planner_node(state)
    sql_generator.sql_generator_node({**state, "plan": {"tables": ["t"]}})
    return captured


@pytest.mark.parametrize("notes", [None, ""])
def test_without_notes_the_prompts_are_unchanged(monkeypatch, notes):
    captured = prompts(monkeypatch, notes)
    assert captured["planner"] == "Question: How many?\n\nSchema:\nTable: t"
    assert captured["generator"] == "Question: How many?\n\nSchema:\nTable: t\n\nPlan:\n{'tables': ['t']}"


def test_notes_reach_both_prompts(monkeypatch):
    captured = prompts(monkeypatch, "- x means y")
    assert captured["planner"] == "Question: How many?\n\nSchema:\nTable: t\n\nDomain notes:\n- x means y"
    assert captured["generator"] == (
        "Question: How many?\n\nSchema:\nTable: t\n\nDomain notes:\n- x means y\n\nPlan:\n{'tables': ['t']}"
    )
