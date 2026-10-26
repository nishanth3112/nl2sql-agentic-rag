"""
Optional node spans. Free and offline: spans are written to a temp file.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import tracing
from src.config import settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh_tracer_state(monkeypatch):
    monkeypatch.setattr(tracing, "_state", {"tracer": None, "path": None, "unavailable": False})


@pytest.fixture
def span_file(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "spans.jsonl"
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "trace_export_path", str(path))
    return path


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_disabled_spans_are_a_no_op(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tracing_enabled", False)
    monkeypatch.setattr(settings, "trace_export_path", str(tmp_path / "spans.jsonl"))

    with tracing.node_span("query_planner", {"db_id": "x"}) as span:
        span.set_attributes({"llm_calls": 2})

    assert not (tmp_path / "spans.jsonl").exists()


def test_opentelemetry_is_never_imported_while_tracing_is_off():
    # A fresh interpreter: this process has long since imported OpenTelemetry through chromadb.
    code = (
        "import sys\n"
        "from src import tracing\n"
        "with tracing.node_span('query_planner', {'db_id': 'x'}) as span:\n"
        "    span.set_attributes({'llm_calls': 1})\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'opentelemetry')\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "TRACING_ENABLED": "false"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_an_enabled_span_is_written_as_one_json_line(span_file):
    with tracing.node_span("query_planner", {"db_id": "superhero", "retry_count": 1}) as span:
        span.set_attributes({"llm_calls": 2, "est_cost_usd": 0.0123})

    (record,) = _lines(span_file)
    assert record["name"] == "node.query_planner"
    assert record["attributes"] == {
        "node": "query_planner",
        "db_id": "superhero",
        "retry_count": 1,
        "llm_calls": 2,
        "est_cost_usd": 0.0123,
    }
    assert record["status"] == "UNSET"
    assert record["duration_s"] >= 0
    assert len(record["trace_id"]) == 32 and len(record["span_id"]) == 16
    assert record["parent_span_id"] is None
    assert record["start_time"].endswith("+00:00")


def test_each_node_run_is_its_own_line_and_the_directory_is_created(span_file):
    for node in ("schema_retriever", "query_planner", "sql_generator"):
        with tracing.node_span(node):
            pass

    assert [r["attributes"]["node"] for r in _lines(span_file)] == [
        "schema_retriever",
        "query_planner",
        "sql_generator",
    ]


def test_an_exception_is_recorded_on_the_span_and_still_raised(span_file):
    with pytest.raises(RuntimeError, match="executor blew up"):
        with tracing.node_span("executor"):
            raise RuntimeError("executor blew up")

    (record,) = _lines(span_file)
    assert record["status"] == "ERROR"


def test_a_missing_opentelemetry_install_degrades_to_untraced_with_one_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "trace_export_path", str(tmp_path / "spans.jsonl"))

    def not_installed(path):
        raise ImportError("No module named 'opentelemetry'")

    monkeypatch.setattr(tracing, "_build_tracer", not_installed)
    caplog.set_level(logging.WARNING, logger="src.tracing")

    for _ in range(3):
        with tracing.node_span("query_planner") as span:
            span.set_attributes({"llm_calls": 1})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "opentelemetry is not installed" in warnings[0].getMessage()
    assert not (tmp_path / "spans.jsonl").exists()


def test_any_failure_starting_tracing_leaves_the_node_untraced_not_broken(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "trace_export_path", str(tmp_path / "spans.jsonl"))

    def broken(path):
        raise ValueError("provider setup failed")

    monkeypatch.setattr(tracing, "_build_tracer", broken)
    caplog.set_level(logging.WARNING, logger="src.tracing")

    for _ in range(2):
        with tracing.node_span("query_planner") as span:
            span.set_attributes({"llm_calls": 1})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "could not start tracing" in warnings[0].getMessage()


def test_a_write_failure_never_breaks_the_node(span_file, caplog):
    span_file.parent.mkdir(parents=True)
    span_file.mkdir()  # the target path is a directory, so opening it for append fails
    caplog.set_level(logging.WARNING, logger="src.tracing")

    with tracing.node_span("query_planner"):
        result = "node output"

    assert result == "node output"
    assert any("could not write spans" in r.getMessage() for r in caplog.records)


def test_switching_tracing_off_again_stops_writing(span_file, monkeypatch):
    with tracing.node_span("a"):
        pass
    monkeypatch.setattr(settings, "tracing_enabled", False)
    with tracing.node_span("b"):
        pass

    assert [r["attributes"]["node"] for r in _lines(span_file)] == ["a"]
