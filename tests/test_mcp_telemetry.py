"""
QUERY_HISTORY failures must stay out of the caller's way and still be visible.

Free and offline: the Snowflake connection is replaced, and the manifest's call
to the API is stubbed.
"""

import asyncio
import logging

import pytest

from mcp_server import server

SECRET_DETAIL = "key file /home/someone/.snowflake/rsa_key.p8 not found"


@pytest.fixture(autouse=True)
def fresh_counters(monkeypatch):
    monkeypatch.setattr(
        server, "_telemetry", {"writes_ok": 0, "writes_failed": 0, "last_failure": None}
    )


class FakeConnection:
    def __init__(self):
        self.executed = []
        self.commits = 0

    def cursor(self):
        conn = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params):
                conn.executed.append(params)

        return Cursor()

    def commit(self):
        self.commits += 1


def _failing_connection():
    raise FileNotFoundError(SECRET_DETAIL)


def test_failed_insert_is_swallowed_counted_and_logged(monkeypatch, caplog, capsys):
    monkeypatch.setattr(server, "_get_logging_connection", _failing_connection)
    caplog.set_level(logging.WARNING, logger="nl2sql_mcp")

    result = server._log_tool_call(None, "nl2sql_list_databases", {}, success=True)

    assert result is None
    snapshot = server._telemetry_snapshot()
    assert snapshot["writes_failed"] == 1
    assert snapshot["writes_ok"] == 0
    assert snapshot["last_failure"]["tool"] == "nl2sql_list_databases"
    assert snapshot["last_failure"]["error_type"] == "FileNotFoundError"
    assert snapshot["last_failure"]["at"].endswith("+00:00")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert "nl2sql_list_databases" in text
    assert "FileNotFoundError" in text
    # The message can carry paths and connection details, and the manifest is read by a model.
    assert SECRET_DETAIL not in text
    assert SECRET_DETAIL not in str(snapshot)
    # Under stdio, stdout is the JSON-RPC stream.
    assert capsys.readouterr().out == ""


def test_successful_insert_is_counted(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(server, "_get_logging_connection", lambda: conn)

    server._log_tool_call(None, "nl2sql_list_databases", {}, success=True)

    assert conn.commits == 1
    assert server._telemetry_snapshot() == {"writes_ok": 1, "writes_failed": 0, "last_failure": None}


def test_a_broken_reporter_still_does_not_raise(monkeypatch, capsys):
    monkeypatch.setattr(server, "_get_logging_connection", _failing_connection)

    def broken_reporter(tool, exc):
        raise RuntimeError("reporter down")

    monkeypatch.setattr(server, "_note_telemetry_failure", broken_reporter)

    server._log_tool_call(None, "nl2sql_list_databases", {}, success=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "QUERY_HISTORY insert failed for nl2sql_list_databases" in captured.err
    assert "reporting it also failed" in captured.err


def test_manifest_exposes_the_counters(monkeypatch):
    monkeypatch.setattr(server, "_call_api", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(server, "_get_logging_connection", _failing_connection)
    server._log_tool_call(None, "nl2sql_execute_query", {}, success=False)

    manifest = asyncio.run(server._build_manifest())

    assert manifest["telemetry"]["writes_failed"] == 1
    assert manifest["telemetry"]["last_failure"]["tool"] == "nl2sql_execute_query"
    assert any("telemetry" in g for g in manifest["guarantees"])
    assert not any("reported on stderr instead" in g for g in manifest["guarantees"])


def test_snapshot_is_a_copy(monkeypatch):
    monkeypatch.setattr(server, "_get_logging_connection", _failing_connection)
    server._log_tool_call(None, "nl2sql_list_databases", {}, success=True)

    snapshot = server._telemetry_snapshot()
    snapshot["last_failure"]["tool"] = "tampered"
    snapshot["writes_failed"] = 99

    assert server._telemetry_snapshot()["last_failure"]["tool"] == "nl2sql_list_databases"
    assert server._telemetry_snapshot()["writes_failed"] == 1
