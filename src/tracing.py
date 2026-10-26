"""
Optional OpenTelemetry spans for graph nodes, written to a local JSONL file.

Off unless TRACING_ENABLED=true, and then nothing leaves the machine: each
finished span becomes one JSON line in TRACE_EXPORT_PATH. No exporter service,
no credentials. Pointing the same spans at Langfuse or another OTLP backend
later is a different exporter, not different instrumentation.

OpenTelemetry is imported only when tracing is switched on, so the default path
costs nothing and a deployment without the package (the Lambda image installs a
slim dependency group) just logs one warning and carries on untraced.

Spans use a private TracerProvider rather than the global one: chromadb also
uses OpenTelemetry, and replacing the global provider would change what it does.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

SERVICE_NAME = "nl2sql-agentic-rag"
TRACER_NAME = "nl2sql.graph"


class _NoSpan:
    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass


_NO_SPAN = _NoSpan()

_state: dict[str, Any] = {"tracer": None, "path": None, "unavailable": False}
_lock = threading.Lock()


def _span_record(span) -> dict[str, Any]:
    started = datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc)
    return {
        "name": span.name,
        "trace_id": format(span.context.trace_id, "032x"),
        "span_id": format(span.context.span_id, "016x"),
        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
        "start_time": started.isoformat(),
        "duration_s": round((span.end_time - span.start_time) / 1e9, 6),
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes or {}),
    }


def _build_tracer(path: str):
    # The only place OpenTelemetry is imported.
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class JsonlSpanExporter(SpanExporter):
        def __init__(self, file_path: str):
            self._path = file_path
            self._write_lock = threading.Lock()

        def export(self, spans):
            try:
                text = "".join(json.dumps(_span_record(s)) + "\n" for s in spans)
                os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
                with self._write_lock, open(self._path, "a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:  # noqa: BLE001 -- a trace write must never break the node it observes
                logger.warning("could not write spans to %s", self._path, exc_info=True)
                return SpanExportResult.FAILURE
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    # Synchronous: a benchmark run emits a handful of spans per question, so
    # batching buys nothing, and a killed run loses nothing that was finished.
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))
    return provider.get_tracer(TRACER_NAME)


def _get_tracer():
    if not settings.tracing_enabled:
        return None
    path = settings.trace_export_path
    with _lock:
        if _state["unavailable"]:
            return None
        if _state["tracer"] is None or _state["path"] != path:
            try:
                _state["tracer"] = _build_tracer(path)
            except ImportError:
                _state["unavailable"] = True
                logger.warning("TRACING_ENABLED is set but opentelemetry is not installed; tracing stays off")
                return None
            except Exception:  # noqa: BLE001 -- tracing must never be what fails a node
                _state["unavailable"] = True
                logger.warning("could not start tracing; it stays off", exc_info=True)
                return None
            _state["path"] = path
        return _state["tracer"]


@contextmanager
def node_span(node: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """
    One span for one run of a graph node. Yields an object with set_attributes()
    so figures only known afterwards (tokens, cost) can be added. Attribute
    values must be str, bool, int or float -- leave a field out rather than pass
    None. An exception inside the block is recorded on the span and re-raised.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield _NO_SPAN
        return
    with tracer.start_as_current_span(f"node.{node}", attributes={"node": node, **(attributes or {})}) as span:
        yield span
