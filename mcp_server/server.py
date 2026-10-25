"""Multi-tool MCP server exposing the agentic NL2SQL pipeline to MCP clients.

WHAT MCP IS, IN ONE PARAGRAPH
-----------------------------
The Model Context Protocol is a wire protocol that lets an LLM host (Claude
Desktop, Claude Code, an IDE, a custom agent) discover and call tools that live
in a separate process. The host speaks JSON-RPC to this process over stdio (or
HTTP/SSE). Every function decorated with `@mcp.tool()` below is advertised to
the host during the `tools/list` handshake -- name, description, and a JSON
Schema for its arguments -- and the host's model decides when to call it. The
docstrings in this file are not documentation for humans alone: FastMCP lifts
them verbatim into the tool description the model reads when deciding whether a
tool is relevant. Vague docstrings produce a model that calls the wrong tool.

THE TOOL SURFACE
----------------
    nl2sql_list_databases     which databases exist           (discovery)
    nl2sql_get_capabilities   what every tool enforces/refuses (discovery)
    nl2sql_generate_sql       question  -> corrected SQL       (LLM, 5 agents)
    nl2sql_validate_sql       SQL       -> read-only verdict   (no LLM)
    nl2sql_execute_query      SQL       -> rows                (no LLM)
    nl2sql_explain_result     rows      -> plain-language text (LLM)
    nl2sql_query_database     question  -> SQL + rows          (all-in-one)

Granular tools let a calling agent make a decision between steps: inspect the
generated SQL, edit it, refuse to run it, or run its own SQL entirely. The
all-in-one tool predates the split and is kept so existing client configs keep
working.

READ-ONLY IS ENFORCED AT THE TOOL BOUNDARY
------------------------------------------
validate and execute accept SQL from ANY caller, so they run the three-layer
gate in src/db/sql_guard.py on every call: a compile-time SQLite authorizer
that permits only reads, then PRAGMA query_only, then a mode=ro connection.
execute re-runs the gate itself rather than trusting that validate was called
first -- "validated SQL" is a claim by the caller, not a fact.

WHY THIS IS A PROXY AND NOT AN IN-PROCESS AGENT
-----------------------------------------------
This module does NOT import `src.graph` or call `build_graph()`. It forwards
HTTP requests to the FastAPI service in `api/main.py` instead. Two reasons:

1. Memory. `build_graph()` pulls in Chroma plus the ONNX embedder used for
   schema retrieval. `k8s/03-api.yaml` sizes the API container at a 1Gi limit
   precisely because 512Mi got OOM-killed (exit 137) on the first real query
   with exactly ONE copy of that stack resident. Building the graph here too
   would mean a second full copy to compute answers the API already computes.

2. Drift. An in-process copy is a second code path. As a proxy, there is
   exactly one implementation of each step, and MCP callers get the same
   answers as the REST API by construction. That includes the safety gate:
   it lives in the API, so calling the API directly bypasses nothing.

Splitting into seven tools did not change this: each tool maps to one API
endpoint, and the graph is still built exactly once, in the API process.

RUNNING IT
----------
    uvicorn api.main:app --port 8000     # the real engine, started separately
    python -m mcp_server.server          # this process, speaking stdio MCP

See mcp_server/README.md for client configuration.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Annotated, Any

import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from src.db.snowflake_connection import get_snowflake_connection

# The server name is part of the MCP handshake. Hosts display it and use it to
# namespace tools, so it should be stable and unique across the servers a user
# has installed. The nl2sql_ prefix on every tool serves the same purpose: a
# host with several servers loaded sees one flat tool list, and an unprefixed
# "execute_query" would collide with any other database server's.
mcp = FastMCP("nl2sql_mcp")

# Where the real engine lives. Env-configurable because the same image runs in
# three places with three different addresses: localhost during development,
# the `nl2sql-api` ClusterIP Service in Kubernetes, and whatever host a user
# points at when running this server from a desktop MCP client.
API_URL = os.getenv("NL2SQL_API_URL", "http://localhost:8000").rstrip("/")

# A full agent run is a planner call, a generator call, execution, and up to
# MAX_RETRIES correction rounds -- each round is another LLM round trip. httpx
# defaults to 5 seconds, which would abort nearly every real query, so the
# timeout is raised well past the worst realistic case.
TIMEOUT_SECONDS = float(os.getenv("NL2SQL_MCP_TIMEOUT", "180"))

# Logged arguments are capped: explain_result's arguments include result
# rows, and the audit trail should record what was asked, not mirror every
# result set into Snowflake.
MAX_LOGGED_ARG_CHARS = 16_000

# The six BIRD-SQL Mini-Dev databases committed to this repo. The full Mini-Dev
# set has eleven; the other five are excluded by .gitignore because they blow
# past GitHub's file-size limit. Keeping this list as a literal (rather than
# scanning the filesystem, as ui/app.py does) is deliberate: this process may
# run on a different machine from the API and has no view of its disk.
DATABASES: dict[str, str] = {
    "california_schools": (
        "California public schools with SAT/ACT scores, funding, and free-lunch "
        "eligibility, joined to per-school administrative records."
    ),
    "formula_1": (
        "Formula 1 racing history: races, circuits, drivers, constructors, "
        "lap times, pit stops, qualifying, and championship standings."
    ),
    "student_club": (
        "A university club's operations: members, majors, events, attendance, "
        "budgets, and expense reimbursements."
    ),
    "superhero": (
        "Comic-book superheroes with publishers, powers, alignments, races, "
        "and physical attributes across normalized lookup tables."
    ),
    "thrombosis_prediction": (
        "Anonymized medical records for thrombosis research: patients, "
        "examinations, and time-series laboratory test results."
    ),
    "toxicology": (
        "Molecular chemistry data: molecules labeled carcinogenic or not, with "
        "their constituent atoms and the bonds between them."
    ),
}

DbId = Annotated[
    str,
    Field(min_length=1, description="Which database. Call nl2sql_list_databases for valid values."),
]


# --- QUERY_HISTORY audit trail ----------------------------------------------
#
# One Snowflake connection for the life of the process, created on first use.
# Establishing a session costs roughly 1-2 seconds; tools are called many times
# per MCP session and the process stays alive between calls, so reconnecting
# per call would add that cost to every call for nothing.
#
# Lazy rather than at import: a client running with no Snowflake credentials
# should not pay for -- or fail on -- a connection it never uses.
_snowflake_conn = None
_snowflake_lock = threading.Lock()

logger = logging.getLogger("nl2sql_mcp")

# QUERY_HISTORY write outcomes for this process. A failed write is swallowed so
# it cannot break a tool call, which makes these counters -- returned in the
# capability manifest -- the one way a caller can tell the audit trail has gaps.
# In-memory, so they reset when the server restarts.
_telemetry_lock = threading.Lock()
_telemetry: dict[str, Any] = {"writes_ok": 0, "writes_failed": 0, "last_failure": None}


def _note_telemetry_ok() -> None:
    with _telemetry_lock:
        _telemetry["writes_ok"] += 1


def _note_telemetry_failure(tool: str, exc: Exception) -> None:
    with _telemetry_lock:
        _telemetry["writes_failed"] += 1
        _telemetry["last_failure"] = {
            "tool": tool,
            "error_type": type(exc).__name__,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    # Type only at WARNING: the message can carry connection details, and the
    # manifest this feeds is read by a model. The full text is at DEBUG.
    logger.warning("QUERY_HISTORY insert failed for %s (%s)", tool, type(exc).__name__)
    logger.debug("QUERY_HISTORY insert failure detail", exc_info=exc)


def _telemetry_snapshot() -> dict[str, Any]:
    with _telemetry_lock:
        last = _telemetry["last_failure"]
        return {**_telemetry, "last_failure": dict(last) if last else None}


def _get_logging_connection():
    """
    Return the process-wide Snowflake connection, opening it if needed.

    The lock guards creation only. FastMCP dispatches sync tools onto a thread
    pool, so two concurrent calls could otherwise open two sessions and leak
    one. Sharing a single connection across threads is the connector's
    supported pattern as long as each thread uses its own cursor, which the
    caller below does.

    A connection that has been closed or expired server-side is replaced
    rather than reused -- a long-idle MCP session would otherwise start
    failing every log write after the first timeout.
    """
    global _snowflake_conn
    with _snowflake_lock:
        if _snowflake_conn is None or _snowflake_conn.is_closed():
            _snowflake_conn = get_snowflake_connection()
        return _snowflake_conn


def _client_info(ctx: Context | None) -> tuple[str | None, str | None]:
    """
    Name and version the client declared in its MCP `initialize` handshake.

    This is self-reported -- any client can call itself anything -- so it
    identifies which integration made a call, not who the human behind it
    is. It is still the right field for an audit trail of a tool surface:
    "which host drove this" is exactly what a misbehaving integration needs
    traced back to.
    """
    try:
        info = ctx.session.client_params.clientInfo  # type: ignore[union-attr]
        return info.name, info.version
    except Exception:  # noqa: BLE001 -- no session (direct call) or older SDK
        return None, None


def _serialise_args(args: dict) -> str:
    text = json.dumps(args, default=str)
    if len(text) <= MAX_LOGGED_ARG_CHARS:
        return text
    return json.dumps(
        {"_truncated": True, "_original_chars": len(text), "preview": text[:MAX_LOGGED_ARG_CHARS]}
    )


def _log_tool_call(
    ctx: Context | None,
    tool: str,
    arguments: dict,
    *,
    success: bool,
    db_id: str | None = None,
    question: str | None = None,
    sql_generated: str | None = None,
    retries: int | None = None,
    error_message: str | None = None,
) -> None:
    """
    Record one tool call in QUERY_HISTORY. Never raises.

    Every tool logs, including refusals and discovery calls -- a refused
    DROP is the single most important event an audit trail can hold, and a
    trail that only recorded successes would hide it.

    Telemetry must not be able to break the thing it observes: if Snowflake is
    unreachable, credentials are missing, or the insert fails, the caller
    still gets their result. So every failure here is swallowed after being
    reported: a WARNING log (stderr, because under the stdio transport stdout
    IS the JSON-RPC stream and a stray print() would corrupt it) and a counter
    in the capability manifest's `telemetry` field.

    invoked_at is left to the column's DEFAULT CURRENT_TIMESTAMP() rather than
    bound from this process, so the recorded time comes from one clock
    (Snowflake's) no matter which machine the MCP client runs on.
    """
    client_name, client_version = _client_info(ctx)
    try:
        conn = _get_logging_connection()
        with conn.cursor() as cur:
            # NOTE: "source" is deliberately unquoted -- the DDL created this
            # column unquoted, so Snowflake folds it to SOURCE and this matches.
            # If the column is ever recreated as a quoted "source" (lowercase),
            # this INSERT breaks with "invalid identifier".
            #
            # INSERT ... SELECT rather than VALUES: Snowflake does not allow
            # PARSE_JSON() inside a VALUES clause, and `arguments` is VARIANT.
            cur.execute(
                """
                INSERT INTO QUERY_HISTORY (
                    db_id, question, sql_generated, success, retries, source,
                    mcp_tool, client_name, client_version, arguments, error_message
                )
                SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
                """,
                (
                    db_id, question, sql_generated, success, retries, "mcp",
                    tool, client_name, client_version, _serialise_args(arguments), error_message,
                ),
            )
        conn.commit()
    except Exception as exc:  # noqa: BLE001 -- deliberately total; see docstring
        try:
            _note_telemetry_failure(tool, exc)
        except Exception as report_exc:  # noqa: BLE001 -- the reporter must not raise either
            print(
                f"[nl2sql_mcp] QUERY_HISTORY insert failed for {tool} ({type(exc).__name__}); "
                f"reporting it also failed ({type(report_exc).__name__})",
                file=sys.stderr,
            )
    else:
        _note_telemetry_ok()


# --- HTTP proxy ---------------------------------------------------------------


def _call_api(method: str, path: str, payload: dict | None = None) -> tuple[dict | None, str | None]:
    """
    One request to the API. Returns (data, error) -- exactly one is None.

    Tool results are read by a model, not by exception-handling code, so
    transport failures come back as an error string the tool can put in its
    result, never as a raised exception. A model that gets a structured
    failure can explain it or retry; one that gets a protocol error usually
    just stops.
    """
    try:
        response = httpx.request(method, f"{API_URL}{path}", json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except httpx.TimeoutException:
        return None, (
            f"The NL2SQL API at {API_URL} did not respond within {TIMEOUT_SECONDS:.0f}s. "
            "A question needing several self-correction rounds can be slow; raise "
            "NL2SQL_MCP_TIMEOUT if this recurs."
        )
    except httpx.HTTPStatusError as exc:
        # Include the body: FastAPI puts the actual cause in there (e.g. the
        # 429 daily-cap message), and a bare status code explains nothing.
        return None, f"NL2SQL API returned HTTP {exc.response.status_code}: {exc.response.text}"
    except httpx.HTTPError as exc:
        return None, (
            f"Could not reach the NL2SQL API at {API_URL}: {exc}. This MCP server is a "
            "proxy; start the API with `uvicorn api.main:app --port 8000` or point "
            "NL2SQL_API_URL at a running instance."
        )
    except ValueError as exc:
        return None, f"NL2SQL API returned a non-JSON response: {exc}"


def _unknown_db(db_id: str) -> str | None:
    if db_id in DATABASES:
        return None
    return (
        f"Unknown db_id {db_id!r}. Available: {', '.join(sorted(DATABASES))}. "
        "Call nl2sql_list_databases for descriptions."
    )


# --- discovery ----------------------------------------------------------------


@mcp.tool(
    annotations={
        "title": "List available databases",
        "readOnlyHint": True,
        "idempotentHint": True,  # a constant: same output every call
        "openWorldHint": False,
    }
)
def nl2sql_list_databases(ctx: Context = None) -> dict[str, str]:
    """List the databases the other nl2sql tools can query, with descriptions.

    Call this FIRST when you do not already know which `db_id` fits the user's
    question. The descriptions say what subject matter each database covers, so
    a question about lap times routes to `formula_1` and one about lab results
    routes to `thrombosis_prediction`.

    These are six databases from the BIRD-SQL Mini-Dev benchmark, the set this
    project is evaluated against. The list is a static constant rather than an
    API call, so discovery still works when the backend is down.

    Returns:
        A dict mapping each valid `db_id` to a one-line description.
    """
    _log_tool_call(ctx, "nl2sql_list_databases", {}, success=True)
    return DATABASES


# Per-tool safety declarations for the capability manifest. Declared here,
# next to the tools, and checked against the live tool registry when the
# manifest is built -- a tool registered without an entry here is reported as
# undeclared rather than silently omitted.
_TOOL_SAFETY: dict[str, dict[str, Any]] = {
    "nl2sql_list_databases": {
        "side_effects": "none",
        "enforces": ["static list; no backend call"],
        "refuses": [],
    },
    "nl2sql_get_capabilities": {
        "side_effects": "none",
        "enforces": ["schemas read from the live tool registry; limits read from the enforcing API"],
        "refuses": [],
    },
    "nl2sql_generate_sql": {
        "side_effects": "LLM API calls (planner, generator, and classifier per retry)",
        "enforces": [
            "db_id restricted to the six bundled databases",
            "runs the full five-agent graph, so returned SQL is the self-corrected output the benchmark measured",
            "internal trial executions use the pipeline's two runtime read-only layers (PRAGMA query_only, mode=ro)",
            "daily spending cap when DEMO_LIMITS_ENABLED=true",
        ],
        "refuses": ["unknown db_id", "calls beyond the daily cap (HTTP 429, when enabled)"],
    },
    "nl2sql_validate_sql": {
        "side_effects": "none -- the statement is compiled, never run",
        "enforces": [
            "compile-time SQLite authorizer: only READ, SELECT, FUNCTION, RECURSIVE",
            "single statement only",
            "must compile against the target database's real schema",
        ],
        "refuses": [
            "INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ATTACH, DETACH, PRAGMA, TRANSACTION, SAVEPOINT",
            "stacked statements (SELECT ...; DELETE ...)",
            "load_extension()",
            "SQL that does not compile (syntax errors, unknown tables or columns)",
        ],
    },
    "nl2sql_execute_query": {
        "side_effects": "none -- reads only",
        "enforces": [
            "re-runs the full validate gate itself; never trusts a prior validate call",
            "authorizer stays live on the executing connection",
            "PRAGMA query_only = ON (statement-level)",
            "mode=ro database URI (file-level)",
            "row ceiling and wall-clock timeout (see live_enforcement)",
        ],
        "refuses": [
            "everything nl2sql_validate_sql refuses",
            "queries exceeding the timeout (aborted mid-execution)",
            "max_rows < 1",
        ],
    },
    "nl2sql_explain_result": {
        "side_effects": "one LLM API call",
        "enforces": [
            "summary restricted to facts present in the supplied rows; IDs reported as IDs",
            "at most 50 rows sent to the model; output flags when it covers a sample",
            "daily spending cap when DEMO_LIMITS_ENABLED=true",
        ],
        "refuses": ["calls beyond the daily cap (HTTP 429, when enabled)"],
    },
    "nl2sql_query_database": {
        "side_effects": "LLM API calls, as nl2sql_generate_sql",
        "enforces": [
            "same pipeline and read-only runtime layers as nl2sql_generate_sql",
            "kept for backward compatibility; does NOT apply the compile-time authorizer or row ceiling",
        ],
        "refuses": ["unknown db_id"],
    },
}

_SERVER_GUARANTEES = [
    "No tool can write to any database. Writes are refused at compile time by the "
    "SQLite authorizer, and independently blocked at runtime by PRAGMA query_only "
    "and a mode=ro connection -- each layer sufficient on its own.",
    "The safety gate runs in the API, not in this proxy, so calling the API directly bypasses nothing.",
    "Every tool call -- including refusals and discovery -- is logged to "
    "Snowflake QUERY_HISTORY with tool name, declared client, and arguments.",
    "Logging failures never fail a tool call. They are logged as a WARNING on stderr and "
    "counted in this manifest's `telemetry` field, so gaps in the audit trail are visible.",
]


@mcp.resource("nl2sql://capabilities", mime_type="application/json")
async def capabilities_resource() -> str:
    """The capability manifest, as an MCP resource for clients that read resources."""
    return json.dumps(await _build_manifest(), indent=2)


@mcp.tool(
    annotations={
        "title": "Describe tool capabilities and safety constraints",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def nl2sql_get_capabilities(ctx: Context = None) -> dict:
    """Return this server's capability manifest: every tool's input schema, what
    safety constraints it enforces, and what it refuses to do.

    Call this before relying on the server for anything with consequences --
    it is how a calling agent can confirm, rather than assume, that this server
    cannot write to a database.

    Also published as the MCP resource `nl2sql://capabilities` for clients
    that read resources; this tool exists because many hosts expose tools to
    their model but not resources.

    The manifest cannot drift from the code: input schemas are read from the
    live tool registry, and numeric limits (row ceiling, timeout, spending cap)
    are fetched from the API that enforces them. If the API is unreachable,
    `live_enforcement` reports the error instead of guessing.
    """
    manifest = await _build_manifest()
    _log_tool_call(ctx, "nl2sql_get_capabilities", {}, success=True)
    return manifest


async def _build_manifest() -> dict:
    registered = {t.name: t for t in await mcp.list_tools()}
    live, live_error = _call_api("GET", "/capabilities")

    tools = []
    for name in sorted(registered):
        tool = registered[name]
        safety = _TOOL_SAFETY.get(name)
        tools.append(
            {
                "name": name,
                "title": tool.annotations.title if tool.annotations else None,
                "annotations": tool.annotations.model_dump(exclude_none=True) if tool.annotations else {},
                "input_schema": tool.inputSchema,
                "side_effects": safety["side_effects"] if safety else "UNDECLARED",
                "enforces": safety["enforces"] if safety else [],
                "refuses": safety["refuses"] if safety else [],
            }
        )

    return {
        "server": "nl2sql_mcp",
        "transport_note": "thin HTTP proxy; all enforcement happens in the API at " + API_URL,
        "guarantees": _SERVER_GUARANTEES,
        "telemetry": _telemetry_snapshot(),
        "tools": tools,
        "undeclared_tools": sorted(set(registered) - set(_TOOL_SAFETY)),
        "live_enforcement": live if live is not None else {"unavailable": live_error},
    }


# --- the pipeline, step by step -------------------------------------------------


@mcp.tool(
    annotations={
        "title": "Generate SQL from a natural-language question",
        "readOnlyHint": True,
        # Three of the five agents call an LLM, so identical input can return
        # different SQL across calls. Advertising idempotency would invite
        # hosts to cache repeat calls and hide that variance.
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def nl2sql_generate_sql(
    question: Annotated[str, Field(min_length=1, description="The question, in plain English.")],
    db_id: DbId,
    evidence: Annotated[
        str | None,
        Field(description="Optional domain hint mapping vague wording to concrete columns."),
    ] = None,
    ctx: Context = None,
) -> dict:
    """Turn a natural-language question into SQL, WITHOUT returning result rows.

    Use this when you want to see -- and possibly check, edit, or decline to
    run -- the SQL before executing it. Follow with nl2sql_validate_sql and
    nl2sql_execute_query. To go straight from question to rows in one call,
    use nl2sql_query_database instead.

    HOW THE SQL IS PRODUCED
    -----------------------
    The full five-agent pipeline runs: schema retrieval (vector search for the
    relevant tables), query planning, SQL generation, trial execution, and --
    on failure -- an error classifier that routes back to whichever agent
    caused it, up to the server's retry budget.

    The trial execution is the reason this is not "just the generator". Self-
    correction is driven by what went wrong when the SQL ran, so the pipeline
    has to run it to correct it. The SQL returned is the corrected output, and
    it is what the project's 44.20% BIRD-SQL benchmark figure describes. Those
    internal runs are read-only (PRAGMA query_only plus a mode=ro connection).

    WHAT THE FIELDS MEAN
    --------------------
    - `sql`: the final statement, or null if none was produced.
    - `pipeline_trial_succeeded`: the SQL ran without error inside the
      pipeline. It does NOT mean the SQL answers the question correctly --
      valid SQL against the wrong column still succeeds.
    - `retries`: self-correction rounds used. Non-zero is normal.
    - `trial_error`: the last execution error, when the trial failed.

    `evidence` only reaches agent prompts if the server has
    INCLUDE_EVIDENCE_IN_PROMPTS enabled; otherwise it is accepted and ignored.
    """
    args = {"question": question, "db_id": db_id, "evidence": evidence}
    if err := _unknown_db(db_id):
        _log_tool_call(ctx, "nl2sql_generate_sql", args, success=False, db_id=db_id, question=question, error_message=err)
        return {"success": False, "error": err}

    data, err = _call_api("POST", "/sql/generate", args)
    if err:
        _log_tool_call(ctx, "nl2sql_generate_sql", args, success=False, db_id=db_id, question=question, error_message=err)
        return {"success": False, "error": err}

    result = {"success": data.get("sql") is not None, **data, "error": None}
    _log_tool_call(
        ctx, "nl2sql_generate_sql", args,
        success=result["success"], db_id=db_id, question=question,
        sql_generated=data.get("sql"), retries=data.get("retries"),
        error_message=data.get("trial_error"),
    )
    return result


@mcp.tool(
    annotations={
        "title": "Check that SQL is a safe, read-only statement",
        "readOnlyHint": True,
        "idempotentHint": True,  # same SQL against the same schema, same verdict
        "openWorldHint": False,
    }
)
def nl2sql_validate_sql(
    db_id: DbId,
    sql: Annotated[str, Field(min_length=1, description="A single SQL statement to check.")],
    ctx: Context = None,
) -> dict:
    """Check whether SQL is a single, read-only statement that compiles against
    the given database -- without running it.

    Use this on ANY SQL before execution, and especially on SQL you did not
    generate with nl2sql_generate_sql: SQL a user typed, SQL from a document,
    or SQL you edited.

    The statement is compiled by SQLite with an authorizer that permits only
    reads. Anything else -- INSERT, UPDATE, DELETE, DROP, CREATE, ALTER,
    ATTACH, PRAGMA, transactions -- is refused at compile time, before any row
    is touched, with the specific action named in `reason`. Because SQLite's
    own parser does the checking, a write hidden inside a CTE or subquery is
    still caught. Stacked statements (`SELECT 1; DELETE ...`) are refused
    outright. Compiling also checks that every table and column exists.

    A refusal is a normal answer, not an error: `valid` is false and
    `refusal_category` says why (not_read_only, multiple_statements,
    does_not_compile, empty).
    """
    args = {"db_id": db_id, "sql": sql}
    if err := _unknown_db(db_id):
        _log_tool_call(ctx, "nl2sql_validate_sql", args, success=False, db_id=db_id, error_message=err)
        return {"valid": False, "refusal_category": "unknown_db", "reason": err, "error": None}

    data, err = _call_api("POST", "/sql/validate", args)
    if err:
        _log_tool_call(ctx, "nl2sql_validate_sql", args, success=False, db_id=db_id, error_message=err)
        return {"valid": False, "error": err}

    _log_tool_call(
        ctx, "nl2sql_validate_sql", args,
        success=bool(data.get("valid")), db_id=db_id,
        error_message=None if data.get("valid") else data.get("reason"),
    )
    return {**data, "error": None}


@mcp.tool(
    annotations={
        "title": "Run read-only SQL and return rows",
        "readOnlyHint": True,
        # The bundled databases are static, so the same SQL returns the same
        # rows every time -- unlike generation, this IS safe to cache.
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def nl2sql_execute_query(
    db_id: DbId,
    sql: Annotated[str, Field(min_length=1, description="A single read-only SQL statement.")],
    max_rows: Annotated[
        int | None,
        Field(ge=1, description="Maximum rows to return. Defaults to the server default; capped at the server ceiling."),
    ] = None,
    ctx: Context = None,
) -> dict:
    """Execute a read-only SQL statement and return its rows.

    This tool re-runs the full nl2sql_validate_sql gate itself on every call.
    It does not trust that the SQL was validated first -- the tools are
    separately callable, so that would be a claim, not a fact. The
    compile-time authorizer also stays live on the connection that actually
    executes, and the connection itself is read-only twice over (PRAGMA
    query_only, mode=ro). Writes are not possible through this tool.

    Results are bounded: at most `max_rows` rows (server default and ceiling
    are listed by nl2sql_get_capabilities), and a wall-clock timeout after
    which the query is aborted. `truncated` is true when more rows existed
    than were returned -- treat a truncated result as partial.

    `rows` is a list of lists in `columns` order. An empty list means the
    query ran and matched nothing, which is a real answer.
    """
    args = {"db_id": db_id, "sql": sql, "max_rows": max_rows}
    if err := _unknown_db(db_id):
        _log_tool_call(ctx, "nl2sql_execute_query", args, success=False, db_id=db_id, error_message=err)
        return {"success": False, "refusal_category": "unknown_db", "reason": err, "error": None}

    data, err = _call_api("POST", "/sql/execute", args)
    if err:
        _log_tool_call(ctx, "nl2sql_execute_query", args, success=False, db_id=db_id, error_message=err)
        return {"success": False, "error": err}

    _log_tool_call(
        ctx, "nl2sql_execute_query", args,
        success=bool(data.get("success")), db_id=db_id,
        error_message=None if data.get("success") else data.get("reason"),
    )
    return {**data, "error": None}


@mcp.tool(
    annotations={
        "title": "Summarise a query result in plain language",
        "readOnlyHint": True,
        "idempotentHint": False,  # an LLM call: wording varies between calls
        "openWorldHint": False,
    }
)
def nl2sql_explain_result(
    columns: Annotated[list[str], Field(description="Column names, as returned by nl2sql_execute_query.")],
    rows: Annotated[list[list[Any]], Field(description="Result rows, as returned by nl2sql_execute_query.")],
    question: Annotated[str | None, Field(description="The question the query was meant to answer.")] = None,
    sql: Annotated[str | None, Field(description="The SQL that produced the rows.")] = None,
    truncated: Annotated[bool, Field(description="Pass through nl2sql_execute_query's `truncated` flag.")] = False,
    ctx: Context = None,
) -> dict:
    """Summarise query results in two to four plain sentences.

    The summary is restricted to what the rows actually contain. It will not
    enrich them from general knowledge: given `constructorId = 6` it reports
    the ID, not a guessed team name, and it says so when the result does not
    really answer the question asked. Pass `question` for that check to work.

    Only the first 50 rows are summarised; `covers_full_result` is false when
    the summary describes a sample or a truncated result.

    This step is NOT part of the five-agent pipeline and is not covered by
    the benchmark numbers -- it is a separate, single LLM call.
    """
    args = {"question": question, "sql": sql, "columns": columns, "rows": rows, "truncated": truncated}
    data, err = _call_api("POST", "/sql/explain", args)
    if err:
        _log_tool_call(ctx, "nl2sql_explain_result", args, success=False, question=question, error_message=err)
        return {"success": False, "error": err}

    _log_tool_call(ctx, "nl2sql_explain_result", args, success=True, question=question)
    return {"success": True, **data, "error": None}


@mcp.tool(
    annotations={
        "title": "Answer a question with SQL and rows in one call",
        "readOnlyHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def nl2sql_query_database(
    question: Annotated[str, Field(min_length=1, description="The question, in plain English.")],
    db_id: DbId,
    evidence: Annotated[
        str | None,
        Field(description="Optional domain hint mapping vague wording to concrete columns."),
    ] = None,
    ctx: Context = None,
) -> dict:
    """Answer a question in one call: generate SQL, run it, return both.

    Convenience composite of nl2sql_generate_sql + nl2sql_execute_query, kept
    for clients configured before the server was split into granular tools.
    Prefer the granular tools when you want to inspect SQL before it runs.

    Note the difference in bounds: rows here come straight from the pipeline's
    own execution, which is read-only (PRAGMA query_only, mode=ro) but applies
    neither the compile-time authorizer nor a row ceiling. For untrusted or
    potentially large results, use the granular tools.

    `success` means the SQL executed cleanly, not that it answers the
    question correctly -- read the returned `sql` before trusting the numbers.
    """
    args = {"question": question, "db_id": db_id, "evidence": evidence}
    if err := _unknown_db(db_id):
        _log_tool_call(ctx, "nl2sql_query_database", args, success=False, db_id=db_id, question=question, error_message=err)
        return {"success": False, "error": err}

    data, err = _call_api("POST", "/query", args)
    if err:
        _log_tool_call(ctx, "nl2sql_query_database", args, success=False, db_id=db_id, question=question, error_message=err)
        return {"success": False, "error": err}

    # The REST API's `trace` is dropped: a verbose per-agent log for the
    # Streamlit debug panel that would burn the caller's context for nothing.
    result = {
        "success": data.get("success", False),
        "sql": data.get("sql"),
        "result": data.get("result"),
        "retries": data.get("retries", 0),
        "error": None if data.get("success") else "The agent could not produce SQL that executed successfully.",
    }
    _log_tool_call(
        ctx, "nl2sql_query_database", args,
        success=result["success"], db_id=db_id, question=question,
        sql_generated=result["sql"], retries=result["retries"], error_message=result["error"],
    )
    return result


if __name__ == "__main__":
    # stdio is FastMCP's default transport: the host launches this file as a
    # subprocess and speaks JSON-RPC over its stdin/stdout. Nothing may be
    # printed to stdout anywhere in this process -- a stray print() corrupts
    # the protocol stream. Use stderr for any debugging output.
    mcp.run()
