# MCP server

Exposes the agentic NL2SQL pipeline to any MCP client (Claude Desktop, Claude
Code, an IDE, a custom agent) as a multi-tool MCP server, with read-only
enforcement at the tool boundary.

## Architecture

```
MCP client  ──stdio/JSON-RPC──▶  mcp_server/server.py  ──HTTP──▶  api/main.py  ──▶  LangGraph agents
                                 (thin proxy)                      (the engine +       + src/db/sql_guard.py
                                                                    the safety gate)
```

`server.py` is a **proxy**. It never imports `src.graph` and never calls
`build_graph()`. Two reasons:

- **Memory.** `build_graph()` loads Chroma plus the ONNX embedder. `k8s/03-api.yaml`
  sets the API container's limit to 1Gi because 512Mi was OOM-killed on the
  first real query with one copy of that stack resident. An in-process copy
  here would roughly double the footprint to compute answers the API already
  computes.
- **Drift.** One implementation of each step means MCP callers and REST callers
  cannot diverge. That includes the safety gate: it lives in the API, so
  calling the API directly bypasses nothing.

Splitting into granular tools did not change this — each tool maps to one API
endpoint, and the graph is still built exactly once.

The trade-off: **the API must be running and reachable.** When it is not, each
tool returns a structured failure naming the URL it tried, rather than failing
at the protocol level.

## Tools

| Tool | Endpoint | Does | Idempotent |
| --- | --- | --- | --- |
| `nl2sql_list_databases()` | — (static) | the six bundled `db_id`s, with descriptions | Yes |
| `nl2sql_get_capabilities()` | `GET /capabilities` | the capability manifest (below) | Yes |
| `nl2sql_generate_sql(question, db_id, evidence=None)` | `POST /sql/generate` | question → corrected SQL, **no rows** | **No** — LLM calls |
| `nl2sql_validate_sql(db_id, sql)` | `POST /sql/validate` | SQL → read-only verdict, without running it | Yes |
| `nl2sql_execute_query(db_id, sql, max_rows=None)` | `POST /sql/execute` | SQL → bounded rows; re-validates itself | Yes |
| `nl2sql_explain_result(columns, rows, question=None, sql=None, truncated=False)` | `POST /sql/explain` | rows → 2–4 plain sentences | **No** — LLM call |
| `nl2sql_query_database(question, db_id, evidence=None)` | `POST /query` | question → SQL + rows in one call (pre-split; kept for existing clients) | **No** |

All seven are annotated `readOnlyHint: True`, and all carry the `nl2sql_`
prefix: a host with several servers loaded sees one flat tool list, and an
unprefixed `execute_query` would collide with any other database server's.

### `generate_sql` is not "just the generator"

Self-correction is driven by execution errors — the classifier routes on what
went wrong when the SQL ran — so `generate_sql` runs the **full** five-agent
graph, including its internal read-only trial executions, and returns the
corrected SQL. Cutting the graph before the executor would return first-draft
SQL and quietly score below the benchmarked 44.20%. It makes the identical
`initial_state(...)` + `graph.invoke(...)` call as `eval/run_benchmark.py`.

### Read-only at the tool boundary

`validate_sql` and `execute_query` accept SQL from **any** caller, so they run
the gate in `src/db/sql_guard.py` on every call. Three layers, each sufficient
on its own:

1. **Compile-time SQLite authorizer.** SQLite consults it for every action
   while compiling a statement; only `READ`, `SELECT`, `FUNCTION` and
   `RECURSIVE` are allowed. Writes, DDL, `ATTACH`, `PRAGMA` and transactions
   are refused before a row is touched — and because SQLite's own parser does
   the checking, a `DELETE` hidden inside a CTE is still a `DELETE`.
   `load_extension()` passes the authorizer as an ordinary function call, so
   it is denied by name.
2. **`PRAGMA query_only = ON`** — statement-level.
3. **`mode=ro` URI** — file-level.

Stacked statements (`SELECT 1; DELETE …`) are refused outright. `execute_query`
re-runs the full gate itself and never trusts that `validate_sql` was called
first — the tools are separately callable, so "validated" is a claim by the
caller, not a fact. Execution is also bounded: a row ceiling (results land in
an LLM's context) and a wall-clock timeout.

The gate is a separate module rather than a change to `src/db/connection.py`,
so the code path the benchmark measured is untouched.

### Capability manifest

`nl2sql_get_capabilities` (also the resource `nl2sql://capabilities`, for hosts
that read resources) returns each tool's input schema, what it enforces, and
what it refuses. It is built so it cannot drift from the code: schemas come
from the live tool registry, numeric limits come from the API that enforces
them, and a tool registered without a safety declaration is reported under
`undeclared_tools` rather than silently omitted.

### Audit trail

Every tool call — including refusals and discovery calls — is written to
Snowflake `QUERY_HISTORY` with the tool name, the client's self-declared
`clientInfo` name and version, the arguments as `VARIANT` (capped at 16KB),
and the refusal reason. A refused `DROP` is the most important row an audit
trail can hold. Logging failures never fail a tool call; they are logged as a WARNING on stderr
and counted under `telemetry` in the capabilities manifest.

## Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `NL2SQL_API_URL` | `http://localhost:8000` | Base URL of the FastAPI service |
| `NL2SQL_MCP_TIMEOUT` | `180` | Per-call timeout in seconds; a run with several self-correction rounds is several LLM round trips |
| `SQL_DEFAULT_MAX_ROWS` | `200` | Rows returned by `execute_query` when `max_rows` is omitted (read by the API) |
| `SQL_MAX_ROWS_CEILING` | `5000` | Hard cap on `max_rows` (read by the API) |
| `SQL_TIMEOUT_SECONDS` | `10` | Wall-clock limit per executed query (read by the API) |
| `EXPLAINER_PROVIDER` / `EXPLAINER_MODEL` | the generator's | Model behind `explain_result` |

## Running

Start the engine first, then the MCP server:

```bash
uv run uvicorn api.main:app --port 8000
uv run python -m mcp_server.server
```

Client config (Claude Desktop `claude_desktop_config.json`, or
`claude mcp add` for Claude Code):

```json
{
  "mcpServers": {
    "nl2sql_mcp": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/nl2sql-agentic-rag",
      "env": { "NL2SQL_API_URL": "http://localhost:8000" }
    }
  }
}
```

`uv run` picks up the project's environment from `cwd`, so run `uv sync` there
once first.

## Limits worth knowing

- `clientInfo` is **self-declared** by the client in the `initialize`
  handshake. It identifies which integration made a call, not who the human
  is.
- `nl2sql_query_database` predates the gate: its rows come straight from the
  pipeline's executor, which is read-only (layers 2 and 3) but applies neither
  the authorizer nor a row ceiling. Use the granular tools for untrusted or
  potentially large results.
- `explain_result` is a separate LLM call outside the five-agent pipeline. It
  is not covered by the benchmark numbers.

## Note on stdio

The default transport is stdio: the host launches this file as a subprocess and
speaks JSON-RPC over stdin/stdout. **Nothing may print to stdout** anywhere in
this process — a stray `print()` corrupts the protocol stream. Use stderr.
