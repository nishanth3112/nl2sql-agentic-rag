# Roadmap

Fork of `v-arun2002/nl2sql-agentic-rag`. One phase = one branch = one PR into `main`, with 5–10 commits each. Status values: planned, in progress, done.

## Already in place (extend, don't rebuild)

- Read-only execution: `mode=ro` + `PRAGMA query_only` on every connection, plus the compile-time authorizer in `src/db/sql_guard.py` at the MCP tool boundary.
- Seven-tool MCP server in `mcp_server/` with a capability manifest.
- Eval history in Snowflake (`EVAL_RUNS`, `EVAL_RESULTS`, `QUERY_HISTORY`), dbt marts in `dbt_project/`, and a weekly Airflow DAG whose drift check runs McNemar only when `db_id_set_hash` matches.
- Result summaries in `src/explain.py`.

## Gaps this roadmap targets (documented in the README)

- The 44.20% headline and the +8.67pp evidence ablation were measured on a configuration that no longer exists. No fresh baseline yet.
- Wrong-but-valid SQL, e.g. returning `constructorId` instead of the constructor's name.
- `evidence` reaches the planner only, never the generator.
- The UI is Streamlit; it will be replaced by a Lovable-built frontend (Phases 7–8).

## Phases

| Phase | Branch | Goal | Done when | Status |
|---|---|---|---|---|
| 0 | `chore/phase-0-foundation` | uv everywhere (root, Dockerfiles, Airflow image, CI, README); commit CLAUDE.md and this roadmap; ignore `.claude/` | `uv run pytest` passes; images build with the same resolved dependency set; no torch | done |
| 1 | `feat/phase-1-cost-tracing` | Per-node tracing plus token and cost accounting in the benchmark CSV and `EVAL_RESULTS` (additive, nullable columns) | An approved small slice records tokens and cost per question; old CSVs still load | done |
| 2 | `feat/phase-2-split-baseline` | Seeded calibration/held-out split, stratified by database and difficulty, with committed question IDs; harness split option; fresh baseline under the current config | Held-out baseline CSV committed and loaded; README headline re-dated with its config | done |
| 3 | `feat/phase-3-semantic-context` | Semantic definitions (formulas, value encodings, synonyms, entity display names) authored from BIRD `database_description` files and calibration questions only; indexed in Chroma; fed to planner and generator | Held-out comparison against Phase 2 with matching `db_id_set_hash` and McNemar, reported whichever way it goes | planned |
| 4 | `feat/phase-4-semantic-layer` | New dbt project `semantic_layer/` (separate from `dbt_project/`) on DuckDB with TPC-DS data and MetricFlow semantic models; about 100 business questions with gold SQL | `mf validate-configs` passes; the existing loop is baselined on the business set | planned |
| 5 | `feat/phase-5-semantic-router` | LangGraph router: questions covered by a metric compile through MetricFlow, the rest fall back to the existing loop; MCP tools to list and query metrics | Coverage and accuracy reported per path; new MCP tools carry safety declarations and are tested through a real client | planned |
| 6 | `feat/phase-6-verifier` | Multi-candidate generation with execution-result voting, result-shape checks (ID returned where a name was asked), clarification interrupt | Held-out comparison against the latest baseline | planned |
| 7 | `feat/phase-7-analyst-api` | API the new frontend will use: answer payload (summary, SQL and explanation, rows, chart hint, path taken, verifier status, cost), streamed agent trace, uploads moved behind the API, feedback into a verified-query store used as few-shot examples (calibration questions only), CORS allowlist; OpenAPI schema committed as the frontend contract. Streamlit keeps working | OpenAPI schema committed; API tests pass; few-shot effect measured on held-out | planned |
| 8 | `feat/phase-8-frontend` | Lovable-built frontend imported into `frontend/`, wired to the Phase 7 API through a typed client; CI build; static container in docker-compose and k8s; Streamlit retired once a parity checklist passes | Parity checklist passes in a browser; CI builds the frontend; Streamlit and `UI_DIRECT_MODE` removed; README updated | planned |
