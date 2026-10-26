# dbt project — `nl2sql_analytics`

Analytics layer over the Snowflake tables in `NL2SQL_ANALYTICS.RAW`, created by
[`snowflake_sql/schema.sql`](../snowflake_sql/schema.sql) and loaded by
[`eval/load_to_snowflake.py`](../eval/load_to_snowflake.py) and the MCP server.

## Why `dbt_project/` and not `dbt/`

`dbt` is the installed package's own import name. A top-level `dbt/` directory
would sit ahead of site-packages on `sys.path` for anything run from the repo
root — the same collision that forced `snowflake/` → `snowflake_sql/` here.

## Layout

```
dbt_project/
├── dbt_project.yml
├── macros/
│   └── generate_schema_name.sql     # STAGING / MARTS, not RAW_STAGING
├── models/
│   ├── staging/
│   │   ├── sources.yml              # raw source: NL2SQL_ANALYTICS.RAW
│   │   ├── stg_eval_runs.sql
│   │   ├── stg_eval_results.sql
│   │   └── stg_query_history.sql
│   ├── intermediate/
│   │   └── int_eval_results_enriched.sql
│   └── marts/
│       ├── mart_accuracy_by_database.sql
│       ├── mart_accuracy_by_difficulty.sql
│       ├── mart_accuracy_over_time.sql
│       └── mart_cost_per_correct.sql   # what each run cost, per correct answer
└── tests/
    ├── assert_accuracy_bounds.sql
    └── assert_cost_bounds.sql
```

Materialisation: `staging` and `intermediate` are views, `marts` are tables.
Staging models are thin pass-throughs whose only job is explicit type casting.

## Connection profile — you must create this by hand

`profiles.yml` holds connection credentials and **is deliberately not in this
repo**, same reasoning as the RSA private key. dbt looks for it at
`~/.dbt/profiles.yml` (on Windows, `C:\Users\<you>\.dbt\profiles.yml`).

Create that directory and file with:

```yaml
nl2sql_analytics:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: <your-account>.<region>

      user: NL2SQL_SERVICE_USER
      # MUST be an absolute path. dbt does NOT expand "~" -- it passes the
      # string straight to the connector, which then looks for a directory
      # literally named "~" and fails with:
      #   [Errno 2] No such file or directory: '~/.snowflake/rsa_key.p8'
      # (src/db/snowflake_connection.py calls Path.expanduser(), so "~" works
      # there; that difference is exactly what makes this easy to get wrong.)
      private_key_path: C:/Users/Arun/.snowflake/rsa_key.p8
      # Key is unencrypted (generated with -nocrypt), so no passphrase.

      role: NL2SQL_APP_ROLE
      warehouse: COMPUTE_WH
      database: NL2SQL_ANALYTICS
      schema: RAW

      threads: 4
      client_session_keep_alive: False
```

`schema: RAW` is the *fallback* schema for models that do not set `+schema:`.
The `generate_schema_name` macro routes staging and marts away from it.

## Running

```bash
cd dbt_project
dbt debug      # connection test
dbt run        # build models
```

## Cost columns: apply the migration first

`stg_eval_results`, `stg_eval_runs` and `int_eval_results_enriched` read the
cost-tracing columns (`llm_calls`, token counts, `est_cost_usd`, per-node
seconds, `pricing_as_of`). Run the migrations at the bottom of
[`snowflake_sql/schema.sql`](../snowflake_sql/schema.sql) **before** building:
those models are ancestors of `mart_accuracy_over_time`, so against a warehouse
that lacks the columns the weekly `dbt build` fails.

A NULL `est_cost_usd` means unknown, not zero (a crashed question, a model with
no listed price, or a run recorded before cost tracing). `mart_cost_per_correct`
exposes `costed_questions` beside `total_questions` so a partly-costed run does
not read as cheap. The Airflow DAG builds only `+mart_accuracy_over_time`, so
build the cost mart by hand for now: `dbt build --select mart_cost_per_correct`.
