USE ROLE NL2SQL_APP_ROLE;
USE DATABASE NL2SQL_ANALYTICS;
USE SCHEMA RAW;

CREATE TABLE IF NOT EXISTS EVAL_RUNS (
    run_id                       VARCHAR(36) PRIMARY KEY,
    run_timestamp                TIMESTAMP_NTZ NOT NULL,
    config_label                 VARCHAR(100),
    include_evidence_in_prompts  BOOLEAN,
    planner_model                VARCHAR(100),
    generator_model              VARCHAR(100),
    classifier_model             VARCHAR(100),
    total_questions               NUMBER
);

CREATE TABLE IF NOT EXISTS EVAL_RESULTS (
    result_id              NUMBER AUTOINCREMENT PRIMARY KEY,
    run_id                  VARCHAR(36) NOT NULL REFERENCES EVAL_RUNS(run_id),
    question_id             NUMBER,
    db_id                    VARCHAR(50),
    question                 VARCHAR,
    evidence                 VARCHAR,
    gold_sql                 VARCHAR,
    predicted_sql            VARCHAR,
    correct                  BOOLEAN,
    difficulty                VARCHAR(20),
    retries                   NUMBER,
    error_classes_hit         VARCHAR,
    fatal_error                VARCHAR,
    schema_context_chars       NUMBER,
    seconds                    FLOAT
);

CREATE TABLE IF NOT EXISTS QUERY_HISTORY (
    history_id      NUMBER AUTOINCREMENT PRIMARY KEY,
    invoked_at       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    db_id             VARCHAR(50),
    question           VARCHAR,
    sql_generated       VARCHAR,
    success              BOOLEAN,
    retries               NUMBER,
    source                 VARCHAR(20)
);

-- ---------------------------------------------------------------------------
-- Migrations. The CREATE statements above are the original DDL and are left
-- verbatim; columns added since are applied here with IF NOT EXISTS, so this
-- file is safe to re-run against an existing schema and complete on a fresh
-- one.
-- ---------------------------------------------------------------------------

-- Provider per role, and a content hash so eval/load_to_snowflake.py can
-- refuse to load the same CSV twice. Already live; recorded here so a fresh
-- environment built from this file does not break the loader.
ALTER TABLE EVAL_RUNS ADD COLUMN IF NOT EXISTS
    planner_provider     VARCHAR(50),
    generator_provider   VARCHAR(50),
    classifier_provider  VARCHAR(50),
    source_file_hash     VARCHAR(64);

-- Audit trail for the multi-tool MCP server: which tool, which client, with
-- what arguments, and why it was refused if it was. Rows logged before these
-- columns existed have them NULL.
ALTER TABLE QUERY_HISTORY ADD COLUMN IF NOT EXISTS
    mcp_tool        VARCHAR(64),
    client_name     VARCHAR(200),
    client_version  VARCHAR(64),
    arguments       VARIANT,
    error_message   VARCHAR;

-- Cost tracing: what each question cost and where its time went, from the
-- columns eval/run_benchmark.py now writes. NULL on rows loaded before they
-- existed, and NULL where the figure is unknown (the question crashed, or a
-- model has no listed price). A NULL est_cost_usd is "unknown", not zero:
-- count the costed questions before averaging.
-- Apply this BEFORE loading a CSV that has these columns, or before the dbt
-- models that read them are built; the loader refuses to start otherwise.
ALTER TABLE EVAL_RESULTS ADD COLUMN IF NOT EXISTS
    llm_calls             NUMBER,
    prompt_tokens         NUMBER,
    cached_prompt_tokens  NUMBER,
    completion_tokens     NUMBER,
    reasoning_tokens      NUMBER,
    est_cost_usd          NUMBER(18, 8),
    retriever_seconds     FLOAT,
    planner_seconds       FLOAT,
    generator_seconds     FLOAT,
    executor_seconds      FLOAT,
    classifier_seconds    FLOAT;

-- The date the prices behind est_cost_usd were read (src/pricing.py).
ALTER TABLE EVAL_RUNS ADD COLUMN IF NOT EXISTS
    pricing_as_of  VARCHAR(10);

-- Which committed split a run covered: calibration, heldout or all
-- (eval/splits/, written to the run metadata by eval/run_benchmark.py). NULL for
-- runs recorded before splits existed, such as the two seeded runs. Compare
-- runs only within the same split. Apply this BEFORE loading a CSV whose
-- metadata has a split, or the loader refuses to start.
ALTER TABLE EVAL_RUNS ADD COLUMN IF NOT EXISTS
    split  VARCHAR(20);
