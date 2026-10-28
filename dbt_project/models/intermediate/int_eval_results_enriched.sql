-- Per-question results with their run's configuration attached, so downstream
-- marts can slice accuracy by model/provider without re-joining every time.
--
-- INNER join is deliberate. Every result row should have a parent run --
-- eval/load_to_snowflake.py inserts EVAL_RUNS before loading EVAL_RESULTS, and
-- the FK is declared in the DDL. But Snowflake does not ENFORCE foreign keys,
-- so an orphan is possible in principle. An inner join drops orphans silently,
-- which is why the `relationships` test in models/staging/schema.yml exists:
-- it fails the build if any result's run_id is missing from stg_eval_runs,
-- turning a silent row-count discrepancy into a loud error.

with results as (

    select * from {{ ref('stg_eval_results') }}

),

runs as (

    select * from {{ ref('stg_eval_runs') }}

),

joined as (

    select
        -- everything from stg_eval_results
        results.result_id,
        results.run_id,
        results.question_id,
        results.db_id,
        results.question,
        results.evidence,
        results.gold_sql,
        results.predicted_sql,
        results.correct,
        results.difficulty,
        results.retries,
        results.error_classes_hit,
        results.fatal_error,
        results.schema_context_chars,
        results.seconds,

        -- cost tracing: NULL on older runs and where unknown
        results.llm_calls,
        results.prompt_tokens,
        results.cached_prompt_tokens,
        results.completion_tokens,
        results.reasoning_tokens,
        results.est_cost_usd,
        results.retriever_seconds,
        results.planner_seconds,
        results.generator_seconds,
        results.executor_seconds,
        results.classifier_seconds,

        -- run-level configuration
        runs.run_timestamp,
        runs.config_label,
        runs.planner_model,
        runs.generator_model,
        runs.classifier_model,
        runs.planner_provider,
        runs.generator_provider,
        runs.classifier_provider,
        runs.pricing_as_of,
        runs.split

    from results

    inner join runs
        on results.run_id = runs.run_id

)

select * from joined
