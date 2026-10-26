-- Thin pass-through over RAW.EVAL_RESULTS. One row per benchmark question.
--
-- Note `correct` is a real boolean here because eval/load_to_snowflake.py
-- converts the CSV's "True"/"False" text before loading. Downstream models can
-- therefore aggregate it directly rather than comparing against strings.

with source as (

    select * from {{ source('raw', 'eval_results') }}

),

casted as (

    select
        cast(result_id            as number)       as result_id,
        cast(run_id               as varchar(36))  as run_id,
        cast(question_id          as number)       as question_id,
        cast(db_id                as varchar(50))  as db_id,

        cast(question             as varchar)      as question,
        cast(evidence             as varchar)      as evidence,
        cast(gold_sql             as varchar)      as gold_sql,
        cast(predicted_sql        as varchar)      as predicted_sql,

        cast(correct              as boolean)      as correct,
        cast(difficulty           as varchar(20))  as difficulty,
        cast(retries              as number)       as retries,

        cast(error_classes_hit    as varchar)      as error_classes_hit,
        cast(fatal_error          as varchar)      as fatal_error,
        cast(schema_context_chars as number)       as schema_context_chars,
        cast(seconds              as float)        as seconds,

        -- Cost tracing, added later (snowflake_sql/schema.sql). NULL on runs
        -- loaded before it existed, and NULL where the figure is unknown -- the
        -- question crashed, or a model has no listed price. NULL cost means
        -- "unknown", not zero.
        cast(llm_calls             as number(38, 0)) as llm_calls,
        cast(prompt_tokens         as number(38, 0)) as prompt_tokens,
        cast(cached_prompt_tokens  as number(38, 0)) as cached_prompt_tokens,
        cast(completion_tokens     as number(38, 0)) as completion_tokens,
        cast(reasoning_tokens      as number(38, 0)) as reasoning_tokens,
        cast(est_cost_usd          as number(18, 8)) as est_cost_usd,
        cast(retriever_seconds     as float)         as retriever_seconds,
        cast(planner_seconds       as float)         as planner_seconds,
        cast(generator_seconds     as float)         as generator_seconds,
        cast(executor_seconds      as float)         as executor_seconds,
        cast(classifier_seconds    as float)         as classifier_seconds

    from source

)

select * from casted
