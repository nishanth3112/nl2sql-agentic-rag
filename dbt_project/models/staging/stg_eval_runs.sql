-- Thin pass-through over RAW.EVAL_RUNS. No filtering, no joins, no business
-- logic -- the only job here is to pin down types so downstream models can
-- rely on them instead of re-deriving them.

with source as (

    select * from {{ source('raw', 'eval_runs') }}

),

casted as (

    select
        cast(run_id                      as varchar(36))   as run_id,
        cast(run_timestamp               as timestamp_ntz) as run_timestamp,
        cast(config_label                as varchar(100))  as config_label,
        cast(include_evidence_in_prompts as boolean)       as include_evidence_in_prompts,

        cast(planner_provider            as varchar(50))   as planner_provider,
        cast(planner_model               as varchar(100))  as planner_model,
        cast(generator_provider          as varchar(50))   as generator_provider,
        cast(generator_model             as varchar(100))  as generator_model,
        cast(classifier_provider         as varchar(50))   as classifier_provider,
        cast(classifier_model            as varchar(100))  as classifier_model,

        cast(total_questions             as number)        as total_questions,
        cast(source_file_hash            as varchar(64))   as source_file_hash,

        -- Date of the prices behind est_cost_usd. Added with cost tracing; NULL
        -- on runs recorded before it.
        cast(pricing_as_of               as varchar(10))   as pricing_as_of

    from source

)

select * from casted
