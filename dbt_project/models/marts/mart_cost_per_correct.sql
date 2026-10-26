-- One row per benchmark run: what it cost, and what each correct answer cost.
--
-- Grain is the RUN, like mart_accuracy_over_time. A per-database breakdown would
-- be a natural follow-up; it is left out so this stays one number per run.
--
-- UNKNOWN COST IS NOT ZERO. est_cost_usd is NULL for a question that crashed, for
-- one that called a model with no listed price, and for every question in a run
-- recorded before cost tracing existed. sum() skips NULLs, so a run whose cost is
-- only partly known would otherwise look cheaper than it was. Read
-- total_cost_usd with costed_questions next to total_questions: when they differ,
-- the cost covers only the costed share.
--
-- cost_per_correct_usd divides the known cost by the correct answers among the
-- SAME costed questions, so numerator and denominator cover the same rows. It is
-- NULL, not 0 or an error, when no costed question was correct.
--
-- Accuracy itself lives in mart_accuracy_over_time and the other accuracy marts;
-- compare runs there only when db_id_set_hash matches. This mart does not make
-- two runs comparable, it only prices each one.

with enriched as (

    select * from {{ ref('int_eval_results_enriched') }}

),

aggregated as (

    select
        run_id,
        run_timestamp,
        config_label,
        pricing_as_of,

        count(*)                                           as total_questions,
        count_if(correct)                                  as correct_count,

        count_if(est_cost_usd is not null)                 as costed_questions,
        count_if(correct and est_cost_usd is not null)     as costed_correct_count,
        sum(est_cost_usd)                                  as total_cost_usd,
        round(
            sum(est_cost_usd) / nullif(count_if(correct and est_cost_usd is not null), 0),
            6
        )                                                  as cost_per_correct_usd,

        sum(llm_calls)                                     as total_llm_calls,
        sum(prompt_tokens)                                 as total_prompt_tokens,
        sum(cached_prompt_tokens)                          as total_cached_prompt_tokens,
        sum(completion_tokens)                             as total_completion_tokens,
        sum(reasoning_tokens)                              as total_reasoning_tokens,

        round(avg(seconds), 2)                             as avg_seconds_per_question

    from enriched

    group by run_id, run_timestamp, config_label, pricing_as_of

)

select * from aggregated
order by run_timestamp
