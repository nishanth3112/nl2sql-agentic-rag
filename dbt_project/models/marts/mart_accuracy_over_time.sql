-- One row per benchmark run: the time series of overall accuracy.
--
-- Grain is deliberately the RUN, not run x database or run x difficulty.
-- Airflow's drift check reads this table and compares consecutive runs; if it
-- were further split, a "drop in accuracy" could just be a shift in which
-- databases a run covered rather than a real regression. Use
-- mart_accuracy_by_database / _by_difficulty for the breakdowns.
--
-- COVERAGE IS NOT CONSTANT ACROSS RUNS -- read `accuracy` with the two
-- coverage columns below. The two seeded runs do not cover the same databases:
--   baseline       spans 11 databases (500 questions)
--   with_evidence  spans  4 databases (150 questions):
--                  debit_card_specializing, european_football_2,
--                  student_club, thrombosis_prediction
-- Those 4 were already above-average performers under baseline, so the raw
-- 0.4420 -> 0.5267 delta visible in this table is NOT a controlled
-- comparison. It conflates "evidence helps" with "this run skipped the hard
-- databases". The README's +8.67pp evidence-ablation figure comes from a
-- MATCHED 150-question slice, which this mart does not reproduce.
--
-- TWO COVERAGE GUARDS, and the second exists because the first is not enough:
--
--   distinct_db_count -- how many databases the run covered. Makes the
--     confound machine-detectable rather than a footnote. But a COUNT is only
--     a size: two runs could each cover 4 databases with no overlap at all
--     and both report 4, so a check keying on this alone would compare them
--     happily. Necessary, not sufficient.
--
--   db_id_set_hash -- MD5 over the sorted, comma-joined distinct db_ids. A
--     fingerprint of the SET ITSELF, not its size, so the two disjoint
--     4-database runs above produce different hashes and are correctly
--     refused. DISTINCT collapses per-question duplicates; WITHIN GROUP
--     (ORDER BY db_id) makes the ordering deterministic, without which the
--     same set could hash differently between runs and the guard would
--     produce false mismatches.
--
-- A drift check should require db_id_set_hash equality before comparing two
-- runs' accuracy. distinct_db_count is kept alongside it because it is human
-- readable in a way a hash is not -- it tells you HOW the coverage differs.

with enriched as (

    select * from {{ ref('int_eval_results_enriched') }}

),

aggregated as (

    select
        run_id,
        run_timestamp,
        config_label,
        split,

        count(*)                                  as total_questions,
        count_if(correct)                         as correct_count,
        round(count_if(correct) / count(*), 4)    as accuracy,

        count(distinct db_id)                     as distinct_db_count,
        md5(listagg(distinct db_id, ',')
            within group (order by db_id))        as db_id_set_hash

    from enriched

    group by run_id, run_timestamp, config_label, split

)

select * from aggregated
order by run_timestamp
