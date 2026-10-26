-- Singular test: fails if it returns ANY rows.
--
-- Guards the arithmetic in mart_cost_per_correct the way assert_accuracy_bounds
-- guards the accuracy marts. Each condition is impossible if the mart is right:
--   costed_questions > total_questions, or costed_correct_count > correct_count
--     or > costed_questions
--     a subset can never be larger than the set it was taken from; firing means
--     the grouping changed under one of the aggregates, e.g. a join fanned out.
--   total_cost_usd < 0, or cost_per_correct_usd < 0
--     a price or token count went negative somewhere upstream.
--   costed_questions = 0 but total_cost_usd is not null
--     "no question had a known cost" must read as unknown, never as a figure.
--   cost_per_correct_usd is not null but costed_correct_count = 0
--     a ratio with nothing to divide by.

SELECT run_id, config_label, total_questions, correct_count,
       costed_questions, costed_correct_count, total_cost_usd, cost_per_correct_usd
FROM {{ ref('mart_cost_per_correct') }}
WHERE costed_questions > total_questions
   OR costed_correct_count > correct_count
   OR costed_correct_count > costed_questions
   OR total_cost_usd < 0
   OR cost_per_correct_usd < 0
   OR (costed_questions = 0 AND total_cost_usd IS NOT NULL)
   OR (costed_correct_count = 0 AND cost_per_correct_usd IS NOT NULL)
