"""
compare_ablation warns when two runs were not on the same split.

Files written before splits existed have no split column, so the committed
baseline-versus-ablation pair must stay silent; anything else that mixes known
and unknown, or two different splits, is a comparison of different questions.
"""

from scripts.compare_ablation import is_correct, mcnemar_exact_two_sided, split_warning


def rows(*splits):
    return [{"question_id": i, "split": split, "correct": "True"} for i, split in enumerate(splits)]


def test_two_runs_on_the_same_split_are_silent():
    assert split_warning(rows("heldout", "heldout"), rows("heldout")) is None


def test_two_runs_from_before_splits_existed_are_silent():
    assert split_warning(rows(None, None), [{"question_id": 1, "correct": "True"}]) is None
    assert split_warning(rows("", ""), rows("")) is None


def test_different_splits_warn():
    message = split_warning(rows("calibration"), rows("heldout"))

    assert "calibration" in message and "heldout" in message


def test_a_run_with_a_split_against_one_without_warns():
    message = split_warning(rows(None), rows("heldout"))

    assert "unknown" in message and "heldout" in message


def test_the_committed_pair_compares_without_a_split_warning():
    import csv

    def load(path):
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    baseline = load("eval/results_500_baseline.csv")
    ablation = load("eval/results_150_with_evidence.csv")

    assert split_warning(baseline, ablation) is None


def test_mcnemar_on_the_published_ablation_still_gives_p_0_0072():
    # 17 fixed, 4 broken: the headline in the README, which this guards against drifting.
    assert round(mcnemar_exact_two_sided(17, 4), 4) == 0.0072
    assert is_correct({"correct": "True"}) and not is_correct({"correct": "False"})
