"""
The calibration/held-out split is frozen, disjoint, covers everything, and can
be regenerated exactly.

The committed ID files are checked on their own (no data needed). Checks that
need the real dev.json skip when it is absent, as it is in CI: the dataset is
not committed. A synthetic dev.json covers the algorithm itself, including the
duplicated question_id that the real file has.
"""

import json
import os
from collections import Counter

import pytest

from scripts import make_split as ms

SPLITS_DIR = "eval/splits"
DEV_JSON = os.path.join(ms.DEFAULT_DATA_PATH, "dev.json")
needs_dev_json = pytest.mark.skipif(not os.path.exists(DEV_JSON), reason="dev.json is not on disk (gitignored)")


def committed(name):
    with open(os.path.join(SPLITS_DIR, f"{name}_ids.json"), encoding="utf-8") as f:
        return json.load(f)


def synthetic_dev(tmp_path):
    """3 dbs x 3 difficulties with uneven sizes, one tiny cell, and one duplicated question_id."""
    sizes = {("a", "simple"): 30, ("a", "moderate"): 20, ("a", "challenging"): 3,
             ("b", "simple"): 10, ("b", "moderate"): 25, ("b", "challenging"): 12,
             ("c", "simple"): 1, ("c", "moderate"): 9, ("c", "challenging"): 40}
    examples, next_id = [], 1
    for (db_id, difficulty), size in sizes.items():
        for _ in range(size):
            examples.append({"question_id": next_id, "db_id": db_id, "difficulty": difficulty,
                             "question": "q", "SQL": "SELECT 1", "evidence": ""})
            next_id += 1
    examples.append(dict(examples[0]))  # same question_id appears twice, as 137 and 138 do
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(examples), encoding="utf-8")
    return str(path), examples


def test_committed_splits_are_disjoint_and_cover_every_question():
    calibration, heldout = committed("calibration"), committed("heldout")
    cal_ids, held_ids = set(calibration["question_ids"]), set(heldout["question_ids"])

    assert not cal_ids & held_ids
    assert len(cal_ids) == calibration["n_question_ids"] == len(calibration["question_ids"])
    assert len(held_ids) == heldout["n_question_ids"] == len(heldout["question_ids"])
    assert calibration["n_rows"] + heldout["n_rows"] == 500
    assert len(cal_ids | held_ids) == 498  # 500 rows; question_ids 137 and 138 are each used twice
    assert calibration["seed"] == heldout["seed"] == ms.SEED


def test_committed_calibration_size_is_150_rows():
    assert abs(committed("calibration")["n_rows"] - ms.CALIBRATION_ROWS) <= 2


@needs_dev_json
def test_committed_splits_match_the_dev_json_they_were_made_from():
    with open(DEV_JSON, encoding="utf-8") as f:
        examples = json.load(f)
    assert {s["source_sha256"] for s in (committed("calibration"), committed("heldout"))} == {
        ms.make_split(DEV_JSON)["calibration"]["source_sha256"]
    }
    assert {ex["question_id"] for ex in examples} == (
        set(committed("calibration")["question_ids"]) | set(committed("heldout")["question_ids"])
    )


@needs_dev_json
def test_regenerating_reproduces_the_committed_files_exactly():
    regenerated = ms.make_split(DEV_JSON)
    for name in ms.SPLIT_NAMES:
        with open(os.path.join(SPLITS_DIR, f"{name}_ids.json"), encoding="utf-8", newline="") as f:
            assert f.read() == ms.render(regenerated[name])


@needs_dev_json
def test_every_database_and_difficulty_cell_is_in_proportion():
    with open(DEV_JSON, encoding="utf-8") as f:
        examples = json.load(f)
    calibration = set(committed("calibration")["question_ids"])
    cells = Counter((ex["db_id"], ex["difficulty"]) for ex in examples)
    in_calibration = Counter((ex["db_id"], ex["difficulty"]) for ex in examples if ex["question_id"] in calibration)
    fraction = committed("calibration")["n_rows"] / len(examples)

    for cell, size in cells.items():
        assert in_calibration[cell] >= 1, f"{cell} has no calibration question"
        assert size - in_calibration[cell] >= 1 or size == 1, f"{cell} has no held-out question"
        assert abs(in_calibration[cell] - size * fraction) <= 1.5, cell


def test_synthetic_split_is_disjoint_covering_and_reproducible(tmp_path):
    path, examples = synthetic_dev(tmp_path)
    first, second = ms.make_split(path, calibration_rows=50), ms.make_split(path, calibration_rows=50)

    assert first == second
    cal, held = set(first["calibration"]["question_ids"]), set(first["heldout"]["question_ids"])
    assert not cal & held
    assert cal | held == {ex["question_id"] for ex in examples}
    # Both rows of the duplicated id are counted, and they travel together.
    assert first["calibration"]["n_rows"] + first["heldout"]["n_rows"] == len(examples)


def test_synthetic_split_represents_every_cell_with_two_or_more_ids(tmp_path):
    path, examples = synthetic_dev(tmp_path)
    cal = set(ms.make_split(path, calibration_rows=50)["calibration"]["question_ids"])
    ids_by_cell = {}
    for ex in examples:
        ids_by_cell.setdefault((ex["db_id"], ex["difficulty"]), set()).add(ex["question_id"])

    for cell, ids in ids_by_cell.items():
        if len(ids) >= 2:
            assert ids & cal, f"{cell} missing from calibration"
            assert ids - cal, f"{cell} missing from held-out"


def test_cell_quotas_sum_to_target_and_keep_the_floor():
    sizes = {"big": 200, "mid": 60, "small": 3, "tiny": 2, "one": 1}
    for target in (40, 80, 100):
        quotas = ms.cell_quotas(sizes, target)
        assert sum(quotas.values()) == target
        assert quotas["small"] >= 1 and quotas["tiny"] >= 1
        assert quotas["one"] <= 1
