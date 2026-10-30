"""
Split BIRD Mini-Dev into a calibration set and a held-out set.

Calibration questions are where prompts, semantic definitions and few-shot
examples may be written and tuned. Held-out questions are only ever scored. The
split is stratified by database x difficulty so both sets cover all 11 databases
(the old "first 150 questions" slice covered 4).

Only question_id, db_id and difficulty are read from dev.json: nothing here
needs the held-out questions' SQL or evidence, and nothing here looks at them.

dev.json has 500 rows but 498 distinct question_ids (137 and 138 each appear
twice). The pipeline joins on question_id everywhere, so the split is by id: a
duplicated id lands in one split, both rows with it.

Usage:
    python -m scripts.make_split [--data-path DIR] [--out-dir DIR]

The output is a pure function of dev.json and SEED, so running it again
rewrites the committed files byte for byte.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

SEED = 20261026
CALIBRATION_ROWS = 150
DEFAULT_DATA_PATH = os.getenv("BENCHMARK_DATA_PATH", "./data/bird-mini-dev")
DEFAULT_OUT_DIR = "eval/splits"
SPLIT_NAMES = ("calibration", "heldout")


def load_cells(dev_json_path: str) -> tuple[dict, Counter]:
    """(db_id, difficulty) -> sorted distinct question_ids, plus rows per question_id."""
    with open(dev_json_path, encoding="utf-8") as f:
        examples = json.load(f)
    rows_per_id = Counter(ex["question_id"] for ex in examples)
    cells = defaultdict(set)
    for ex in examples:
        cells[(ex["db_id"], ex["difficulty"])].add(ex["question_id"])
    return {cell: sorted(ids) for cell, ids in sorted(cells.items())}, rows_per_id


def cell_quotas(cell_sizes: dict, target_ids: int) -> dict:
    """
    How many ids each cell gives to calibration: proportional (largest remainder),
    except that any cell with two or more ids gives at least one, so no
    database x difficulty combination is missing from calibration.
    """
    total = sum(cell_sizes.values())
    exact = {cell: size * target_ids / total for cell, size in cell_sizes.items()}
    floor = {cell: 1 if cell_sizes[cell] >= 2 else 0 for cell in cell_sizes}
    quota = {cell: max(int(exact[cell]), floor[cell]) for cell in cell_sizes}

    def remainder(cell):
        return exact[cell] - quota[cell]

    while sum(quota.values()) < target_ids:
        quota[max(quota, key=remainder)] += 1
    while sum(quota.values()) > target_ids:
        # Take back from the cell that is furthest above its proportional share.
        removable = [cell for cell in quota if quota[cell] > floor[cell]]
        quota[min(removable, key=remainder)] -= 1
    return quota


def make_split(dev_json_path: str, calibration_rows: int = CALIBRATION_ROWS) -> dict:
    cells, rows_per_id = load_cells(dev_json_path)
    total_rows = sum(rows_per_id.values())
    target_ids = round(len(rows_per_id) * calibration_rows / total_rows)
    quotas = cell_quotas({cell: len(ids) for cell, ids in cells.items()}, target_ids)

    rng = random.Random(SEED)
    calibration = []
    for cell, ids in cells.items():
        calibration.extend(rng.sample(ids, quotas[cell]))
    calibration_set = set(calibration)
    heldout = [qid for ids in cells.values() for qid in ids if qid not in calibration_set]

    with open(dev_json_path, "rb") as f:
        source_sha256 = hashlib.sha256(f.read()).hexdigest()

    def describe(name, ids):
        return {
            "split": name,
            "seed": SEED,
            "source": "dev.json",
            "source_sha256": source_sha256,
            "n_question_ids": len(ids),
            "n_rows": sum(rows_per_id[qid] for qid in ids),
            "question_ids": sorted(ids),
        }

    return {"calibration": describe("calibration", calibration), "heldout": describe("heldout", heldout)}


def render(split: dict) -> str:
    return json.dumps(split, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Split BIRD Mini-Dev into calibration and held-out question-ID files.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    splits = make_split(os.path.join(args.data_path, "dev.json"))
    os.makedirs(args.out_dir, exist_ok=True)
    for name in SPLIT_NAMES:
        path = os.path.join(args.out_dir, f"{name}_ids.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(render(splits[name]))
        print(f"{path}: {splits[name]['n_question_ids']} ids, {splits[name]['n_rows']} rows")


if __name__ == "__main__":
    main()
