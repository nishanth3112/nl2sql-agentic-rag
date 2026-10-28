"""
Runs the full agent graph against BIRD-SQL's Mini-Dev set, computes overall
+ per-difficulty + per-database execution accuracy, and writes a per-example
failure taxonomy to a CSV.

BIRD-SQL's confirmed dev.json fields (verified against a real exported
example): question_id, db_id, question, evidence, SQL (capital -- not
"query"), and difficulty. By default this runs WITHOUT feeding `evidence`
into the agents (see src/config.py's include_evidence_in_prompts) -- a
harder test of whether the pipeline works from schema alone.

Results checkpoint to disk every 10 questions. Each run writes a new
timestamped CSV and refuses to reuse an existing one; pass --resume (or
BENCHMARK_RESUME=1) with BENCHMARK_RESULTS_PATH to continue an interrupted run.
BENCHMARK_SPLIT picks the committed split (calibration, heldout or all).
IMPORTANT: don't change models/providers mid-run -- resumed rows and
new rows would then describe two different systems, making the combined
number meaningless.

Each row also records what the question cost: LLM calls, tokens by type, an
estimated cost, and seconds spent in each graph node (summed over the runs of a
node in the correction loop). A cell is left EMPTY, never 0, when the figure is
unknown -- the question crashed, or a model has no listed price -- so "free" and
"unknown" cannot be confused. Files written before these columns existed have
the first 14 columns unchanged and simply lack the rest.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sqlite3
import time
from collections import Counter

from eval.metrics import execution_match
from src.agents.state import initial_state
from src.config import settings
from src.graph import build_graph
from src.llm_providers import is_quota_exhausted
from src.pricing import PRICING_AS_OF

# The first 14 are the columns every committed results CSV has, in their
# original order; the rest were added with cost tracing. A fixed list, not the
# first row's keys, so resuming a file written without the new columns keeps them.
FIELDNAMES = [
    "index", "question_id", "db_id", "question", "evidence", "gold_sql", "predicted_sql",
    "correct", "difficulty", "retries", "error_classes_hit", "fatal_error",
    "schema_context_chars", "seconds",
    "llm_calls", "prompt_tokens", "cached_prompt_tokens", "completion_tokens", "reasoning_tokens",
    "est_cost_usd",
    "retriever_seconds", "planner_seconds", "generator_seconds", "executor_seconds", "classifier_seconds",
    "split",
]

# Graph node name (as src/graph.py registers it) -> its latency column.
NODE_SECONDS_COLUMNS = {
    "schema_retriever": "retriever_seconds",
    "query_planner": "planner_seconds",
    "sql_generator": "generator_seconds",
    "executor": "executor_seconds",
    "classify_error": "classifier_seconds",
}

TOKEN_COLUMNS = ("llm_calls", "prompt_tokens", "cached_prompt_tokens", "completion_tokens", "reasoning_tokens")

COST_DECIMALS = 6

# Distinct from 1 (an ordinary crash) so a launcher can tell "out of credit,
# resumable" apart from "something is broken".
EXIT_QUOTA_EXHAUSTED = 3

# Every run writes its own timestamped files by default. Reusing a path is the
# hazard: a run pointed at an existing results file would skip every question it
# already contains and emit a byte-identical file, which the Snowflake loader
# then rejects as a duplicate hash. run_benchmark() therefore refuses an
# existing results file unless resuming is requested explicitly. The env vars
# still let a scheduler choose its own paths.
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
DEFAULT_RESULTS_PATH = f"eval/results_{RUN_STAMP}.csv"
DEFAULT_METADATA_PATH = f"eval/run_metadata_{RUN_STAMP}.json"
RESULTS_PATH = os.getenv("BENCHMARK_RESULTS_PATH", DEFAULT_RESULTS_PATH)
METADATA_PATH = os.getenv("BENCHMARK_METADATA_PATH", DEFAULT_METADATA_PATH)

# Committed by scripts/make_split.py. Questions are chosen from these files,
# never as "the first N of dev.json".
SPLITS_DIR = "eval/splits"
SPLIT_NAMES = ("calibration", "heldout")
ALL_SPLIT = "all"


def load_dev_set(path: str) -> list:
    # Explicit utf-8: the dataset contains non-ASCII characters, and Windows
    # would otherwise default to cp1252 and fail on them.
    with open(f"{path}/dev.json", encoding="utf-8") as f:
        return json.load(f)


def load_split_ids(split: str) -> set | None:
    """The question_ids of a committed split, or None for "all" (every question)."""
    if split == ALL_SPLIT:
        return None
    if split not in SPLIT_NAMES:
        raise SystemExit(f"Unknown BENCHMARK_SPLIT {split!r}: use one of {', '.join((*SPLIT_NAMES, ALL_SPLIT))}.")
    with open(os.path.join(SPLITS_DIR, f"{split}_ids.json"), encoding="utf-8") as f:
        return set(json.load(f)["question_ids"])


def select_examples(examples: list, split: str, limit: int | None = None) -> list:
    """
    The questions of one split, in dev.json order. `limit` keeps the first N of
    them -- a quick plumbing check, not a measurement: the first N of a split
    still cluster in a few databases, so never report a number from one.
    """
    ids = load_split_ids(split)
    if ids is not None:
        examples = [ex for ex in examples if ex["question_id"] in ids]
    return examples[:limit] if limit else examples


def split_file_sha256(split: str) -> str | None:
    """Hash of the committed split file a run drew from; None for "all", which has no file."""
    if split == ALL_SPLIT:
        return None
    with open(os.path.join(SPLITS_DIR, f"{split}_ids.json"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_gold_result(db_id: str, gold_sql: str) -> list:
    db_path = f"{settings.benchmark_data_path}/dev_databases/{db_id}/{db_id}.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(gold_sql)
    result = cursor.fetchall()
    conn.close()
    return result


def is_correct_row(row: dict) -> bool:
    """
    Rows loaded from CSV store booleans as the strings "True"/"False", while
    freshly-computed rows hold real booleans. Normalising here avoids
    int("True") blowing up when summarising a resumed run.
    """
    return str(row.get("correct")) == "True"


def usage_columns(node_runs: list | None) -> dict:
    """
    The cost columns for one question, from the `node_runs` the graph recorded.

    None means the question has no final state to read them from (it crashed
    before finishing), so every cell is left empty. A question that made no LLM
    calls (a failed retrieval) costs exactly 0.0; one with any call to an
    unpriced model has tokens but an empty est_cost_usd.
    """
    if node_runs is None:
        return {name: "" for name in (*TOKEN_COLUMNS, "est_cost_usd", *NODE_SECONDS_COLUMNS.values())}

    columns: dict[str, int | float | str] = {
        name: sum(run.get(name, 0) for run in node_runs) for name in TOKEN_COLUMNS
    }
    costs = [run.get("cost_usd") for run in node_runs]
    columns["est_cost_usd"] = "" if any(c is None for c in costs) else round(sum(costs), COST_DECIMALS)
    for node, column in NODE_SECONDS_COLUMNS.items():
        columns[column] = round(sum(run.get("seconds", 0) for run in node_runs if run.get("node") == node), 3)
    return columns


def write_results(results: list) -> None:
    if not results:
        return
    # encoding + errors="replace": dataset text includes non-ASCII characters
    # that Windows' default cp1252 codec cannot encode, which would otherwise
    # crash the write AFTER a full expensive run.
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8", errors="replace") as f:
        # restval: rows resumed from a file without the cost columns lack those keys.
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, restval="")
        writer.writeheader()
        writer.writerows(results)


def run_config(split: str) -> dict:
    """The settings that decide what a row means; a resumed run must still match them."""
    return {
        "planner_provider": settings.planner_provider,
        "planner_model": settings.planner_model,
        "generator_provider": settings.generator_provider,
        "generator_model": settings.generator_model,
        "classifier_provider": settings.classifier_provider,
        "classifier_model": settings.classifier_model,
        "include_evidence_in_prompts": settings.include_evidence_in_prompts,
        "split": split,
    }


def check_resume_matches_metadata(split: str) -> bool:
    """
    Refuse to resume under a different config than the file was started with.
    Returns whether a metadata file exists; when it does, it is left untouched,
    since rewriting it would relabel the earlier rows with the new models.
    """
    if not os.path.exists(METADATA_PATH):
        return False
    with open(METADATA_PATH, encoding="utf-8") as f:
        recorded = json.load(f)
    changed = {k: (recorded.get(k), v) for k, v in run_config(split).items() if recorded.get(k) != v}
    if changed:
        raise SystemExit(
            f"Cannot resume {RESULTS_PATH}: the run was started with a different config than now "
            f"(recorded vs current): {changed}. Restore .env and the split, or start a new run."
        )
    return True


def write_run_metadata(total_questions: int, split: str) -> None:
    """
    Record which configuration produced this run, alongside results.csv.

    Written at the START of a run, not the end, so an interrupted run still
    leaves a record of what it was. results.csv on its own says what happened
    but not under which models -- and the answer stops being recoverable the
    moment .env changes, which is exactly when it matters. The two historical
    CSVs in this directory predate this file and are why it exists: their
    model configuration is no longer knowable and loads to
    "unknown (pre-tracking)".

    Consumed by eval/load_to_snowflake.py, which reads it to populate
    EVAL_RUNS without anyone having to remember or retype the config.

    The timestamp is timezone-aware (via astimezone()) rather than a naive
    local time, so a run is still unambiguously orderable against runs from a
    machine in another zone.
    """
    metadata = {
        "run_timestamp": datetime.datetime.now().astimezone().isoformat(),
        **run_config(split),
        "total_questions": total_questions,
        # Two runs are only comparable on the same split.
        "split_file_sha256": split_file_sha256(split),
        # The prices est_cost_usd was computed with; see src/pricing.py.
        "pricing_as_of": PRICING_AS_OF,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Run metadata written to {METADATA_PATH}")


def print_summary(results: list, total_elapsed: float, questions_run_now: int) -> None:
    if not results:
        return

    correct = sum(1 for r in results if is_correct_row(r))
    print(f"\nOverall execution accuracy: {correct / len(results):.2%} ({correct}/{len(results)})")

    for key, label in (("difficulty", "By difficulty"), ("db_id", "By database")):
        groups: dict = {}
        for r in results:
            g = groups.setdefault(r[key], [0, 0])
            g[1] += 1
            g[0] += int(is_correct_row(r))
        print(f"\n{label}:")
        for name, (c, t) in sorted(groups.items()):
            print(f"  {name:>26}: {c / t:.2%} ({c}/{t})")

    # Only rows with a known cost: blank cells (crashes, unpriced models, rows
    # resumed from a file without the columns) would otherwise read as free.
    costed = [r for r in results if r.get("est_cost_usd") not in (None, "")]
    if costed:
        total_cost = sum(float(r["est_cost_usd"]) for r in costed)
        print(
            f"\nEstimated cost: ${total_cost:.4f} over {len(costed)} of {len(results)} questions "
            f"with a known cost (${total_cost / len(costed):.4f}/question)"
        )

    if questions_run_now:
        avg = total_elapsed / questions_run_now
        print(f"\nThis session: {questions_run_now} questions in {total_elapsed / 60:.1f} min ({avg:.1f}s/question)")


def read_existing_results() -> list:
    with open(RESULTS_PATH, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def run_benchmark(limit: int | None = None, split: str = ALL_SPLIT, resume: bool = False) -> None:
    examples = select_examples(load_dev_set(settings.benchmark_data_path), split, limit)
    if os.path.exists(RESULTS_PATH) and not resume:
        raise SystemExit(
            f"{RESULTS_PATH} already exists. A run never reuses a results file silently: pass --resume "
            f"(or BENCHMARK_RESUME=1) to continue it, or point BENCHMARK_RESULTS_PATH at a new file."
        )
    graph = build_graph()
    if limit:
        print(f"Running a LIMITED slice: the first {len(examples)} questions of split '{split}'.\n")
    else:
        print(f"Running split '{split}': {len(examples)} examples.\n")

    # Resume support (only when asked for): skip the questions the results file
    # already holds and append the rest, so a long run survives interruption
    # across sessions. Matched by question_id, not position, so the file can
    # only ever be continued against the same set of questions. A counter, not
    # a set: BIRD repeats a question_id on two rows, and each row is done once.
    results: list = []
    done: Counter = Counter()
    if resume and not os.path.exists(RESULTS_PATH):
        raise SystemExit(
            f"Cannot resume: {RESULTS_PATH} does not exist. Point BENCHMARK_RESULTS_PATH at the file to continue."
        )
    resuming = resume
    if resuming:
        has_metadata = check_resume_matches_metadata(split)
        results = read_existing_results()
        done = Counter(str(r["question_id"]) for r in results)
        unknown = set(done) - {str(ex.get("question_id")) for ex in examples}
        if unknown:
            raise SystemExit(
                f"Cannot resume {RESULTS_PATH}: it holds question_ids outside split '{split}' "
                f"(e.g. {sorted(unknown)[:3]}). Resume with the same split and limit it was started with."
            )
        other_splits = {r["split"] for r in results if r.get("split")} - {split}
        if other_splits:
            raise SystemExit(f"Cannot resume {RESULTS_PATH}: it holds rows from split {sorted(other_splits)}, not '{split}'.")
        if results:
            print(f"Resuming: {len(results)} questions already recorded, {len(examples) - len(results)} to go.\n")

    # Written before any work happens, so the config record survives even if
    # the run dies. A resume keeps the original record: it already matches.
    if not (resuming and has_metadata):
        write_run_metadata(len(examples), split)

    correct = sum(1 for r in results if is_correct_row(r))
    start_time = time.monotonic()
    questions_run_now = 0

    for i, ex in enumerate(examples):
        question_id = ex.get("question_id", i)
        if done[str(question_id)] > 0:
            done[str(question_id)] -= 1
            continue

        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["SQL"]
        evidence = ex.get("evidence")
        difficulty = ex.get("difficulty", "unknown")

        question_start = time.monotonic()
        state = initial_state(db_id, question, evidence=evidence, max_retries=settings.max_retries)

        fatal_error = None
        try:
            final_state = graph.invoke(state)
            gold_result = get_gold_result(db_id, gold_sql)
            row_correct = execution_match(final_state.get("execution_result"), gold_result)
            # A failed schema retrieval no longer raises -- the graph ends the
            # question itself (see schema_retriever.py) -- so it would
            # otherwise land in the CSV as an unexplained failure. Record its
            # cause in the same column a crash used, so it stays findable.
            # Scoring is untouched: row_correct is computed exactly as before,
            # and is False here because execution_result is None.
            if final_state.get("retrieval_failed"):
                fatal_error = final_state.get("execution_error")
        except Exception as e:
            if is_quota_exhausted(e):
                # Stop the RUN, not just this question. With no credit, every
                # remaining question fails the same way, and recording each as
                # a wrong answer would turn the rest of the CSV into a
                # measurement of an empty account. Checkpoint what genuinely
                # completed and exit non-zero -- this question is NOT recorded,
                # so resuming after a top-up re-runs it from scratch.
                write_results(results)
                print(
                    f"\nSTOPPED at question #{i + 1} (question_id {question_id}): provider account "
                    f"has no credit.\n  {e}\n{len(results)} completed questions checkpointed to "
                    f"{RESULTS_PATH}. Add credit, then re-run with --resume to continue from #{i + 1}."
                )
                raise SystemExit(EXIT_QUOTA_EXHAUSTED) from e
            fatal_error = f"{type(e).__name__}: {e}"
            final_state = {"sql_query": None, "trace": [{"node": "benchmark", "fatal_error": fatal_error}], "retry_count": 0}
            row_correct = False
        question_elapsed = time.monotonic() - question_start

        correct += int(row_correct)
        questions_run_now += 1

        error_classes_hit = [t.get("error_class") for t in final_state.get("trace", []) if "error_class" in t]
        schema_context = final_state.get("schema_context") or ""

        results.append({
            "index": i,
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "gold_sql": gold_sql,
            "predicted_sql": final_state.get("sql_query"),
            "correct": row_correct,
            "difficulty": difficulty,
            "retries": final_state.get("retry_count", 0),
            "error_classes_hit": ";".join(error_classes_hit),
            "fatal_error": fatal_error or "",
            "schema_context_chars": len(schema_context),
            "seconds": round(question_elapsed, 1),
            "split": split,
            **usage_columns(final_state.get("node_runs")),
        })

        # Checkpoint every 10 questions so an interruption costs at most the
        # last few questions rather than the entire run.
        if (i + 1) % 10 == 0 or limit:
            write_results(results)
            elapsed_now = time.monotonic() - start_time
            rate = elapsed_now / questions_run_now
            remaining = rate * (len(examples) - i - 1)
            print(
                f"  ...{i + 1}/{len(examples)} | {question_elapsed:.1f}s this one | "
                f"accuracy {correct}/{len(results)} | ~{remaining / 60:.0f} min left"
            )

    write_results(results)
    print_summary(results, time.monotonic() - start_time, questions_run_now)
    print(f"\nPer-example results written to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agent graph on a split of BIRD Mini-Dev.")
    parser.add_argument(
        "--resume", action="store_true", default=os.getenv("BENCHMARK_RESUME") == "1",
        help="continue the results file at BENCHMARK_RESULTS_PATH instead of refusing it",
    )
    limit_env = os.getenv("BENCHMARK_LIMIT")
    run_benchmark(
        limit=int(limit_env) if limit_env else None,
        split=os.getenv("BENCHMARK_SPLIT", ALL_SPLIT),
        resume=parser.parse_args().resume,
    )