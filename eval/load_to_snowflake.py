"""
Load one benchmark run's results CSV into Snowflake.

Standalone on purpose. This is the script Airflow will invoke unattended, so
it has to be a complete unit of work on its own: parse arguments, resolve
configuration, open its own connection, load, exit with a meaningful status.
Bolting it onto run_benchmark.py would tie loading to running, and the two
have genuinely different lifecycles -- a run happens once and takes hours, a
load can be re-run, backfilled, or pointed at a CSV produced months earlier.

Usage:
    python eval/load_to_snowflake.py --csv eval/results.csv --config-label baseline

Configuration resolution, in precedence order:
  1. An explicit --flag on the command line always wins.
  2. Otherwise the field is read from --metadata (default
     the file next to the CSV), the sidecar run_benchmark.py now writes.
  3. Otherwise model fields become "unknown (pre-tracking)".

There is deliberately NO fallback for run_timestamp. If no metadata file
exists and --run-timestamp was not passed, this exits with an error rather
than using the current time. Defaulting there would stamp a months-old
benchmark as having run today -- silently corrupting the one column the table
exists to keep honest, in a way nobody would ever catch by reading the data.

Note on run_timestamp storage: EVAL_RUNS.run_timestamp is TIMESTAMP_NTZ,
which has no timezone. Offset-aware inputs are therefore converted to UTC
before insert, so runs recorded in different offsets stay comparable. The
original offset label is not preserved; a TIMESTAMP_TZ column would be needed
for that.
"""

import argparse
import datetime
import hashlib
import json
import sys
import uuid
from pathlib import Path

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

# Run as a file, not a module: `python eval/load_to_snowflake.py` puts eval/ on
# sys.path rather than the project root, so `src` would not resolve. Airflow
# invokes scripts by absolute path too, so the bootstrap has to live here
# rather than depending on the caller's cwd or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.db.snowflake_connection import get_snowflake_connection  # noqa: E402

UNKNOWN = "unknown (pre-tracking)"

# CSV columns that exist in results.csv but not in EVAL_RESULTS. "index" is
# the runner's loop counter -- positional, meaningless once rows are in a
# table with its own identity column. "split" describes the whole run, so it
# belongs to EVAL_RUNS, not to each result row.
DROP_COLUMNS = ["index", "split"]

MODEL_FIELDS = [
    "planner_provider",
    "planner_model",
    "generator_provider",
    "generator_model",
    "classifier_provider",
    "classifier_model",
]

# Cost-tracing columns, present only in CSVs written after run_benchmark.py
# started recording them. Token counts are whole numbers; the rest are decimals.
INTEGER_COLUMNS = [
    "llm_calls",
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
]
DECIMAL_COLUMNS = [
    "est_cost_usd",
    "retriever_seconds",
    "planner_seconds",
    "generator_seconds",
    "executor_seconds",
    "classifier_seconds",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a benchmark results CSV into Snowflake (EVAL_RUNS + EVAL_RESULTS).",
    )
    parser.add_argument("--csv", required=True, help="Path to the results CSV to load.")
    parser.add_argument("--config-label", required=True, help="Short name for this run, e.g. 'baseline'.")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Run metadata sidecar written by run_benchmark.py (default: the one next to --csv, "
        "results_<stamp>.csv -> run_metadata_<stamp>.json).",
    )

    # Manual overrides. All optional; each one beats the metadata file.
    parser.add_argument("--run-timestamp", help="ISO 8601 timestamp. REQUIRED when no metadata file exists.")
    for field in MODEL_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", help=f"Override {field}.")
    parser.add_argument(
        "--include-evidence",
        choices=["true", "false"],
        help="Whether evidence was fed into agent prompts for this run.",
    )

    args = parser.parse_args(argv)
    if args.metadata is None:
        # Each run writes its own pair of files, so a fixed default would attach
        # another run's split and models to this CSV.
        csv_path = Path(args.csv)
        args.metadata = str(csv_path.with_name(csv_path.name.replace("results", "run_metadata", 1)).with_suffix(".json"))
    return args


def load_metadata(path: str) -> dict:
    """Return the sidecar's contents, or {} if it is not there."""
    meta_path = Path(path)
    if not meta_path.is_file():
        return {}
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_config(args: argparse.Namespace, metadata: dict) -> dict:
    """
    Merge CLI overrides with the metadata sidecar into the EVAL_RUNS row.

    Raises SystemExit with a clear message if run_timestamp cannot be
    determined -- see the module docstring for why that is fatal rather than
    defaulted.
    """
    run_timestamp = args.run_timestamp or metadata.get("run_timestamp")
    if not run_timestamp:
        raise SystemExit(
            f"error: no metadata file at '{args.metadata}' and --run-timestamp was not given.\n"
            "Refusing to default to the current time: that would record this data as having\n"
            "been produced today, which is almost certainly false for a CSV old enough to\n"
            "predate the metadata sidecar. Pass the real run time explicitly, e.g.\n"
            '  --run-timestamp "2026-08-10T19:25:20-04:00"'
        )

    config = {field: getattr(args, field) or metadata.get(field) or UNKNOWN for field in MODEL_FIELDS}
    config["run_timestamp"] = run_timestamp
    # None for a run recorded before cost tracing; the EVAL_RUNS column stays NULL.
    config["pricing_as_of"] = metadata.get("pricing_as_of")
    # Which questions the run covered (calibration, heldout, all). None for a run
    # recorded before splits existed; the EVAL_RUNS column stays NULL.
    config["split"] = metadata.get("split")

    if args.include_evidence is not None:
        config["include_evidence_in_prompts"] = args.include_evidence == "true"
    else:
        # Left as None (-> SQL NULL) when genuinely unknown. A default of
        # False would be a guess indistinguishable from a recorded fact.
        config["include_evidence_in_prompts"] = metadata.get("include_evidence_in_prompts")

    return config


def to_naive_utc(timestamp: str) -> datetime.datetime:
    """
    Parse an ISO 8601 string into a naive UTC datetime for TIMESTAMP_NTZ.

    Offset-aware input is converted to UTC and then stripped of its tzinfo,
    because binding an aware datetime to an NTZ column would otherwise just
    drop the offset and keep the local wall clock -- making two runs an hour
    apart in different zones look simultaneous.
    """
    parsed = datetime.datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def prepare_results(csv_path: str, run_id: str) -> pd.DataFrame:
    """
    Read the results CSV into the shape EVAL_RESULTS expects.

    Three normalisations, each covering a real mismatch between what the
    runner writes and what the table holds:
      - "True"/"False" text becomes real booleans. The runner writes Python
        bools that round-trip through CSV as those strings; loading them as
        text into a BOOLEAN column fails, and loading them into a VARCHAR
        column would make `WHERE correct` silently wrong.
      - Empty-string fatal_error becomes NULL. "" and "no error occurred" are
        the same fact, and only one of them is countable with COUNT().
      - The runner's positional "index" column is dropped; it has no
        counterpart in the table.

    The cost-tracing columns are optional: a CSV from before they existed has
    none of them, and one from after has them with empty cells wherever a figure
    is unknown. Empty becomes NULL, never 0. Token counts are cast to a nullable
    integer so a blank does not turn the whole column into floats ("5.0" into a
    NUMBER column); unparseable text becomes NULL rather than failing the load.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    for column in INTEGER_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in DECIMAL_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "correct" in df.columns:
        df["correct"] = df["correct"].map(
            lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true"
        )

    for column in ("fatal_error", "error_classes_hit", "evidence"):
        if column in df.columns:
            df[column] = df[column].replace("", None)

    df.insert(0, "run_id", run_id)

    # write_pandas quotes identifiers, and the DDL created these columns
    # unquoted -- so Snowflake folded them to upper case. Match that, or every
    # column comes back "invalid identifier".
    df.columns = [c.upper() for c in df.columns]
    return df


def sha256_file(path: str) -> str:
    """
    SHA-256 of the CSV's bytes, stored on the run and used to detect reloads.

    Read in chunks rather than all at once: a results CSV is a few hundred KB
    today, but nothing about this script should assume that stays true.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_existing_run(conn, file_hash: str):
    """
    Return (run_id, config_label, run_timestamp) for a prior load of these
    exact bytes, or None.

    Content-addressed rather than filename-based on purpose: the same run gets
    copied and renamed (eval/results.csv is a byte-identical copy of
    eval/results_150_with_evidence.csv), so a filename check would miss the
    duplicate that matters most.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_id, config_label, run_timestamp FROM EVAL_RUNS WHERE source_file_hash = %s",
            (file_hash,),
        )
        return cur.fetchone()


def find_missing_columns(conn, needed: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Which of the columns we are about to write are absent from their tables.

    Checked BEFORE anything is inserted. insert_run commits-by-default ahead of
    the results load, so a CSV with a column the table lacks used to fail inside
    write_pandas with the EVAL_RUNS row already written -- and that row's hash
    then made every retry look like a duplicate load. Catching it here leaves
    nothing behind to clean up.

    `needed` maps table name to column names; comparison is case-insensitive
    because the DDL's unquoted identifiers fold to upper case.
    """
    tables = [t.upper() for t in needed]
    placeholders = ", ".join(["%s"] * len(tables))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE table_schema = %s AND table_name IN ({placeholders})",
            (settings.snowflake_schema.upper(), *tables),
        )
        known = {(table.upper(), column.upper()) for table, column in cur.fetchall()}

    missing = {}
    for table, columns in needed.items():
        absent = [c for c in columns if (table.upper(), c.upper()) not in known]
        if absent:
            missing[table.upper()] = absent
    return missing


def insert_run(conn, run_id: str, config_label: str, config: dict, total_questions: int, file_hash: str) -> None:
    """
    Write the parent EVAL_RUNS row. Must happen before the results load:
    EVAL_RESULTS.run_id carries a foreign key to this table.

    pricing_as_of and split are written only when the run has one, so a run from
    before cost tracing or splits still loads into an EVAL_RUNS table that lacks
    the column.
    """
    columns = [
        "run_id", "run_timestamp", "config_label", "include_evidence_in_prompts",
        "planner_provider", "planner_model",
        "generator_provider", "generator_model",
        "classifier_provider", "classifier_model",
        "total_questions", "source_file_hash",
    ]
    values = [
        run_id,
        to_naive_utc(config["run_timestamp"]),
        config_label,
        config["include_evidence_in_prompts"],
        config["planner_provider"],
        config["planner_model"],
        config["generator_provider"],
        config["generator_model"],
        config["classifier_provider"],
        config["classifier_model"],
        total_questions,
        file_hash,
    ]
    if config.get("pricing_as_of"):
        columns.append("pricing_as_of")
        values.append(config["pricing_as_of"])
    if config.get("split"):
        columns.append("split")
        values.append(config["split"])

    placeholders = ", ".join(["%s"] * len(columns))
    conn.cursor().execute(
        f"INSERT INTO EVAL_RUNS ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(values),
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    if not Path(args.csv).is_file():
        raise SystemExit(f"error: CSV not found: {args.csv}")

    metadata = load_metadata(args.metadata)
    if metadata:
        print(f"Using metadata from {args.metadata}")
    else:
        print(f"No metadata file at {args.metadata}; using CLI arguments and '{UNKNOWN}' fallbacks.")

    config = resolve_config(args, metadata)
    run_id = str(uuid.uuid4())
    file_hash = sha256_file(args.csv)
    df = prepare_results(args.csv, run_id)

    print(f"run_id:          {run_id}")
    print(f"config_label:    {args.config_label}")
    print(f"run_timestamp:   {config['run_timestamp']} -> {to_naive_utc(config['run_timestamp'])} UTC (NTZ)")
    print(f"include_evidence:{config['include_evidence_in_prompts']}")
    print(f"pricing_as_of:   {config['pricing_as_of']}")
    print(f"split:           {config['split']}")
    for field in MODEL_FIELDS:
        print(f"{field:21}{config[field]}")
    print(f"source_file_hash {file_hash}")
    print(f"rows to load:    {len(df)}")

    conn = get_snowflake_connection()
    try:
        # Checked before anything is written. Re-loading the same bytes would
        # silently double every aggregate computed over EVAL_RESULTS, and the
        # duplicate is indistinguishable from a real second run once it lands.
        existing = find_existing_run(conn, file_hash)
        if existing:
            existing_run_id, existing_label, existing_timestamp = existing
            raise SystemExit(
                f"error: refusing to load '{args.csv}' -- these exact bytes are already in EVAL_RUNS.\n"
                f"  sha256:        {file_hash}\n"
                f"  existing run:  {existing_run_id}\n"
                f"  config_label:  {existing_label}\n"
                f"  run_timestamp: {existing_timestamp}\n"
                "Loading it again would duplicate every row and skew any accuracy computed\n"
                "over EVAL_RESULTS. If this really is a distinct run, its CSV should differ;\n"
                "if you meant to replace the existing run, delete it first."
            )

        missing = find_missing_columns(
            conn,
            {
                "EVAL_RESULTS": list(df.columns),
                "EVAL_RUNS": [c for c in ("pricing_as_of", "split") if config.get(c)],
            },
        )
        if missing:
            details = "\n".join(f"  {table}: {', '.join(columns)}" for table, columns in missing.items())
            raise SystemExit(
                f"error: refusing to load '{args.csv}' -- the Snowflake tables lack columns it needs:\n"
                f"{details}\n"
                "Apply the migrations in snowflake_sql/schema.sql first. Nothing was written."
            )

        insert_run(conn, run_id, args.config_label, config, len(df), file_hash)

        success, nchunks, nrows, _ = write_pandas(
            conn,
            df,
            table_name="EVAL_RESULTS",
            database=settings.snowflake_database,
            schema=settings.snowflake_schema,
        )
        if not success:
            conn.rollback()
            raise SystemExit("error: write_pandas reported failure; EVAL_RUNS insert rolled back.")

        conn.commit()
        print(f"\nLoaded {nrows} rows into EVAL_RESULTS in {nchunks} chunk(s) under run_id {run_id}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
