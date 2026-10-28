"""
Weekly NL2SQL regression pipeline: benchmark -> load -> drift check.

WEEKLY, NOT DAILY. Every run makes real LLM API calls for all 150 calibration questions
(planner + generator + a classifier call per retry), so the schedule is a cost
decision, not a freshness one. catchup=False for the same reason: on first
unpause Airflow would otherwise backfill one run per missed interval since
start_date, each one a full paid benchmark.

The three tasks are strictly sequential because each consumes the previous
one's output: the loader needs the CSV, and the drift check needs the run row
the loader created in Snowflake.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

# Matches the mounts in airflow/docker-compose.yaml. Overridable so the DAG can
# be exercised outside the container.
PROJECT_ROOT = Path(os.getenv("NL2SQL_PROJECT_ROOT", "/opt/airflow/project"))

# The regression set: the committed calibration split (eval/splits/), 150
# questions stratified over all 11 databases. Not "the first 150 of dev.json",
# which covered only 4 databases and could hide a regression in the other 7.
BENCHMARK_SPLIT = os.getenv("NL2SQL_BENCHMARK_SPLIT", "calibration")

CONFIG_LABEL = "scheduled_regression"

# Which run every scheduled run is measured against. Set as an Airflow Variable
# (preferred -- changeable from the UI without a restart) or as an env var.
# There is deliberately no default: silently comparing against an arbitrary run
# would produce authoritative-looking numbers with no defined meaning.
BASELINE_RUN_ID_KEY = "nl2sql_baseline_run_id"

ALPHA = 0.05

# Where docker-compose.yaml mounts the key, and where the host's ~/.dbt lands.
KEY_PATH_IN_CONTAINER = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "/opt/secrets/rsa_key.p8")
HOST_DBT_PROFILES = Path(os.getenv("DBT_PROFILES_SRC", "/opt/dbt-profiles/profiles.yml"))

# Only what drift_check reads. "+model" selects that model and its ancestors,
# so this builds stg_eval_runs, stg_eval_results, int_eval_results_enriched and
# mart_accuracy_over_time -- and deliberately NOT mart_accuracy_by_database,
# mart_accuracy_by_difficulty or stg_query_history, which nothing downstream
# here consumes and which would cost extra warehouse time every week.
DBT_SELECTOR = "+mart_accuracy_over_time"


def _run(argv: list[str], env_extra: dict | None = None) -> None:
    """
    Run a project script from the project root.

    cwd matters: run_benchmark.py resolves BENCHMARK_DATA_PATH and its output
    paths relative to the working directory, and the container's default cwd is
    /opt/airflow, not the project. check=True turns a non-zero exit into a
    failed Airflow task rather than a silently green one.
    """
    env = {**os.environ, **(env_extra or {})}
    print(f"$ {' '.join(argv)}  (cwd={PROJECT_ROOT})")
    subprocess.run(argv, cwd=PROJECT_ROOT, env=env, check=True)


def _materialise_dbt_profiles() -> str:
    """
    Copy the host's profiles.yml to a writable temp dir, fixing the key path.

    THE PROBLEM: ~/.dbt/profiles.yml sets

        private_key_path: C:/Users/Arun/.snowflake/rsa_key.p8

    which is correct on the Windows host and meaningless inside a Linux
    container. Worse, it would not even error as a missing absolute path --
    a colon is a legal character in a Linux filename, so "C:/Users/..." is a
    *relative* path and dbt would look for it under the current directory and
    report a confusing miss.

    WHY NOT JUST EDIT profiles.yml: it is the user's file, lives outside the
    repo alongside the private key, and holds the credentials for host-side
    dbt runs. Rewriting it to suit containers would break `dbt build` on the
    host, and mounting it read-write so the container could edit it defeats
    the point of mounting it :ro.

    WHY NOT A SEPARATE CONTAINER-ONLY profiles.yml IN THE REPO: it would
    duplicate account/user/role/warehouse/schema, and the copies would drift.
    The host file stays the single source of truth.

    SO: read the read-only mount, rewrite exactly one line, write the result
    to a fresh temp dir, and point dbt at that with --profiles-dir. Everything
    else -- account, user, role (NL2SQL_DBT_ROLE), schema, warehouse -- comes
    through unchanged from the host file.
    """
    if not HOST_DBT_PROFILES.is_file():
        raise RuntimeError(
            f"No dbt profiles.yml at {HOST_DBT_PROFILES}. docker-compose.yaml mounts the "
            "host's ~/.dbt at /opt/dbt-profiles read-only; check DBT_PROFILES_HOST_DIR "
            "and that the file exists on the host."
        )

    original = HOST_DBT_PROFILES.read_text(encoding="utf-8")
    rewritten, n = re.subn(
        r"(?m)^(\s*private_key_path:\s*).*$",
        lambda m: f"{m.group(1)}{KEY_PATH_IN_CONTAINER}",
        original,
    )
    if n == 0:
        # Not fatal: the profile may authenticate some other way. Say so rather
        # than silently proceeding to an opaque auth failure.
        print(
            "WARNING: no private_key_path line found in profiles.yml; leaving it as-is. "
            "If dbt then fails to authenticate, this is why."
        )
    else:
        print(f"Rewrote {n} private_key_path -> {KEY_PATH_IN_CONTAINER}")

    out_dir = Path(tempfile.mkdtemp(prefix="dbt-profiles-"))
    (out_dir / "profiles.yml").write_text(rewritten, encoding="utf-8")
    return str(out_dir)


def _resolve_baseline_run_id() -> str:
    """Airflow Variable first, env var as fallback, hard error if neither."""
    try:
        from airflow.sdk import Variable

        value = Variable.get(BASELINE_RUN_ID_KEY, default=None)
    except Exception:  # noqa: BLE001 -- SDK layout differs across Airflow versions
        value = None
    value = value or os.getenv("NL2SQL_BASELINE_RUN_ID")
    if not value:
        raise RuntimeError(
            "No baseline run configured. Set the Airflow Variable "
            f"'{BASELINE_RUN_ID_KEY}' (Admin -> Variables) or the env var "
            "NL2SQL_BASELINE_RUN_ID to the run_id this pipeline should be "
            "measured against. Query EVAL_RUNS to pick one."
        )
    return value


@dag(
    dag_id="nl2sql_eval",
    description="Weekly calibration-split regression: benchmark, load to Snowflake, drift check.",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,  # two concurrent runs would contend on the Chroma SQLite store
    default_args={"retries": 0},  # a retry re-pays for 150 LLM calls; fail loudly instead
    tags=["nl2sql", "benchmark", "snowflake"],
)
def nl2sql_eval():

    @task
    def run_benchmark(ts_nodash: str | None = None) -> dict:
        """
        Run the calibration split into a timestamped CSV.

        BENCHMARK_SPLIT is read in run_benchmark.py's __main__ block and picks
        the committed question-ID file.

        The timestamped output path matters for two reasons. The obvious one is
        that each run keeps its own artifact. The other is that a results file
        is a run's identity: run_benchmark.py refuses an existing one unless
        --resume is given (it is not), so a reused path fails loudly instead of
        skipping every question and emitting byte-identical output that the
        loader would reject as a duplicate hash.

        Invoked as `python -m eval.run_benchmark`, not by file path, because
        the module does `from eval.metrics import ...` -- running it by path
        puts eval/ on sys.path instead of the project root and that import
        fails.

        ts_nodash is a RESERVED TaskFlow context key: Airflow injects the run's
        timestamp automatically, and declaring any default other than None is a
        DAG-parse error ("Context key parameter ts_nodash can't have a default
        other than None"). So it is annotated Optional and filled by Airflow --
        do not give it a "{{ ts_nodash }}" default.
        """
        if not ts_nodash:
            raise RuntimeError("ts_nodash was not injected; this task must run inside a DAG run.")

        results_csv = PROJECT_ROOT / "eval" / f"results_{CONFIG_LABEL}_{ts_nodash}.csv"
        metadata = PROJECT_ROOT / "eval" / f"run_metadata_{CONFIG_LABEL}_{ts_nodash}.json"

        _run(
            [sys.executable, "-m", "eval.run_benchmark"],
            {
                "BENCHMARK_SPLIT": BENCHMARK_SPLIT,
                "BENCHMARK_RESULTS_PATH": str(results_csv),
                "BENCHMARK_METADATA_PATH": str(metadata),
            },
        )

        if not results_csv.is_file():
            raise RuntimeError(f"Benchmark reported success but produced no CSV at {results_csv}")

        return {"results_csv": str(results_csv), "metadata": str(metadata)}

    @task
    def load_to_snowflake(paths: dict) -> str:
        """
        Load the run into EVAL_RUNS + EVAL_RESULTS, and return its file hash.

        NO --on-duplicate skip flag, deliberately. The loader refuses a CSV
        whose SHA-256 already exists and exits 1. For a scheduled run that
        rejection is the most valuable signal the pipeline produces: identical
        bytes mean the benchmark did not actually re-run (a resumed file, a
        stale mount, a no-op). Skipping past it would turn a broken pipeline
        into a green one.

        The returned hash is how the drift check finds this run's row --
        content-addressed, rather than parsing a run_id out of stdout.
        """
        _run(
            [
                sys.executable,
                "eval/load_to_snowflake.py",
                "--csv", paths["results_csv"],
                "--config-label", CONFIG_LABEL,
                "--metadata", paths["metadata"],
            ]
        )

        sys.path.insert(0, str(PROJECT_ROOT))
        from eval.load_to_snowflake import sha256_file

        return sha256_file(paths["results_csv"])

    @task
    def dbt_build(file_hash: str) -> str:
        """
        Rebuild only the models drift_check reads, then pass the hash through.

        NECESSARY, not cosmetic: mart_accuracy_over_time is a dbt TABLE, not a
        view. A run freshly loaded into EVAL_RUNS/EVAL_RESULTS does not appear
        in it until the model is rebuilt, so without this step drift_check
        would always fail with "Missing from MART_ACCURACY_OVER_TIME".

        Scoped with "+mart_accuracy_over_time" rather than a bare `dbt build`:
        that selects the target model plus its ancestors and stops there,
        leaving the two accuracy-breakdown marts and stg_query_history
        untouched. They are not read here, and rebuilding them would burn
        warehouse time every week for nothing. Ancestor tests (including the
        relationships test on run_id) still run, which is the point of using
        `build` over `run`.

        Takes and returns file_hash purely to make the dependency explicit in
        the DAG graph -- dbt must finish before drift_check reads the mart.
        """
        profiles_dir = _materialise_dbt_profiles()
        _run(
            [
                "dbt", "build",
                "--select", DBT_SELECTOR,
                "--project-dir", str(PROJECT_ROOT / "dbt_project"),
                "--profiles-dir", profiles_dir,
            ]
        )
        return file_hash

    @task
    def drift_check(file_hash: str) -> dict:
        """
        Compare this run against the baseline, and fail on real degradation.

        Two gates, in order, and the order is the point.

        1. COVERAGE. The runs must be on the same split, and
           mart_accuracy_over_time.db_id_set_hash fingerprints the
           exact set of databases a run covered. If the two runs cover
           different sets their accuracies are not comparable, so this fails
           with that explanation rather than reporting a delta that means
           nothing. distinct_db_count alone is not enough: two disjoint
           4-database runs both report 4. A baseline recorded before splits
           existed has no split, so it is refused too: point the baseline at a
           calibration-split run.

        2. PAIRED SIGNIFICANCE, only if coverage matches. Reuses
           scripts.compare_ablation.mcnemar_exact_two_sided -- the same exact
           binomial McNemar behind the project's headline +8.67pp / p=0.0072
           result -- rather than a second implementation that could drift from
           it. Questions pair on question_id, not row position, so a reordered
           or partial run cannot compare two different questions. BIRD gives
           two questions the same question_id (137, 138), so rows are also
           numbered within their id; a plain join on question_id would pair
           each of those rows with both rows of the other run.

        The failure condition is directional: significant AND more broken than
        fixed. A significant improvement is not a failure. The two-sided
        p-value is reported either way.
        """
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.compare_ablation import mcnemar_exact_two_sided
        from src.db.snowflake_connection import get_snowflake_connection

        baseline_run_id = _resolve_baseline_run_id()

        conn = get_snowflake_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT run_id FROM EVAL_RUNS WHERE source_file_hash = %s",
                    (file_hash,),
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError(
                        f"No EVAL_RUNS row with source_file_hash={file_hash}. The load "
                        "task reported success, so this means the hash computed here "
                        "does not match the one stored."
                    )
                current_run_id = row[0]

                # --- gate 1: coverage ------------------------------------
                cur.execute(
                    """
                    SELECT run_id, config_label, accuracy, distinct_db_count, db_id_set_hash, split
                    FROM MARTS.MART_ACCURACY_OVER_TIME
                    WHERE run_id IN (%s, %s)
                    """,
                    (current_run_id, baseline_run_id),
                )
                rows = {r[0]: r for r in cur.fetchall()}

                missing = [r for r in (current_run_id, baseline_run_id) if r not in rows]
                if missing:
                    raise RuntimeError(
                        f"Missing from MART_ACCURACY_OVER_TIME: {missing}. That mart is a "
                        "dbt TABLE, not a view, so it only reflects runs present at the "
                        "last `dbt build`. Rebuild it after loading."
                    )

                cur_row, base_row = rows[current_run_id], rows[baseline_run_id]
                if cur_row[5] is None or cur_row[5] != base_row[5]:
                    raise RuntimeError(
                        "NOT COMPARABLE: the two runs are not on the same split, so they "
                        "answered different questions.\n"
                        f"  current  {current_run_id} ({cur_row[1]}): split {cur_row[5]}\n"
                        f"  baseline {baseline_run_id} ({base_row[1]}): split {base_row[5]}\n"
                        "A split of None means the run was recorded before splits existed. "
                        "Point the baseline at a run on the same split."
                    )
                if cur_row[4] != base_row[4]:
                    raise RuntimeError(
                        "NOT COMPARABLE: the two runs cover different database sets, so "
                        "any accuracy delta between them is meaningless.\n"
                        f"  current  {current_run_id} ({cur_row[1]}): "
                        f"{cur_row[3]} dbs, set hash {cur_row[4]}\n"
                        f"  baseline {baseline_run_id} ({base_row[1]}): "
                        f"{base_row[3]} dbs, set hash {base_row[4]}\n"
                        "Re-point the baseline at a run with matching coverage, or re-run "
                        "the benchmark over the baseline's database set."
                    )

                # --- gate 2: paired significance -------------------------
                cur.execute(
                    """
                    WITH numbered AS (
                        SELECT run_id, question_id, correct,
                               ROW_NUMBER() OVER (
                                   PARTITION BY run_id, question_id ORDER BY question
                               ) AS occurrence
                        FROM EVAL_RESULTS
                        WHERE run_id IN (%s, %s)
                    )
                    SELECT b.correct AS baseline_correct, c.correct AS current_correct
                    FROM numbered b
                    JOIN numbered c
                      ON b.question_id = c.question_id AND b.occurrence = c.occurrence
                    WHERE b.run_id = %s AND c.run_id = %s
                    """,
                    (baseline_run_id, current_run_id, baseline_run_id, current_run_id),
                )
                pairs = cur.fetchall()
        finally:
            conn.close()

        if not pairs:
            raise RuntimeError(
                "Coverage matched but no question_id paired between the two runs -- "
                "nothing to test. Check that both runs cover the same question slice."
            )

        fixed = sum(1 for b, c in pairs if c and not b)
        broken = sum(1 for b, c in pairs if b and not c)
        p = mcnemar_exact_two_sided(fixed, broken)
        significant = p < ALPHA

        summary = {
            "current_run_id": current_run_id,
            "baseline_run_id": baseline_run_id,
            "paired_questions": len(pairs),
            "baseline_accuracy": float(base_row[2]),
            "current_accuracy": float(cur_row[2]),
            "fixed": fixed,
            "broken": broken,
            "net": fixed - broken,
            "p_value": round(p, 6),
            "alpha": ALPHA,
        }
        for key, value in summary.items():
            print(f"  {key:<20} {value}")

        if significant and broken > fixed:
            raise RuntimeError(
                f"SIGNIFICANT DEGRADATION: {broken} questions broke vs {fixed} fixed "
                f"across {len(pairs)} paired questions "
                f"(McNemar exact two-sided p={p:.4f} < {ALPHA}).\n"
                f"  baseline accuracy {base_row[2]} -> current {cur_row[2]}\n"
                "Inspect question-by-question with:\n"
                "  python -m scripts.compare_ablation <baseline.csv> <current.csv>"
            )

        if significant:
            print(f"\nSignificant IMPROVEMENT (p={p:.4f}): {fixed} fixed vs {broken} broken.")
        else:
            print(f"\nNo significant change (p={p:.4f} >= {ALPHA}); net {fixed - broken:+d}.")

        return summary

    drift_check(dbt_build(load_to_snowflake(run_benchmark())))


nl2sql_eval()
