"""
The Snowflake loader with and without the cost-tracing columns.

Free and offline: the connection is a recording fake and write_pandas is stubbed,
so no account is needed. What can't be checked here is the DDL itself and the
real COPY -- see the migration in snowflake_sql/schema.sql.
"""

import csv
import json

import pandas as pd
import pytest

from eval import load_to_snowflake as loader
from eval import run_benchmark as rb

OLD_RESULTS_COLUMNS = [
    "RESULT_ID", "RUN_ID", "QUESTION_ID", "DB_ID", "QUESTION", "EVIDENCE", "GOLD_SQL", "PREDICTED_SQL",
    "CORRECT", "DIFFICULTY", "RETRIES", "ERROR_CLASSES_HIT", "FATAL_ERROR", "SCHEMA_CONTEXT_CHARS", "SECONDS",
]
COST_COLUMNS = [c.upper() for c in (*loader.INTEGER_COLUMNS, *loader.DECIMAL_COLUMNS)]
OLD_RUNS_COLUMNS = [
    "RUN_ID", "RUN_TIMESTAMP", "CONFIG_LABEL", "INCLUDE_EVIDENCE_IN_PROMPTS", "PLANNER_PROVIDER", "PLANNER_MODEL",
    "GENERATOR_PROVIDER", "GENERATOR_MODEL", "CLASSIFIER_PROVIDER", "CLASSIFIER_MODEL", "TOTAL_QUESTIONS",
    "SOURCE_FILE_HASH",
]

COMMITTED = {"eval/results_500_baseline.csv": 500, "eval/results_150_with_evidence.csv": 150}


def _table_columns(cost=True, pricing=True, split=True):
    rows = [("EVAL_RESULTS", c) for c in OLD_RESULTS_COLUMNS]
    if cost:
        rows += [("EVAL_RESULTS", c) for c in COST_COLUMNS]
    rows += [("EVAL_RUNS", c) for c in OLD_RUNS_COLUMNS]
    if pricing:
        rows.append(("EVAL_RUNS", "PRICING_AS_OF"))
    if split:
        rows.append(("EVAL_RUNS", "SPLIT"))
    return rows


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        if "INFORMATION_SCHEMA.COLUMNS" in sql:
            self.rows = list(self.conn.table_columns)
        elif "source_file_hash = %s" in sql:
            self.rows = list(self.conn.existing_runs)
        else:
            self.rows = []

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, table_columns, existing_runs=()):
        self.table_columns = table_columns
        self.existing_runs = existing_runs
        self.executed = []
        self.commits = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        self.closed = True

    def inserts(self):
        return [(sql, params) for sql, params in self.executed if sql.strip().startswith("INSERT INTO EVAL_RUNS")]


@pytest.fixture
def snowflake(monkeypatch):
    """Returns a function that installs a fake connection and records write_pandas calls."""
    written = []

    def install(table_columns, existing_runs=()):
        conn = FakeConnection(table_columns, existing_runs)
        monkeypatch.setattr(loader, "get_snowflake_connection", lambda: conn)

        def fake_write_pandas(connection, df, table_name, database, schema):
            written.append(df)
            return True, 1, len(df), []

        monkeypatch.setattr(loader, "write_pandas", fake_write_pandas)
        return conn

    return install, written


def _new_csv(tmp_path, rows=None):
    base = {
        "index": 0, "question_id": 100, "db_id": "superhero", "question": "q", "evidence": "", "gold_sql": "SELECT 1",
        "predicted_sql": "SELECT 1", "correct": True, "difficulty": "simple", "retries": 0, "error_classes_hit": "",
        "fatal_error": "", "schema_context_chars": 120, "seconds": 4.2,
        "llm_calls": 2, "prompt_tokens": 3000, "cached_prompt_tokens": 500, "completion_tokens": 250,
        "reasoning_tokens": 100, "est_cost_usd": 0.001275,
        "retriever_seconds": 0.25, "planner_seconds": 2.0, "generator_seconds": 1.5,
        "executor_seconds": 0.125, "classifier_seconds": 0.0,
    }
    rows = rows or [
        base,
        {**base, "index": 1, "question_id": 101, "fatal_error": "RuntimeError: boom", "correct": False,
         **{c: "" for c in rb.FIELDNAMES[14:]}},                      # crashed: every cost cell unknown
        {**base, "index": 2, "question_id": 102, "est_cost_usd": ""},  # unpriced model: tokens known, cost not
        {**base, "index": 3, "question_id": 103, "llm_calls": "n/a"},  # junk where a count belongs
    ]
    path = tmp_path / "results_new.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rb.FIELDNAMES, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _metadata(tmp_path, **extra):
    path = tmp_path / "meta.json"
    meta = {
        "run_timestamp": "2026-10-25T12:00:00+00:00", "planner_provider": "openai", "planner_model": "gpt-5-mini",
        "generator_provider": "openai", "generator_model": "gpt-5-mini", "classifier_provider": "groq",
        "classifier_model": "openai/gpt-oss-20b", "include_evidence_in_prompts": False, **extra,
    }
    path.write_text(json.dumps(meta), encoding="utf-8")
    return str(path)


# --- reading a CSV --------------------------------------------------------------


@pytest.mark.parametrize("path,rows", COMMITTED.items())
def test_a_committed_csv_without_cost_columns_still_prepares(path, rows):
    df = loader.prepare_results(path, "run-1")

    assert len(df) == rows
    assert list(df.columns)[0] == "RUN_ID"
    assert all(c == c.upper() for c in df.columns)
    assert not set(COST_COLUMNS) & set(df.columns)
    assert df["CORRECT"].dtype == bool
    assert not (df["FATAL_ERROR"] == "").any()


def test_cost_columns_are_typed_and_blank_means_null_not_zero(tmp_path):
    df = loader.prepare_results(str(_new_csv(tmp_path)), "run-1")

    assert str(df["LLM_CALLS"].dtype) == "Int64"
    assert str(df["PROMPT_TOKENS"].dtype) == "Int64"
    assert df.loc[0, "LLM_CALLS"] == 2 and df.loc[0, "PROMPT_TOKENS"] == 3000
    assert df.loc[0, "EST_COST_USD"] == pytest.approx(0.001275)

    crashed = df.loc[1]
    assert all(pd.isna(crashed[c]) for c in COST_COLUMNS)  # unknown, and NOT 0

    unpriced = df.loc[2]
    assert unpriced["PROMPT_TOKENS"] == 3000
    assert pd.isna(unpriced["EST_COST_USD"])


def test_text_where_a_count_belongs_becomes_null_instead_of_failing_the_load(tmp_path):
    df = loader.prepare_results(str(_new_csv(tmp_path)), "run-1")

    assert pd.isna(df.loc[3, "LLM_CALLS"])
    assert df.loc[3, "PROMPT_TOKENS"] == 3000


def test_a_zero_cost_is_kept_as_zero(tmp_path):
    base_row: dict = {c: "" for c in rb.FIELDNAMES}
    base_row.update({"question_id": 1, "db_id": "d", "correct": True, "difficulty": "simple",
                     "llm_calls": 0, "est_cost_usd": 0.0})

    df = loader.prepare_results(str(_new_csv(tmp_path, rows=[base_row])), "run-1")

    assert df.loc[0, "LLM_CALLS"] == 0
    assert df.loc[0, "EST_COST_USD"] == 0.0 and not pd.isna(df.loc[0, "EST_COST_USD"])


def test_a_file_whose_cost_columns_are_all_blank_loads_as_all_null(tmp_path):
    # Every row resumed from a file written before cost tracing.
    blank = {c: "" for c in rb.FIELDNAMES}
    rows = [{**blank, "question_id": i, "db_id": "d", "correct": True, "difficulty": "simple"} for i in range(3)]

    df = loader.prepare_results(str(_new_csv(tmp_path, rows=rows)), "run-1")

    assert all(df[c].isna().all() for c in COST_COLUMNS)
    assert str(df["LLM_CALLS"].dtype) == "Int64"


# --- checking the tables before writing -----------------------------------------


def test_missing_columns_are_found_case_insensitively():
    conn = FakeConnection(_table_columns(cost=False, pricing=False))

    missing = loader.find_missing_columns(
        conn, {"eval_results": ["RUN_ID", "llm_calls", "EST_COST_USD"], "EVAL_RUNS": ["pricing_as_of"]}
    )

    assert missing == {"EVAL_RESULTS": ["llm_calls", "EST_COST_USD"], "EVAL_RUNS": ["pricing_as_of"]}


def test_nothing_is_missing_after_the_migration():
    conn = FakeConnection(_table_columns())

    assert loader.find_missing_columns(conn, {"EVAL_RESULTS": COST_COLUMNS, "EVAL_RUNS": ["PRICING_AS_OF"]}) == {}


def test_a_table_with_nothing_needed_is_not_reported():
    conn = FakeConnection(_table_columns(cost=False, pricing=False))

    assert loader.find_missing_columns(conn, {"EVAL_RESULTS": ["RUN_ID"], "EVAL_RUNS": []}) == {}


# --- the whole load -------------------------------------------------------------


def _main(csv_path, metadata_path, *extra):
    return loader.main(["--csv", str(csv_path), "--config-label", "t", "--metadata", metadata_path, *extra])


def test_a_new_csv_is_refused_before_anything_is_written_when_the_tables_lack_its_columns(tmp_path, snowflake):
    install, written = snowflake
    conn = install(_table_columns(cost=False, pricing=False))

    with pytest.raises(SystemExit) as exc:
        _main(_new_csv(tmp_path), _metadata(tmp_path, pricing_as_of="2026-10-25"))

    message = str(exc.value)
    assert "schema.sql" in message and "Nothing was written" in message
    assert "LLM_CALLS" in message and "pricing_as_of" in message
    assert conn.inserts() == []   # no orphan EVAL_RUNS row to block the retry
    assert written == []
    assert conn.commits == 0
    assert conn.closed is True


def test_the_runs_table_is_checked_too(tmp_path, snowflake):
    install, written = snowflake
    conn = install(_table_columns(cost=True, pricing=False))

    with pytest.raises(SystemExit) as exc:
        _main(_new_csv(tmp_path), _metadata(tmp_path, pricing_as_of="2026-10-25"))

    assert "EVAL_RUNS: pricing_as_of" in str(exc.value)
    assert conn.inserts() == [] and written == []


def test_a_new_csv_loads_after_the_migration_and_records_the_pricing_date(tmp_path, snowflake):
    install, written = snowflake
    conn = install(_table_columns())

    assert _main(_new_csv(tmp_path), _metadata(tmp_path, pricing_as_of="2026-10-25")) == 0

    (sql, params), = conn.inserts()
    assert "pricing_as_of" in sql
    assert params[-1] == "2026-10-25"
    assert sql.count("%s") == len(params)
    (df,) = written
    assert set(COST_COLUMNS) <= set(df.columns)
    assert len(df) == 4
    assert conn.commits == 1


def test_an_old_csv_still_loads_into_tables_that_have_not_been_migrated(tmp_path, snowflake):
    # The proof that adding the columns breaks nothing that already worked.
    install, written = snowflake
    conn = install(_table_columns(cost=False, pricing=False))

    code = loader.main([
        "--csv", "eval/results_150_with_evidence.csv", "--config-label", "old",
        "--metadata", str(tmp_path / "absent.json"), "--run-timestamp", "2026-08-10T19:25:20-04:00",
    ])

    assert code == 0
    (sql, params), = conn.inserts()
    assert "pricing_as_of" not in sql
    assert sql.count("%s") == len(params) == 12
    (df,) = written
    assert len(df) == 150 and not set(COST_COLUMNS) & set(df.columns)
    assert conn.commits == 1


def test_a_duplicate_load_is_still_refused_first(tmp_path, snowflake):
    install, written = snowflake
    conn = install(_table_columns(), existing_runs=[("run-0", "baseline", "2026-10-01 00:00:00")])

    with pytest.raises(SystemExit, match="already in EVAL_RUNS"):
        _main(_new_csv(tmp_path), _metadata(tmp_path, pricing_as_of="2026-10-25"))

    assert conn.inserts() == [] and written == []


def test_a_metadata_file_without_a_pricing_date_gives_none():
    args = loader.parse_args(["--csv", "x.csv", "--config-label", "t"])

    config = loader.resolve_config(args, {"run_timestamp": "2026-10-25T12:00:00+00:00"})

    assert config["pricing_as_of"] is None


def test_the_split_column_is_a_run_property_not_a_result_column(tmp_path):
    rows = [{"index": 0, "question_id": 100, "db_id": "superhero", "question": "q", "evidence": "",
             "gold_sql": "SELECT 1", "predicted_sql": "SELECT 1", "correct": True, "difficulty": "simple",
             "retries": 0, "error_classes_hit": "", "fatal_error": "", "schema_context_chars": 1, "seconds": 1.0,
             "split": "heldout"}]

    df = loader.prepare_results(str(_new_csv(tmp_path, rows)), "run-1")

    assert "SPLIT" not in df.columns


# --- the split recorded on the run ----------------------------------------------


def test_a_metadata_file_without_a_split_gives_none():
    args = loader.parse_args(["--csv", "x.csv", "--config-label", "t"])

    config = loader.resolve_config(args, {"run_timestamp": "2026-10-25T12:00:00+00:00"})

    assert config["split"] is None


def test_a_run_with_a_split_records_it_on_the_runs_row(tmp_path, snowflake):
    install, written = snowflake
    conn = install(_table_columns())

    assert _main(_new_csv(tmp_path), _metadata(tmp_path, pricing_as_of="2026-10-25", split="heldout")) == 0

    (sql, params), = conn.inserts()
    assert ", split)" in sql and params[-1] == "heldout"
    assert sql.count("%s") == len(params)
    (df,) = written
    assert "SPLIT" not in df.columns


def test_a_run_with_a_split_is_refused_before_anything_is_written_when_the_runs_table_lacks_it(tmp_path, snowflake):
    install, written = snowflake
    conn = install(_table_columns(split=False))

    with pytest.raises(SystemExit) as exc:
        _main(_new_csv(tmp_path), _metadata(tmp_path, pricing_as_of="2026-10-25", split="heldout"))

    assert "EVAL_RUNS: split" in str(exc.value) and "Nothing was written" in str(exc.value)
    assert conn.inserts() == [] and written == []


def test_a_run_without_a_split_still_loads_into_a_runs_table_that_lacks_the_column(tmp_path, snowflake):
    install, written = snowflake
    conn = install(_table_columns(split=False))

    assert _main(_new_csv(tmp_path), _metadata(tmp_path, pricing_as_of="2026-10-25")) == 0

    (sql, params), = conn.inserts()
    assert "split" not in sql
