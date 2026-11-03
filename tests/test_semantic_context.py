"""
Semantic definitions are selected by retrieved table, and every committed
definition is well formed and traceable to an allowed source.

The source check is the leakage control: a definition may cite a BIRD
description column or a calibration question, never a held-out one. Checks
that need a database or its description CSVs skip when those are absent, as
five of the eleven are in CI (gitignored).
"""

import csv
import json
import os
import re

import pytest

from data.build_schema_index import DB_IDS, sqlite_path_for
from src import semantic_context as sc
from src.config import settings
from src.db.connection import connect_readonly

SOURCE_RE = re.compile(r"^(description:([^.]+)\.(.+)|calibration:(\d+))$")


def write(tmp_path, db_id, definitions):
    (tmp_path / f"{db_id}.json").write_text(json.dumps(definitions), encoding="utf-8")


def d(text, *tables):
    return {"kind": "encoding", "tables": list(tables), "text": text, "source": "calibration:1"}


def test_no_file_means_no_notes(tmp_path):
    assert sc.load("nowhere", tmp_path) == []
    assert sc.for_tables("nowhere", ["t"], tmp_path) == ""


def test_selects_definitions_for_retrieved_tables_in_file_order(tmp_path):
    write(tmp_path, "db", [d("first", "a"), d("skipped", "c"), d("spans", "c", "b")])
    assert sc.for_tables("db", ["b", "a"], tmp_path) == "- first\n- spans"


def test_nothing_retrieved_selects_nothing(tmp_path):
    write(tmp_path, "db", [d("first", "a")])
    assert sc.for_tables("db", [], tmp_path) == ""


def split_ids(name):
    with open(f"eval/splits/{name}_ids.json", encoding="utf-8") as f:
        return set(json.load(f)["question_ids"])


def committed_files():
    if not sc.SEMANTIC_DIR.is_dir():
        return []
    return sorted(p.stem for p in sc.SEMANTIC_DIR.glob("*.json"))


def table_names(db_id):
    conn = connect_readonly(sqlite_path_for(db_id))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def described_columns(db_id, table):
    path = f"{settings.benchmark_data_path}/dev_databases/{db_id}/database_description/{table}.csv"
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        return {(r.get("original_column_name") or r.get("column_name") or "").strip() for r in csv.DictReader(f)}


def assert_valid(db_id, definitions):
    assert db_id in DB_IDS
    calibration, heldout = split_ids("calibration"), split_ids("heldout")
    have_db = os.path.isfile(sqlite_path_for(db_id))
    tables = table_names(db_id) if have_db else None

    for definition in definitions:
        assert set(definition) == {"kind", "tables", "text", "source"}, definition
        assert definition["kind"] in sc.KINDS, definition
        assert definition["text"].strip(), definition
        assert definition["tables"] and all(isinstance(t, str) for t in definition["tables"]), definition
        if tables is not None:
            assert set(definition["tables"]) <= tables, definition

        match = SOURCE_RE.match(definition["source"])
        assert match, definition
        if match.group(4):
            qid = int(match.group(4))
            assert qid in calibration and qid not in heldout, definition
        elif have_db:
            table, column = match.group(2), match.group(3)
            assert column in described_columns(db_id, table), definition


@pytest.mark.parametrize("db_id", committed_files())
def test_committed_definitions_are_well_formed_and_sourced(db_id):
    assert_valid(db_id, sc.load(db_id))


@pytest.mark.parametrize("split", ["heldout", "calibration"])
def test_source_check_rejects_a_heldout_question(split):
    qid = min(split_ids(split))
    definition = {"kind": "formula", "tables": ["account"], "text": "x", "source": f"calibration:{qid}"}
    if split == "heldout":
        with pytest.raises(AssertionError):
            assert_valid("financial", [definition])
    else:
        assert_valid("financial", [definition])


def test_source_check_rejects_unknown_kind_and_bad_source():
    for bad in ({"kind": "guess", "tables": ["account"], "text": "x", "source": "calibration:1"},
                {"kind": "formula", "tables": ["account"], "text": "x", "source": "evidence:5"}):
        with pytest.raises(AssertionError):
            assert_valid("financial", [bad])
