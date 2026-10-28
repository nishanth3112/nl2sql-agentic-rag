# Dataset setup

This project benchmarks against **BIRD-SQL Mini-Dev** (500 examples, 11
SQLite databases): california_schools, card_games, codebase_community,
debit_card_specializing, european_football_2, financial, formula_1,
student_club, superhero, thrombosis_prediction, toxicology.

## Getting the data

Two options -- pick whichever works more smoothly for you:

**Option A -- Hugging Face `datasets` library:**
```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('birdsql/bird_mini_dev', trust_remote_code=True)"
```
This is the official BIRD team's own Hugging Face org (`birdsql`), not a
third-party mirror. Check where it lands files on your machine (typically
under `~/.cache/huggingface/datasets/`) and either point
`BENCHMARK_DATA_PATH` there directly, or copy/symlink the relevant folders
into `data/bird-mini-dev/` to match the layout below.

**Option B -- direct download from the BIRD team's site:**
Check [bird-bench.github.io](https://bird-bench.github.io) for current
download links (the dev/mini-dev zip locations occasionally move). Extract
so this directory looks like:

```
data/bird-mini-dev/
  dev.json                          # question, evidence, SQL, db_id, difficulty per example
  dev_databases/
    california_schools/california_schools.sqlite
    card_games/card_games.sqlite
    ...
```

## Verify the format before running anything

BIRD's team released a quality-reviewed dev split (`bird-sql-dev-1106`) in
November 2025 -- field names *should* be stable (`db_id`, `question`,
`evidence`, `SQL`, `difficulty`), but open `dev.json` yourself and confirm
before trusting `eval/run_benchmark.py`'s assumptions blindly. This is worth
doing as a habit regardless of the specific benchmark -- verify the actual
file over trusting documentation (including this file), since releases
drift over time.

## Indexing schemas into the vector store

Unlike the original plan, you don't need to hand-write per-table
descriptions -- introspect the .sqlite files directly, which is simpler and
always in sync with the real database:

```python
import sqlite3
from src.retrieval.vector_store import SchemaVectorStore

def build_tables_dict(sqlite_path: str) -> dict:
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {}
    for (table_name,) in cursor.fetchall():
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = [row[1] for row in cursor.fetchall()]  # row[1] = column name
        tables[table_name] = {"columns": columns, "description": ""}
    conn.close()
    return tables

store = SchemaVectorStore()
db_ids = ["california_schools", "card_games", "codebase_community",
          "debit_card_specializing", "european_football_2", "financial",
          "formula_1", "student_club", "superhero", "thrombosis_prediction",
          "toxicology"]

for db_id in db_ids:
    sqlite_path = f"data/bird-mini-dev/dev_databases/{db_id}/{db_id}.sqlite"
    store.index_schema(db_id, build_tables_dict(sqlite_path))
    print(f"Indexed {db_id}")
```

Worth saving this as `data/build_schema_index.py` once you've confirmed the
real folder layout matches -- run it once before the smoke test, and again
any time you re-download the dataset into a fresh `chroma_db`.

## On the `evidence` field

BIRD's `evidence` gives domain-knowledge hints that make questions
meaningfully easier. This project runs **without** feeding it into agent
prompts by default (`INCLUDE_EVIDENCE_IN_PROMPTS=false` in `.env`) -- a
harder, more honest test of whether the schema retriever agent can supply
enough context on its own. `evidence` is still captured per-example in
the results CSV for analysis even when unused, so you can check afterward
how much it would have helped on the questions the pipeline got wrong.
