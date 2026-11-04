"""
Semantic definitions: domain knowledge a schema does not carry -- what a coded
value means, how a business quantity is computed, what a question's words map
to, and which column holds an entity's display name (so a question asking for
a name gets the name, not the ID).

One JSON file per database, data/semantic_context/<db_id>.json, holding a list
of {"kind", "tables", "text", "source"}. A source is either
"description:<table>.<column>" (BIRD's database_description CSVs) or
"calibration:<question_id>". Nothing is authored from held-out questions or
from BIRD `evidence`; tests/test_semantic_context.py checks every source.

A definition is selected when any of its tables was retrieved for the
question. Riding on the schema retriever means no second retrieval step that
could miss on its own.
"""

import json
from pathlib import Path
from typing import Iterable, List

KINDS = {"encoding", "formula", "synonym", "display_name"}

SEMANTIC_DIR = Path(__file__).resolve().parent.parent / "data" / "semantic_context"


def load(db_id: str, directory: Path = SEMANTIC_DIR) -> List[dict]:
    """A database with no file (an upload, or one not authored yet) has no definitions."""
    path = Path(directory) / f"{db_id}.json"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def for_tables(db_id: str, table_names: Iterable[str], directory: Path = SEMANTIC_DIR) -> str:
    """Prompt-ready notes for the retrieved tables, in file order; empty when none apply."""
    retrieved = set(table_names)
    lines = [f"- {d['text']}" for d in load(db_id, directory) if retrieved & set(d["tables"])]
    return "\n".join(lines)
