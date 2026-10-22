# CLAUDE.md

Agentic NL2SQL: a LangGraph multi-agent text-to-SQL system evaluated on BIRD Mini-Dev. `README.md` covers architecture, results and known limits; `docs/ROADMAP.md` tracks the current phase.

## Working rules (distilled from Karpathy's notes on LLM coding pitfalls)
1. Think before coding: state assumptions, surface tradeoffs, ask when a request is ambiguous.
2. Simplicity first: write the minimum code the task needs. No speculative abstractions, flags or config.
3. Surgical changes: touch only what the task requires. Don't reformat, rename or "improve" nearby code; remove only what your change made unused.
4. Goal-driven: turn each task into a check (failing test, metric, command) and loop until it passes.

## Commands
- `uv sync` and `uv run pytest`
- `uv run uvicorn api.main:app --reload` and `uv run streamlit run ui/app.py`
- `uv run python -m eval.run_benchmark` spends API credit: ask first.

## Invariants
- uv only; never pip or venv.
- Database access stays read-only (`src/db/connection.py`, `src/db/sql_guard.py`). Test a security control by attempting what it forbids.
- ~1Gi container budget: no PyTorch; keep the ONNX embedder.
- Never edit committed result CSVs in `eval/`; write new ones.
- BIRD `evidence` never feeds prompts or semantic definitions. Author on the calibration split, report on held-out.
- Every reported number states n, split and config, backed by a committed per-question CSV.
- Config changes update `.env.example`, the k8s ConfigMap and Terraform together.
