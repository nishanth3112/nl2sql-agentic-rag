FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/uv

WORKDIR /app

# The venv's bin first on PATH, so CMD and the compose `streamlit run` command
# resolve exactly as they did with a system-wide install.
ENV PATH="/app/.venv/bin:$PATH"

# Dependencies before the code, so editing source does not reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
