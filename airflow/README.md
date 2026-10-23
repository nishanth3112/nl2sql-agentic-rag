# Airflow

Orchestrates the benchmark pipeline: run the benchmark, load results into
Snowflake, build the dbt models.

The image is built from [`Dockerfile`](Dockerfile), which extends
`apache/airflow:3.3.1` with the project's dependencies from `pyproject.toml`,
installed with uv. The build context is the **project root**, not `airflow/`,
so the Dockerfile's `COPY` can reach `pyproject.toml` one level up. `docker build airflow/` will fail; use
`docker compose build` from this directory.

## Required: `SNOWFLAKE_ACCOUNT` is not set

The containers read the project's `.env` (`../.env`) directly, so API keys and
model settings are defined in one place. **`SNOWFLAKE_ACCOUNT` is not in that
file**, and unlike every other Snowflake setting it has no default in
`src/config.py` — an account identifier is installation-specific and guessing
one is worse than failing.

Any task that touches Snowflake will raise:

```
ValueError: SNOWFLAKE_ACCOUNT is not set. This is your account identifier
(e.g. 'abc12345.us-east-1' or 'myorg-myaccount'); it has no default because
it is specific to your Snowflake installation.
```

Add it to the **project** `.env` (not `airflow/.env`):

```
SNOWFLAKE_ACCOUNT=<your-account>.<region>
```

The remaining Snowflake settings (`SNOWFLAKE_USER`, `SNOWFLAKE_WAREHOUSE`,
`SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_ROLE`) fall back to the
defaults in `src/config.py` and only need setting to override them.

## Do not use chroma_db/ from the host while a DAG is running

`../chroma_db` is bind-mounted read-write into every Airflow service, and
**Chroma opens its SQLite store read-write even to answer a query** — so this
mount cannot be made read-only.

That means the local API and Streamlit UI share one SQLite file with the
Airflow workers:

```bash
uv run uvicorn api.main:app --port 8000     # DON'T, while a DAG task is running
uv run streamlit run ui/app.py              # DON'T, while a DAG task is running
```

Concurrent writers to one SQLite database across a Docker bind mount produce
lock contention and `database is locked` errors, and the failure is
intermittent — it depends on who holds the write lock when. On Windows/WSL2
bind mounts this is notably less reliable than on native Linux.

**Run one at a time.** Stop the host API and UI before triggering a DAG that
touches the schema index. Note this applies between Airflow services too: two
tasks embedding simultaneously hit the same contention, so keep
Chroma-touching tasks off parallel branches.

## Paths inside the container

The project `.env` sets host-relative paths (`./chroma_db`,
`./data/bird-mini-dev`) that mean nothing in a container, so
`docker-compose.yaml` overrides them under `environment:` — which takes
precedence over `env_file:`.

| Setting | Container value |
| --- | --- |
| `PYTHONPATH` | `/opt/airflow/project` |
| `BENCHMARK_DATA_PATH` | `/opt/airflow/project/data/bird-mini-dev` |
| `CHROMA_PERSIST_DIR` | `/opt/airflow/project/chroma_db` |
| `SNOWFLAKE_PRIVATE_KEY_PATH` | `/opt/secrets/rsa_key.p8` (read-only mount) |

The key path is absolute, not `~/...`: `src/db/snowflake_connection.py` calls
`Path.expanduser()` but dbt does not, and that difference has already caused
one failure here.

## Local files

`airflow/.env` holds `AIRFLOW_UID` and a generated `FERNET_KEY`. It is
gitignored (by the repo-wide `.env` rule), so a fresh clone will not have it —
recreate with:

```bash
echo "AIRFLOW_UID=50000" > airflow/.env
python -c "from cryptography.fernet import Fernet; print('FERNET_KEY=' + Fernet.generate_key().decode())" >> airflow/.env
```

`logs/`, `plugins/`, and `config/` are gitignored runtime state. `dags/`,
`Dockerfile`, and `docker-compose.yaml` are tracked — they are the pipeline
definition.

## Running

```bash
cd airflow
docker compose build
docker compose run --rm airflow-cli airflow version   # sanity check
docker compose up -d
```

UI at http://localhost:8080 (`airflow` / `airflow`). Example DAGs are disabled
(`AIRFLOW__CORE__LOAD_EXAMPLES: 'false'`).
