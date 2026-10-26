import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    benchmark_data_path: str = os.getenv("BENCHMARK_DATA_PATH", "./data/bird-mini-dev")
    include_evidence_in_prompts: bool = os.getenv("INCLUDE_EVIDENCE_IN_PROMPTS", "false").lower() == "true"
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    # Optional OpenTelemetry spans for graph nodes, written as JSON lines to a
    # local file. Off by default; see src/tracing.py.
    tracing_enabled: bool = os.getenv("TRACING_ENABLED", "false").lower() == "true"
    trace_export_path: str = os.getenv("TRACE_EXPORT_PATH", "./logs/spans.jsonl")

    # Reasoning models (OpenAI's GPT-5 family and o-series) spend tokens on
    # internal reasoning BEFORE emitting any visible output, and their token
    # cap covers reasoning + output COMBINED. Without headroom, the model can
    # burn the entire budget thinking and return an empty string -- with no
    # error raised, since the API call itself succeeded. This budget is added
    # on top of the caller's requested output size for models that require
    # the max_completion_tokens parameter. Raise it if empty responses
    # persist; lower it to cut cost.
    reasoning_token_budget: int = int(os.getenv("REASONING_TOKEN_BUDGET", "3000"))

    # Per-role provider + model. Mix and match freely across the three
    # providers wired up in src/llm_providers.py.
    #
    # NOTE on Gemini: it's fully wired up and works, but gemini-3.6-flash's
    # free tier is capped at 20 requests PER DAY -- unusable for a 500-question
    # benchmark. Groq's free tier (1,000 RPD for llama-3.3-70b-versatile,
    # 14,400 for llama-3.1-8b-instant) is the practical free choice, so the
    # defaults below route there instead.
    planner_provider: str = os.getenv("PLANNER_PROVIDER", "groq")
    planner_model: str = os.getenv("PLANNER_MODEL", "llama-3.3-70b-versatile")

    generator_provider: str = os.getenv("GENERATOR_PROVIDER", "openai")
    generator_model: str = os.getenv("GENERATOR_MODEL", "gpt-5-mini")

    # Was llama-3.1-8b-instant, which Groq retired; the 44.20% / +8.67pp
    # results in the README were measured on it. gpt-oss-20b is a reasoning
    # model -- it needs the empty-on-length retry in llm_providers.py, or its
    # 20-token classification budget is spent thinking and nothing comes back.
    classifier_provider: str = os.getenv("CLASSIFIER_PROVIDER", "groq")
    classifier_model: str = os.getenv("CLASSIFIER_MODEL", "openai/gpt-oss-20b")

    # Summarises a result set for the MCP explain tool. Not part of the
    # five-agent pipeline -- nothing benchmarked runs through it. Defaults to
    # the GENERATOR's model, not the classifier's: it was first tied to the
    # classifier as the cheaper option, and broke when Groq retired
    # llama-3.1-8b-instant. Following the generator means it tracks whichever
    # model the pipeline actually depends on working.
    explainer_provider: str = os.getenv("EXPLAINER_PROVIDER", generator_provider)
    explainer_model: str = os.getenv("EXPLAINER_MODEL", generator_model)

    # Snowflake, via key-pair auth (see src/db/snowflake_connection.py).
    #
    # snowflake_account has no default on purpose: an account identifier is
    # installation-specific and cannot be guessed, so a missing value should
    # fail loudly at connect time rather than silently point somewhere wrong.
    # Every other field defaults to the names this project provisions.
    #
    # The private key path keeps its literal "~" -- expansion happens at load
    # time in snowflake_connection.py, so an env-supplied "~" works too.
    snowflake_account: str = os.getenv("SNOWFLAKE_ACCOUNT", "")
    snowflake_user: str = os.getenv("SNOWFLAKE_USER", "NL2SQL_SERVICE_USER")
    snowflake_private_key_path: str = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "~/.snowflake/rsa_key.p8")
    snowflake_warehouse: str = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    snowflake_database: str = os.getenv("SNOWFLAKE_DATABASE", "NL2SQL_ANALYTICS")
    snowflake_schema: str = os.getenv("SNOWFLAKE_SCHEMA", "RAW")
    snowflake_role: str = os.getenv("SNOWFLAKE_ROLE", "NL2SQL_APP_ROLE")


settings = Settings()