"""
Multi-provider LLM dispatch.

Three providers are wired up: Gemini, Groq (OpenAI-compatible endpoint,
very fast LPU inference), and OpenAI.

Each agent node picks its provider/model independently via src/config.py,
so you can mix and match by task.

Provider quirks handled here rather than leaking into agent code:
  1. Newer OpenAI models reject the legacy `max_tokens` param and require
     `max_completion_tokens` instead.
  2. Those same models are REASONING models -- their token cap covers
     internal reasoning plus visible output, so a budget sized only for the
     expected output can be fully consumed by reasoning, returning an empty
     string with no error.
  3. Requests that hang open (rather than fail) would block a long run
     forever, since backoff only catches failures -- hence explicit timeouts.
"""

import time

from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from src import usage
from src.config import settings

_gemini_client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None


# --- billing exhaustion ---------------------------------------------------------
#
# An exhausted account answers HTTP 429, the same status as a per-minute rate
# limit, so both retry layers treated it as transient: the SDK made 3 attempts
# per call, our backoff repeated that 5 times, and only then did the question
# fail. Retrying cannot help -- no credit appears between attempts -- so on the
# first night of a fresh baseline every question after the balance ran out
# burned ~143s and was recorded as a wrong answer. This is not a model failure,
# and scoring it as one silently corrupts a run.
#
# Detection is on OpenAI's billing codes, not on 429 alone: a genuine rate
# limit ('rate_limit_exceeded') is still retried exactly as before.
_QUOTA_MARKERS = ("insufficient_quota", "credit_balance_exhausted")


class QuotaExhaustedError(RuntimeError):
    """The provider account has no credit. Not retryable; the caller must stop."""


def is_quota_exhausted(exc: BaseException) -> bool:
    """True if `exc`, or anything in its cause chain, is billing exhaustion."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, QuotaExhaustedError):
            return True
        if any(m in str(exc) for m in _QUOTA_MARKERS):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


class _OpenAICompatible(OpenAI):
    """OpenAI client that does not retry an out-of-credit 429.

    _should_retry is the SDK's own retry decision, called with the failed
    response. Every other case -- real rate limits, 408, 409, 5xx -- is left to
    the SDK's default logic unchanged."""

    def _should_retry(self, response) -> bool:
        if response.status_code == 429:
            try:
                body = response.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 -- unreadable body: fall back to default logic
                body = ""
            if any(m in body for m in _QUOTA_MARKERS):
                return False
        return super()._should_retry(response)


# timeout + max_retries: without an explicit timeout, a request that opens but
# never returns will block the entire benchmark indefinitely -- our backoff
# layer only catches calls that FAIL, not ones that hang. 120s is generous for
# a reasoning model; max_retries=2 lets the SDK absorb transient network blips
# before our own backoff ever sees an error.
_openai_client = (
    _OpenAICompatible(api_key=settings.openai_api_key, timeout=120.0, max_retries=2)
    if settings.openai_api_key
    else None
)
_groq_client = (
    _OpenAICompatible(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=120.0,
        max_retries=2,
    )
    if settings.groq_api_key
    else None
)


def _with_backoff(fn, max_attempts: int = 5, initial_delay: float = 2.0):
    delay = initial_delay
    last_error = None
    for _ in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if is_quota_exhausted(e):
                raise QuotaExhaustedError(f"Provider account has no credit remaining: {e}") from e
            msg = str(e).lower()
            if "429" not in msg and "resource_exhausted" not in msg and "rate_limit" not in msg:
                raise  # not a rate-limit error -- don't retry blindly
            last_error = e
            time.sleep(delay)
            delay *= 2
    raise last_error


# --- usage metering ---------------------------------------------------------------
#
# One record per request that comes back, written right after the response
# arrives -- before any check that can raise on it. A reasoning model that burns
# its whole budget thinking returns empty content, and that request is billed all
# the same. Requests that fail outright (a rejected parameter, a 429) never
# produce a response, so they are not counted. Retries the SDK makes inside one
# create() call are invisible from here, so the totals can under-count them.
#
# getattr throughout: the usage block is Optional in the SDK, OpenAI-compatible
# providers omit pieces of it, and the test fakes carry none.


def _record_openai_usage(provider: str, model: str, response) -> None:
    block = getattr(response, "usage", None)
    choices = getattr(response, "choices", None) or []
    usage.record(
        provider,
        model,
        prompt_tokens=getattr(block, "prompt_tokens", None),
        # Already includes reasoning tokens.
        completion_tokens=getattr(block, "completion_tokens", None),
        reasoning_tokens=getattr(getattr(block, "completion_tokens_details", None), "reasoning_tokens", None),
        cached_prompt_tokens=getattr(getattr(block, "prompt_tokens_details", None), "cached_tokens", None),
        finish_reason=getattr(choices[0], "finish_reason", None) if choices else None,
    )


def _record_gemini_usage(model: str, response) -> None:
    block = getattr(response, "usage_metadata", None)
    answer_tokens = getattr(block, "candidates_token_count", None)
    thought_tokens = getattr(block, "thoughts_token_count", None)
    # Gemini bills thinking as output but reports it apart from the answer, so
    # completion is their sum, matching the OpenAI convention.
    completion_tokens = None
    if answer_tokens is not None or thought_tokens is not None:
        completion_tokens = (answer_tokens or 0) + (thought_tokens or 0)
    candidates = getattr(response, "candidates", None) or []
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    usage.record(
        "gemini",
        model,
        prompt_tokens=getattr(block, "prompt_token_count", None),
        completion_tokens=completion_tokens,
        reasoning_tokens=thought_tokens,
        cached_prompt_tokens=getattr(block, "cached_content_token_count", None),
        finish_reason=None if finish_reason is None else str(getattr(finish_reason, "name", finish_reason)),
    )


def _generate_gemini(model: str, system_prompt: str, user_prompt: str, max_output_tokens: int, json_mode: bool) -> str:
    def _call(use_json_mode: bool) -> str:
        config_kwargs = {"system_instruction": system_prompt, "max_output_tokens": max_output_tokens}
        if use_json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        response = _gemini_client.models.generate_content(
            model=model, contents=user_prompt, config=genai_types.GenerateContentConfig(**config_kwargs)
        )
        # Before .text, which can raise on a blocked or empty response that was still billed.
        _record_gemini_usage(model, response)
        return response.text

    if not json_mode:
        return _with_backoff(lambda: _call(False))
    try:
        return _with_backoff(lambda: _call(True))
    except Exception:
        return _with_backoff(lambda: _call(False))  # model may not support the mime-type constraint


class _NeedsReasoningHeadroom(Exception):
    """Internal signal: empty output, cut off by length on max_tokens."""


def _generate_openai_compatible(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    json_mode: bool,
    provider: str = "unknown",
) -> str:
    def _call(use_json_mode: bool, tokens_param: str, reasoning_headroom: bool = False) -> str:
        # Models on the max_completion_tokens param are reasoning models --
        # give them room to think on top of the requested output size.
        budget = max_output_tokens
        if tokens_param == "max_completion_tokens" or reasoning_headroom:
            budget += settings.reasoning_token_budget

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokens_param: budget,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        # Before the empty-content check below: that request was billed even when it raises.
        _record_openai_usage(provider, model, response)
        choice = response.choices[0]
        content = choice.message.content

        if not content or not content.strip():
            if choice.finish_reason == "length" and tokens_param == "max_tokens" and not reasoning_headroom:
                # A reasoning model that ACCEPTS max_tokens (Groq's gpt-oss
                # does) spends that budget thinking and returns nothing --
                # the classifier's 20-token cap ran out on every call. The
                # max_completion_tokens path above never sees this, because
                # only providers that reject max_tokens are routed there.
                raise _NeedsReasoningHeadroom()
            # Loud failure instead of returning "" -- an empty string would
            # flow downstream as an empty SQL query and silently score as
            # wrong with no diagnosable cause.
            raise RuntimeError(
                f"Model {model!r} returned empty content (finish_reason={choice.finish_reason!r}, "
                f"budget={budget}). If finish_reason is 'length', raise REASONING_TOKEN_BUDGET."
            )
        return content

    def _call_with_token_param_fallback(use_json_mode: bool) -> str:
        # Try the legacy param first, and only switch if the provider
        # specifically complains -- rather than hardcoding either name and
        # breaking on whichever provider didn't make the change.
        try:
            return _call(use_json_mode, "max_tokens")
        except _NeedsReasoningHeadroom:
            # Retried once with the same headroom max_completion_tokens
            # models get. Only reached by a call that would otherwise have
            # raised, so every call that already succeeds is unchanged.
            return _call(use_json_mode, "max_tokens", reasoning_headroom=True)
        except Exception as e:
            msg = str(e)
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                return _call(use_json_mode, "max_completion_tokens")
            raise

    if not json_mode:
        return _with_backoff(lambda: _call_with_token_param_fallback(False))
    try:
        return _with_backoff(lambda: _call_with_token_param_fallback(True))
    except Exception:
        # Not every Groq-hosted model supports response_format=json_object --
        # fall back to plain text; callers already handle non-JSON gracefully.
        return _with_backoff(lambda: _call_with_token_param_fallback(False))


def generate_text(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 500,
    json_mode: bool = False,
) -> str:
    if provider == "gemini":
        if not _gemini_client:
            raise RuntimeError("GEMINI_API_KEY not set")
        return _generate_gemini(model, system_prompt, user_prompt, max_output_tokens, json_mode)

    if provider == "groq":
        if not _groq_client:
            raise RuntimeError("GROQ_API_KEY not set")
        return _generate_openai_compatible(
            _groq_client, model, system_prompt, user_prompt, max_output_tokens, json_mode, provider="groq"
        )

    if provider == "openai":
        if not _openai_client:
            raise RuntimeError("OPENAI_API_KEY not set")
        return _generate_openai_compatible(
            _openai_client, model, system_prompt, user_prompt, max_output_tokens, json_mode, provider="openai"
        )

    raise ValueError(f"Unknown provider: {provider!r} (expected 'gemini', 'groq', or 'openai')")