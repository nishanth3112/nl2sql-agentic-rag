"""
Reasoning-model headroom in the OpenAI-compatible provider path.

Free and offline: a fake client records every request. The case under test is
the one that broke the classifier -- Groq's gpt-oss-20b accepts max_tokens,
spends the whole 20-token budget reasoning, and returns empty content with
finish_reason='length'.
"""

from types import SimpleNamespace

import pytest

from src import llm_providers, usage
from src.config import settings


class FakeClient:
    """Answers chat.completions.create from a script of (content, finish_reason[, usage]).

    A reply without a third element carries no `usage` attribute at all, like the
    fakes this file used before usage was metered."""

    def __init__(self, replies, reject_max_tokens=False, reject_json_mode=False):
        self.replies = list(replies)
        self.reject_max_tokens = reject_max_tokens
        self.reject_json_mode = reject_json_mode
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_max_tokens and "max_tokens" in kwargs:
            raise RuntimeError("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
        if self.reject_json_mode and "response_format" in kwargs:
            raise RuntimeError("response_format is not supported for this model")
        reply = self.replies.pop(0)
        content, finish = reply[0], reply[1]
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)]
        )
        if len(reply) > 2:
            response.usage = reply[2]
        return response


def _gen(client, max_output_tokens=20):
    return llm_providers._generate_openai_compatible(client, "m", "sys", "user", max_output_tokens, False)


def test_empty_on_length_retries_once_with_headroom():
    client = FakeClient([("", "length"), ("LOGIC_ERROR", "stop")])
    assert _gen(client) == "LOGIC_ERROR"
    assert [c["max_tokens"] for c in client.calls] == [20, 20 + settings.reasoning_token_budget]


def test_successful_first_call_is_unchanged():
    # The guarantee that lets this ship without moving other roles: a call
    # that already returns content makes exactly one request, budget as-is.
    client = FakeClient([("SCHEMA_ERROR", "stop")])
    assert _gen(client) == "SCHEMA_ERROR"
    assert len(client.calls) == 1
    assert client.calls[0]["max_tokens"] == 20


def test_empty_for_another_reason_is_not_retried():
    client = FakeClient([("", "stop")])
    with pytest.raises(RuntimeError, match="returned empty content"):
        _gen(client)
    assert len(client.calls) == 1


def test_headroom_is_tried_once_not_forever():
    client = FakeClient([("", "length"), ("", "length")])
    with pytest.raises(RuntimeError, match="returned empty content"):
        _gen(client)
    assert len(client.calls) == 2


def test_max_completion_tokens_path_is_unchanged():
    # OpenAI reasoning models (the planner and generator) reject max_tokens
    # and already get headroom on max_completion_tokens -- same as before.
    client = FakeClient([("SELECT 1", "stop")], reject_max_tokens=True)
    assert _gen(client) == "SELECT 1"
    assert client.calls[-1]["max_completion_tokens"] == 20 + settings.reasoning_token_budget
    assert len(client.calls) == 2  # the rejected max_tokens try, then the real one


# --- billing exhaustion: never retried ----------------------------------------
#
# The real response from an exhausted OpenAI account, verbatim. A 429 -- the
# same status as a per-minute rate limit, which is why it used to be retried.
QUOTA_MSG = (
    "Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to "
    "continue using the API at https://platform.openai.com/settings/organization/billing/.', "
    "'type': 'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}"
)
RATE_MSG = "Error code: 429 - {'error': {'message': 'Rate limit reached', 'code': 'rate_limit_exceeded'}}"


def _counting(exc_or_value_seq):
    calls = []

    def fn():
        calls.append(1)
        item = exc_or_value_seq[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    return fn, calls


def test_backoff_does_not_retry_quota_exhaustion(monkeypatch):
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: pytest.fail("must not back off on exhausted credit"))
    fn, calls = _counting([RuntimeError(QUOTA_MSG)])
    with pytest.raises(llm_providers.QuotaExhaustedError):
        llm_providers._with_backoff(fn)
    assert len(calls) == 1


def test_backoff_still_retries_a_real_rate_limit(monkeypatch):
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)
    fn, calls = _counting([RuntimeError(RATE_MSG), RuntimeError(RATE_MSG), "ok"])
    assert llm_providers._with_backoff(fn) == "ok"
    assert len(calls) == 3


def test_is_quota_exhausted_follows_the_cause_chain():
    try:
        try:
            raise RuntimeError(QUOTA_MSG)
        except RuntimeError as inner:
            raise ValueError("wrapped by some layer") from inner
    except ValueError as outer:
        assert llm_providers.is_quota_exhausted(outer)
    assert not llm_providers.is_quota_exhausted(RuntimeError(RATE_MSG))


def _response(status, text):
    import httpx2
    return httpx2.Response(status, text=text, request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"))


@pytest.mark.parametrize(
    "status,text,expected",
    [
        (429, QUOTA_MSG, False),   # exhausted credit: the SDK must NOT retry
        (429, RATE_MSG, True),     # real rate limit: SDK default, retried as before
        (500, "server error", True),
        (408, "timeout", True),
        (400, "bad request", False),
    ],
)
def test_sdk_retry_decision(status, text, expected):
    client = llm_providers._OpenAICompatible(api_key="test-key")
    assert client._should_retry(_response(status, text)) is expected


# --- usage metering ---------------------------------------------------------------
#
# Every request that comes back is recorded, billed-but-empty ones included, and
# nothing about what the provider layer returns or how many requests it makes changes.


def _usage(prompt, completion, reasoning=None, cached=None):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=None if reasoning is None else SimpleNamespace(reasoning_tokens=reasoning),
        prompt_tokens_details=None if cached is None else SimpleNamespace(cached_tokens=cached),
    )


def _metered(client, json_mode=False, provider="openai"):
    with usage.collect() as collector:
        text = llm_providers._generate_openai_compatible(
            client, "m", "sys", "user", 20, json_mode, provider=provider
        )
    return text, collector.calls


def test_a_successful_call_records_every_token_type():
    client = FakeClient([("SELECT 1", "stop", _usage(120, 40, reasoning=25, cached=64))])

    text, calls = _metered(client, provider="groq")

    assert text == "SELECT 1"
    assert len(client.calls) == 1
    (call,) = calls
    assert (call.provider, call.model) == ("groq", "m")
    assert (call.prompt_tokens, call.completion_tokens) == (120, 40)
    assert (call.reasoning_tokens, call.cached_prompt_tokens) == (25, 64)
    assert call.finish_reason == "stop"
    assert call.usage_known is True


def test_a_billed_empty_reply_is_counted_even_though_it_raises():
    client = FakeClient([("", "stop", _usage(50, 20, reasoning=20))])

    with usage.collect() as collector:
        with pytest.raises(RuntimeError, match="returned empty content"):
            llm_providers._generate_openai_compatible(client, "m", "sys", "user", 20, False)

    assert [(c.prompt_tokens, c.completion_tokens) for c in collector.calls] == [(50, 20)]


def test_the_headroom_retry_records_both_billed_requests():
    client = FakeClient(
        [("", "length", _usage(30, 20, reasoning=20)), ("LOGIC_ERROR", "stop", _usage(30, 60, reasoning=50))]
    )

    text, calls = _metered(client)

    assert text == "LOGIC_ERROR"
    assert len(client.calls) == 2
    assert [c.completion_tokens for c in calls] == [20, 60]
    assert [c.finish_reason for c in calls] == ["length", "stop"]


def test_a_rejected_request_has_no_usage_and_is_not_counted():
    # The max_tokens try is refused outright, so only the retry that works is billed.
    client = FakeClient([("SELECT 1", "stop", _usage(10, 5))], reject_max_tokens=True)

    text, calls = _metered(client)

    assert text == "SELECT 1"
    assert len(client.calls) == 2
    assert len(calls) == 1


def test_the_json_mode_fallback_counts_only_the_request_that_answered():
    client = FakeClient([("plain answer", "stop", _usage(10, 5))], reject_json_mode=True)

    text, calls = _metered(client, json_mode=True)

    assert text == "plain answer"
    assert len(calls) == 1


def test_a_reply_with_no_usage_is_one_call_with_unknown_tokens():
    client = FakeClient([("SELECT 1", "stop")])

    text, calls = _metered(client)

    assert text == "SELECT 1"
    assert len(client.calls) == 1
    assert [c.usage_known for c in calls] == [False]
    assert calls[0].prompt_tokens == calls[0].completion_tokens == 0


def test_usage_fields_a_provider_leaves_out_count_as_zero():
    client = FakeClient([("SELECT 1", "stop", _usage(70, 9))])  # no reasoning or cached detail blocks

    _, calls = _metered(client)

    assert (calls[0].reasoning_tokens, calls[0].cached_prompt_tokens) == (0, 0)
    assert calls[0].usage_known is True


def test_calls_outside_a_collector_behave_as_before():
    client = FakeClient([("SELECT 1", "stop", _usage(10, 5))])

    assert _gen(client) == "SELECT 1"
    assert len(client.calls) == 1


def test_generate_text_still_returns_a_plain_string_and_names_the_provider(monkeypatch):
    monkeypatch.setattr(llm_providers, "_openai_client", FakeClient([("SELECT 1", "stop", _usage(10, 5))]))
    monkeypatch.setattr(llm_providers, "_groq_client", FakeClient([("SELECT 2", "stop", _usage(20, 6))]))

    with usage.collect() as collector:
        first = llm_providers.generate_text("openai", "gpt-5-mini", "sys", "user")
        second = llm_providers.generate_text("groq", "openai/gpt-oss-20b", "sys", "user")

    assert (first, second) == ("SELECT 1", "SELECT 2")
    assert type(first) is str and type(second) is str
    assert [(c.provider, c.model) for c in collector.calls] == [
        ("openai", "gpt-5-mini"),
        ("groq", "openai/gpt-oss-20b"),
    ]


class FakeGemini:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_gemini_counts_thinking_as_output(monkeypatch):
    response = SimpleNamespace(
        text="SELECT 1",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=10,
            thoughts_token_count=30,
            cached_content_token_count=20,
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )
    gemini = FakeGemini(response)
    monkeypatch.setattr(llm_providers, "_gemini_client", gemini)

    with usage.collect() as collector:
        text = llm_providers.generate_text("gemini", "gemini-x", "sys", "user")

    assert text == "SELECT 1"
    assert len(gemini.calls) == 1
    (call,) = collector.calls
    assert (call.provider, call.model) == ("gemini", "gemini-x")
    assert (call.prompt_tokens, call.completion_tokens, call.reasoning_tokens) == (100, 40, 30)
    assert call.cached_prompt_tokens == 20
    assert call.finish_reason == "STOP"


def test_a_gemini_response_whose_text_raises_is_still_counted(monkeypatch):
    class Blocked:
        usage_metadata = SimpleNamespace(
            prompt_token_count=40, candidates_token_count=0, thoughts_token_count=None, cached_content_token_count=None
        )
        candidates = []

        @property
        def text(self):
            raise ValueError("response was blocked")

    monkeypatch.setattr(llm_providers, "_gemini_client", FakeGemini(Blocked()))

    with usage.collect() as collector:
        with pytest.raises(ValueError, match="blocked"):
            llm_providers.generate_text("gemini", "gemini-x", "sys", "user")

    assert [c.prompt_tokens for c in collector.calls] == [40]


def test_gemini_response_without_usage_metadata_is_unknown(monkeypatch):
    monkeypatch.setattr(llm_providers, "_gemini_client", FakeGemini(SimpleNamespace(text="SELECT 1")))

    with usage.collect() as collector:
        assert llm_providers.generate_text("gemini", "gemini-x", "sys", "user") == "SELECT 1"

    assert [c.usage_known for c in collector.calls] == [False]
