"""
Usage metering and cost estimation. Free and offline.
"""

import ast
import re
import sys
import threading
from pathlib import Path

import pytest

from src import pricing, usage
from src.pricing import Price
from src.usage import LLMCall

SRC = Path(__file__).resolve().parents[1] / "src"


def _call(provider="openai", model="gpt-5-mini", **overrides):
    fields = dict(
        provider=provider,
        model=model,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        cached_prompt_tokens=0,
        finish_reason="stop",
        usage_known=True,
    )
    fields.update(overrides)
    return LLMCall(**fields)


# --- collector ---------------------------------------------------------------


def test_record_outside_a_collector_does_nothing():
    usage.record("openai", "gpt-5-mini", prompt_tokens=5, completion_tokens=5)

    with usage.collect() as c:
        pass

    assert c.calls == []


def test_a_collector_gets_only_calls_made_inside_its_block():
    with usage.collect() as first:
        usage.record("openai", "gpt-5-mini", prompt_tokens=10, completion_tokens=2)
    usage.record("openai", "gpt-5-mini", prompt_tokens=99, completion_tokens=99)
    with usage.collect() as second:
        usage.record("groq", "openai/gpt-oss-20b", prompt_tokens=1, completion_tokens=1)

    assert [c.prompt_tokens for c in first.calls] == [10]
    assert [c.provider for c in second.calls] == ["groq"]


def test_a_nested_collector_shadows_the_outer_one():
    with usage.collect() as outer:
        usage.record("openai", "gpt-5-mini", prompt_tokens=1, completion_tokens=1)
        with usage.collect() as inner:
            usage.record("openai", "gpt-5-mini", prompt_tokens=2, completion_tokens=2)
        usage.record("openai", "gpt-5-mini", prompt_tokens=3, completion_tokens=3)

    assert [c.prompt_tokens for c in inner.calls] == [2]
    assert [c.prompt_tokens for c in outer.calls] == [1, 3]


def test_collectors_on_different_threads_do_not_mix():
    results = {}
    barrier = threading.Barrier(2)

    def worker(name, tokens):
        with usage.collect() as c:
            barrier.wait()
            usage.record("openai", "gpt-5-mini", prompt_tokens=tokens, completion_tokens=0)
            barrier.wait()
        results[name] = [call.prompt_tokens for call in c.calls]

    threads = [
        threading.Thread(target=worker, args=("a", 7)),
        threading.Thread(target=worker, args=("b", 11)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == {"a": [7], "b": [11]}


def test_totals_sum_every_token_type():
    with usage.collect() as c:
        usage.record(
            "openai", "gpt-5-mini",
            prompt_tokens=100, completion_tokens=50, reasoning_tokens=30, cached_prompt_tokens=40,
        )
        usage.record("openai", "gpt-5-mini", prompt_tokens=10, completion_tokens=5)

    assert c.totals() == {
        "llm_calls": 2,
        "prompt_tokens": 110,
        "cached_prompt_tokens": 40,
        "completion_tokens": 55,
        "reasoning_tokens": 30,
    }


def test_missing_token_fields_count_as_zero():
    with usage.collect() as c:
        usage.record("openai", "gpt-5-mini", prompt_tokens=8, completion_tokens=None, reasoning_tokens=None)

    call = c.calls[0]
    assert (call.prompt_tokens, call.completion_tokens, call.reasoning_tokens) == (8, 0, 0)
    assert call.usage_known is True


def test_a_response_with_no_usage_is_still_a_call_but_unknown():
    with usage.collect() as c:
        usage.record("groq", "openai/gpt-oss-20b")

    assert c.totals()["llm_calls"] == 1
    assert c.calls[0].usage_known is False
    assert pricing.call_cost_usd(c.calls[0]) is None


def test_a_collector_is_reset_even_if_the_block_raises():
    with pytest.raises(RuntimeError):
        with usage.collect():
            raise RuntimeError("node failed")

    usage.record("openai", "gpt-5-mini", prompt_tokens=1, completion_tokens=1)
    with usage.collect() as c:
        pass
    assert c.calls == []


# --- pricing -----------------------------------------------------------------


def test_cost_splits_cached_prompt_tokens_from_the_rest():
    # 800k uncached input + 200k cached input + 100k output, at $0.25 / $0.025 / $2.00 per 1M.
    call = _call(prompt_tokens=1_000_000, cached_prompt_tokens=200_000, completion_tokens=100_000)

    assert pricing.call_cost_usd(call) == pytest.approx(0.8 * 0.25 + 0.2 * 0.025 + 0.1 * 2.00)


def test_reasoning_tokens_are_not_billed_twice():
    # completion_tokens already includes reasoning; the subset must not add cost.
    plain = _call(prompt_tokens=1000, completion_tokens=500)
    reasoning = _call(prompt_tokens=1000, completion_tokens=500, reasoning_tokens=400)

    assert pricing.call_cost_usd(reasoning) == pricing.call_cost_usd(plain)


def test_cached_tokens_bill_as_input_when_no_cached_rate_is_listed(monkeypatch):
    monkeypatch.setitem(
        pricing.PRICES,
        ("test", "no-cache-rate"),
        Price(1.0, 2.0, None, "https://example.com", "2026-01-01"),
    )
    call = _call("test", "no-cache-rate", prompt_tokens=1_000_000, cached_prompt_tokens=500_000)

    assert pricing.call_cost_usd(call) == pytest.approx(1.0)


def test_cached_tokens_beyond_the_prompt_are_clamped():
    call = _call(prompt_tokens=1_000_000, cached_prompt_tokens=5_000_000)

    assert pricing.call_cost_usd(call) == pytest.approx(1.0 * 0.025)


def test_unlisted_models_have_no_cost():
    assert pricing.call_cost_usd(_call("groq", "llama-3.3-70b-versatile", prompt_tokens=10)) is None
    assert pricing.call_cost_usd(_call("groq", "llama-3.1-8b-instant", prompt_tokens=10)) is None
    assert pricing.call_cost_usd(_call("somebody", "gpt-5-mini", prompt_tokens=10)) is None


def test_no_calls_cost_exactly_zero():
    assert pricing.calls_cost_usd([]) == 0.0


def test_one_unpriced_call_makes_the_whole_cost_unknown():
    priced = _call(prompt_tokens=1_000_000)
    unpriced = _call("groq", "llama-3.3-70b-versatile", prompt_tokens=1_000_000)

    assert pricing.calls_cost_usd([priced]) == pytest.approx(0.25)
    assert pricing.calls_cost_usd([priced, unpriced]) is None


def test_every_price_names_a_source_and_a_date():
    assert pricing.PRICES
    for (provider, model), price in pricing.PRICES.items():
        assert provider and model
        assert price.source_url.startswith("https://")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", price.as_of)
        assert price.input_per_mtok > 0 and price.output_per_mtok > 0


def test_pricing_as_of_is_the_latest_price_date():
    assert pricing.PRICING_AS_OF == max(p.as_of for p in pricing.PRICES.values())


# --- import weight -----------------------------------------------------------


@pytest.mark.parametrize("module", ["usage.py", "pricing.py"])
def test_modules_import_only_stdlib_and_each_other(module):
    # src/graph.py imports these at load time, and the Lambda image carries only a slim
    # dependency group, so a third-party import here would break the handler on cold start.
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])

    assert imported - set(sys.stdlib_module_names) - {"src"} == set()
