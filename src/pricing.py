"""
Estimated LLM cost from recorded usage.

Prices are copied from the providers' own pages, each with its source and the
date it was read. They are never guessed: a (provider, model) pair that isn't
listed here has no price, and a cost that can't be computed is None rather than
zero, so "free" and "unknown" can never be mistaken for each other downstream.

Stdlib only; see src/usage.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.usage import LLMCall


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    # None when the provider lists no cached rate; cached tokens then bill as input.
    cached_input_per_mtok: float | None
    source_url: str
    as_of: str


# Keyed by the exact (provider, model) strings passed to generate_text.
#
# Deliberately absent, because their providers publish no price:
#   groq / llama-3.3-70b-versatile and groq / llama-3.1-8b-instant are listed as
#   "ContactSales" (Enterprise). (src/config.py and the README note that Groq
#   retired llama-3.1-8b-instant.) Gemini models were not checked.
PRICES: dict[tuple[str, str], Price] = {
    # OpenAI's model page marks gpt-5-mini "Deprecated" and prices the
    # gpt-5-mini-2025-08-07 snapshot.
    ("openai", "gpt-5-mini"): Price(
        input_per_mtok=0.25,
        output_per_mtok=2.00,
        cached_input_per_mtok=0.025,
        source_url="https://developers.openai.com/api/docs/models/gpt-5-mini",
        as_of="2026-10-25",
    ),
    # Input and output also appear on https://console.groq.com/docs/models;
    # the cached rate is only on the model's own page.
    ("groq", "openai/gpt-oss-20b"): Price(
        input_per_mtok=0.075,
        output_per_mtok=0.30,
        cached_input_per_mtok=0.037,
        source_url="https://console.groq.com/docs/model/openai/gpt-oss-20b",
        as_of="2026-10-25",
    ),
}

# The latest date any price above was read; recorded with each benchmark run so
# a cost figure can be traced to the prices it used.
PRICING_AS_OF = max(price.as_of for price in PRICES.values())

TOKENS_PER_MTOK = 1_000_000


def call_cost_usd(call: LLMCall) -> float | None:
    price = PRICES.get((call.provider, call.model))
    if price is None or not call.usage_known:
        return None
    cached = min(call.cached_prompt_tokens, call.prompt_tokens)
    cached_rate = price.cached_input_per_mtok
    if cached_rate is None:
        cached_rate = price.input_per_mtok
    return (
        (call.prompt_tokens - cached) * price.input_per_mtok
        + cached * cached_rate
        + call.completion_tokens * price.output_per_mtok
    ) / TOKENS_PER_MTOK


def calls_cost_usd(calls: list[LLMCall]) -> float | None:
    """
    Total cost of the calls: 0.0 when there were none, None when any one of them
    can't be priced -- a partial sum would read as the whole cost.
    """
    total = 0.0
    for call in calls:
        cost = call_cost_usd(call)
        if cost is None:
            return None
        total += cost
    return total
