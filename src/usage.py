"""
Per-call LLM usage metering.

The provider layer calls record() after every request that comes back; whoever
wants the numbers wraps the work in `with collect() as c:` and reads c.calls
afterwards. With no collector active record() does nothing, so callers outside
a metered scope (the explain tool, scripts, tests that fake the provider) are
unaffected.

The collector lives in a ContextVar, so concurrent API requests, each on its own
thread, cannot see each other's calls. The graph opens one collector per node
run, which also means no state carries over from one question to the next.

Stdlib only: src/graph.py imports this at load time, and the Lambda image
installs only a slim dependency group.
"""

from __future__ import annotations

import contextvars
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMCall:
    provider: str
    model: str
    prompt_tokens: int
    # Includes reasoning tokens, which every provider bills as output.
    completion_tokens: int
    # The part of completion_tokens spent reasoning, where the provider says.
    reasoning_tokens: int
    # The part of prompt_tokens served from the provider's prompt cache.
    cached_prompt_tokens: int
    finish_reason: str | None
    # False when the response carried no usage block at all. The call still
    # happened and was still billed; its token counts are just unknown, so a
    # cost must not be computed from the zeros.
    usage_known: bool


@dataclass
class Collector:
    calls: list[LLMCall] = field(default_factory=list)

    def totals(self) -> dict[str, int]:
        return {
            "llm_calls": len(self.calls),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "cached_prompt_tokens": sum(c.cached_prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "reasoning_tokens": sum(c.reasoning_tokens for c in self.calls),
        }


_active: contextvars.ContextVar[Collector | None] = contextvars.ContextVar(
    "llm_usage_collector", default=None
)


@contextmanager
def collect() -> Generator[Collector, None, None]:
    """
    Meter every LLM call made inside the block. A nested block shadows the outer
    one: calls inside it count there only, not in both.
    """
    collector = Collector()
    token = _active.set(collector)
    try:
        yield collector
    finally:
        _active.reset(token)


def _count(value: int | None) -> int:
    return int(value) if value else 0


def record(
    provider: str,
    model: str,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    cached_prompt_tokens: int | None = None,
    finish_reason: str | None = None,
) -> None:
    collector = _active.get()
    if collector is None:
        return
    collector.calls.append(
        LLMCall(
            provider=provider,
            model=model,
            prompt_tokens=_count(prompt_tokens),
            completion_tokens=_count(completion_tokens),
            reasoning_tokens=_count(reasoning_tokens),
            cached_prompt_tokens=_count(cached_prompt_tokens),
            finish_reason=finish_reason,
            usage_known=prompt_tokens is not None or completion_tokens is not None,
        )
    )
