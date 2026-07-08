"""Token budget estimation and context window management.

Provides a zero-dependency token estimator based on character counting,
plus a :class:`TokenBudget` class that manages the LLM context window
budget across system prompt, conversation history, memory context, and
the user's goal.
"""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)

# Approximate characters per token for estimation (no external tokenizer).
# English prose: ~4 chars/token, code: ~3.5 chars/token.
# Using a conservative 3.5 to avoid exceeding the LLM's context window.
_CHARS_PER_TOKEN: float = 3.5

# Budget reserved for the LLM's response tokens (never allocated to the prompt).
_RESPONSE_RESERVE_TOKENS: int = 1024

# Minimum tokens we must keep free after allocating for prompt parts.
_MIN_FREE_TOKENS: int = 256


def estimate_tokens(text: str, chars_per_token: float = _CHARS_PER_TOKEN) -> int:
    """Estimate the number of tokens *text* would consume.

    Uses a simple character-count heuristic::

        estimate = len(text) / chars_per_token

    When *chars_per_token* is ``0`` every text is estimated as 0 tokens
    (useful for disabling budget enforcement).
    """
    if chars_per_token <= 0 or not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


class TokenBudget:
    """Manages the LLM context window budget across prompt components.

    Usage::

        budget = TokenBudget(max_context_tokens=8192)
        budget.allocate_system("You are a helpful assistant.")
        budget.allocate_history(history_text)
        budget.allocate_memory(memory_context)
        goal_budget = budget.remaining  # tokens available for the user's goal
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        chars_per_token: float = _CHARS_PER_TOKEN,
        reserve_tokens: int = _RESPONSE_RESERVE_TOKENS,
    ) -> None:
        self._max = max(1, max_context_tokens)
        self._cpt = chars_per_token
        self._used = reserve_tokens  # reserve space for the response
        self._reserve = reserve_tokens

    @property
    def max_tokens(self) -> int:
        """The configured context window size."""
        return self._max

    @property
    def used(self) -> int:
        """Tokens consumed so far by allocated components."""
        return self._used

    @property
    def remaining(self) -> int:
        """Tokens available for additional components (e.g. the user's goal).

        Returns ``0`` when the budget is exhausted.
        """
        return max(0, self._max - self._used)

    @property
    def is_over_budget(self) -> bool:
        """``True`` when allocated components already exceed the budget."""
        return self._used >= self._max

    def allocate_system(self, system_prompt: str | None) -> int:
        """Account for the system prompt tokens."""
        tokens = estimate_tokens(system_prompt or "", self._cpt)
        self._used += tokens
        _logger.debug("Budget: +%d for system prompt (total used=%d)", tokens, self._used)
        return tokens

    def allocate_history(self, history_text: str) -> int:
        """Account for conversation history tokens."""
        tokens = estimate_tokens(history_text, self._cpt)
        self._used += tokens
        _logger.debug("Budget: +%d for history (total used=%d)", tokens, self._used)
        return tokens

    def allocate_memory(self, memory_context: str) -> int:
        """Account for memory context tokens."""
        tokens = estimate_tokens(memory_context, self._cpt)
        self._used += tokens
        _logger.debug("Budget: +%d for memory (total used=%d)", tokens, self._used)
        return tokens

    def can_fit(self, text: str) -> bool:
        """Predicate: would *text* fit within the remaining budget?"""
        needed = estimate_tokens(text, self._cpt)
        return (self._used + needed + _MIN_FREE_TOKENS) <= self._max

    def oversize_by(self, text: str) -> int:
        """Return how many tokens *text* exceeds the remaining budget by.

        Returns ``0`` if *text* fits within the budget.
        """
        needed = estimate_tokens(text, self._cpt)
        available = self.remaining
        if needed <= available:
            return 0
        return needed - available

    def snapshot(self) -> dict[str, Any]:
        """Return current budget state for logging / diagnostics."""
        return {
            "max_tokens": self._max,
            "used": self._used,
            "remaining": self.remaining,
            "reserved_for_response": self._reserve,
            "is_over_budget": self.is_over_budget,
        }
