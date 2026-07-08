"""LLM Guard — wraps all LLM calls with retry, backoff, timeout, model fallback.

Ensures the application never crashes due to an LLM timeout or provider error.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any, TypeVar

from llm import BaseLLMProvider, LLMError, LLMTimeoutError
from llm.interfaces import LLMResponse

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class AllProvidersExhaustedError(RuntimeError):
    """All LLM providers/models were tried and all failed."""


@dataclass
class FallbackModel:
    """A fallback model configuration to try when the primary fails."""

    provider: str
    model: str
    base_url: str | None = None
    timeout_seconds: float = 60.0


@dataclass
class GuardConfig:
    """Configuration for the LLM Guard."""

    primary_timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    fallback_context_ratio: float = 0.5
    max_retry_context_ratio: float = 0.75
    graceful_message: str = (
        "I'm having trouble contacting the language model. "
        "Please check that Ollama (or your LLM provider) is running and try again."
    )


class RetryState:
    """Tracks retry state for diagnostics."""

    def __init__(self) -> None:
        self.attempts: int = 0
        self.last_error: str = ""
        self.fallback_used: bool = False
        self.total_time_ms: float = 0.0


class LLMGuard:
    """Wraps an LLM provider with retry, timeout, and fallback protection.

    Usage::

        guard = LLMGuard(primary_provider, config=GuardConfig())
        response = await guard.generate("Hello")
    """

    def __init__(
        self,
        primary_provider: BaseLLMProvider | None,
        fallback_providers: list[BaseLLMProvider] | None = None,
        config: GuardConfig | None = None,
    ) -> None:
        self._primary = primary_provider
        self._fallbacks = fallback_providers or []
        self._config = config or GuardConfig()
        self._retry_state = RetryState()

    @property
    def retry_state(self) -> RetryState:
        return self._retry_state

    @property
    def is_available(self) -> bool:
        return self._primary is not None

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Generate a response with retry + fallback. Never raises."""
        if self._primary is None:
            return self._graceful_response("No LLM provider configured.")

        providers_to_try: list[tuple[str, BaseLLMProvider]] = [("primary", self._primary)]
        for i, fb in enumerate(self._fallbacks):
            providers_to_try.append((f"fallback_{i}", fb))

        last_error: Exception | None = None
        start = asyncio.get_event_loop().time()

        for label, provider in providers_to_try:
            adjusted_prompt, adjusted_system = self._adjust_context(
                prompt, system_prompt, label != "primary",
            )

            for attempt in range(self._config.max_retries + 1):
                self._retry_state.attempts += 1
                try:
                    response = await provider.generate(
                        adjusted_prompt,
                        system_prompt=adjusted_system,
                        tools=tools,
                    )
                    elapsed = (asyncio.get_event_loop().time() - start) * 1000
                    self._retry_state.total_time_ms = elapsed
                    if label != "primary":
                        self._retry_state.fallback_used = True
                    return response
                except LLMTimeoutError as exc:
                    last_error = exc
                    self._retry_state.last_error = str(exc)
                    _logger.warning(
                        "LLM timeout on %s (attempt %d/%d): %s",
                        label, attempt + 1, self._config.max_retries + 1, exc,
                    )
                    if attempt < self._config.max_retries:
                        wait = self._config.retry_backoff_seconds * (2**attempt)
                        await asyncio.sleep(wait)
                except LLMError as exc:
                    last_error = exc
                    self._retry_state.last_error = str(exc)
                    _logger.error("LLM error on %s: %s", label, exc)
                    break

        _logger.error("All LLM providers exhausted. Last error: %s", last_error)
        return self._graceful_response(str(last_error) if last_error else None)

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
    ) -> AsyncIterable[str]:
        """Stream a response with retry + fallback."""
        if self._primary is None:
            yield self._config.graceful_message
            return

        response = await self.generate(prompt, system_prompt, tools)
        if response.content:
            yield response.content

    def _adjust_context(
        self,
        prompt: str,
        system_prompt: str | None,
        is_fallback: bool,
    ) -> tuple[str, str | None]:
        """Reduce context size on fallback or retry to improve reliability."""
        if not is_fallback:
            return prompt, system_prompt

        max_chars = int(len(prompt) * self._config.fallback_context_ratio)
        truncated = prompt[:max_chars]
        _logger.info("Context truncated to %d chars for fallback provider", max_chars)
        return truncated, system_prompt

    def _graceful_response(self, error: str | None = None) -> LLMResponse:
        """Return a graceful error response instead of crashing."""
        msg = self._config.graceful_message
        if error:
            msg = f"{msg}\n\nDetails: {error}"
        return LLMResponse(content=msg)
