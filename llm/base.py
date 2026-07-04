"""Base abstractions for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import sleep
from typing import TypeVar

from llm.errors import LLMError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Runtime settings shared by LLM providers."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25


class BaseLLMProvider(ABC):
    """Interface all LLM providers must implement."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the provider with shared LLM configuration."""
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a complete response for a prompt."""

    @abstractmethod
    def stream(self, prompt: str, system_prompt: str | None = None) -> Iterable[str]:
        """Stream response chunks for a prompt."""

    def _with_retries(self, operation: Callable[[], T]) -> T:
        """Execute an operation with retry and backoff handling."""
        last_error: LLMError | None = None
        attempts = self.config.max_retries + 1

        for attempt in range(attempts):
            try:
                return operation()
            except LLMError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                sleep(self.config.retry_backoff_seconds * (2**attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM retry loop exited without a result or error.")
