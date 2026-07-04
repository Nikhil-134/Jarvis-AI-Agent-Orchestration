"""Tests for LLM provider behavior without network calls."""

from collections.abc import Iterable

import pytest

from llm import LLMConfig, LLMProviderError, OllamaProvider, OpenAIProvider
from llm.base import BaseLLMProvider


class FlakyProvider(BaseLLMProvider):
    """Provider that fails once before succeeding."""

    def __init__(self) -> None:
        """Initialize retry test provider."""
        super().__init__(
            LLMConfig(
                provider="flaky",
                model="test",
                max_retries=1,
                retry_backoff_seconds=0,
            )
        )
        self.calls = 0

    @property
    def name(self) -> str:
        """Return provider name."""
        return "flaky"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate using retry helper."""
        return self._with_retries(self._operation)

    def stream(self, prompt: str, system_prompt: str | None = None) -> Iterable[str]:
        """Return no stream chunks."""
        return ()

    def _operation(self) -> str:
        """Fail once, then succeed."""
        self.calls += 1
        if self.calls == 1:
            raise LLMProviderError("temporary")
        return "ok"


def test_base_provider_retry_logic_retries_llm_errors() -> None:
    """BaseLLMProvider should retry provider errors."""
    provider = FlakyProvider()

    assert provider.generate("hello") == "ok"
    assert provider.calls == 2


def test_openai_provider_parses_generate_response(monkeypatch) -> None:
    """OpenAIProvider should extract message content."""
    provider = OpenAIProvider(
        LLMConfig(provider="openai", model="test", api_key="key", retry_backoff_seconds=0)
    )
    monkeypatch.setattr(
        provider,
        "_post_json",
        lambda payload: {"choices": [{"message": {"content": "hello"}}]},
    )

    assert provider.generate("prompt") == "hello"


def test_openai_provider_parses_stream_response(monkeypatch) -> None:
    """OpenAIProvider should extract streamed delta content."""
    provider = OpenAIProvider(
        LLMConfig(provider="openai", model="test", api_key="key", retry_backoff_seconds=0)
    )
    monkeypatch.setattr(
        provider,
        "_stream_json_events",
        lambda payload: iter(
            [
                {"choices": [{"delta": {"content": "he"}}]},
                {"choices": [{"delta": {"content": "llo"}}]},
                "[DONE]",
            ]
        ),
    )

    assert list(provider.stream("prompt")) == ["he", "llo"]


def test_ollama_provider_parses_generate_response(monkeypatch) -> None:
    """OllamaProvider should extract message content."""
    provider = OllamaProvider(LLMConfig(provider="ollama", model="test", retry_backoff_seconds=0))
    monkeypatch.setattr(provider, "_post_json", lambda payload: {"message": {"content": "hello"}})

    assert provider.generate("prompt") == "hello"


def test_ollama_provider_parses_stream_response(monkeypatch) -> None:
    """OllamaProvider should extract streamed message content."""
    provider = OllamaProvider(LLMConfig(provider="ollama", model="test", retry_backoff_seconds=0))
    monkeypatch.setattr(
        provider,
        "_stream_json_lines",
        lambda payload: iter(
            [
                {"message": {"content": "he"}, "done": False},
                {"message": {"content": "llo"}, "done": True},
            ]
        ),
    )

    assert list(provider.stream("prompt")) == ["he", "llo"]


def test_openai_provider_requires_api_key() -> None:
    """OpenAIProvider should fail clearly without an API key."""
    provider = OpenAIProvider(LLMConfig(provider="openai", model="test", max_retries=0))

    with pytest.raises(LLMProviderError):
        provider.generate("prompt")
