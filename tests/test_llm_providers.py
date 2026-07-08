"""Tests for LLM provider behavior without network calls."""

from collections.abc import AsyncIterable

import pytest

from llm import LLMConfig, LLMProviderError, OllamaProvider, OpenAIProvider
from llm.base import BaseLLMProvider


class FlakyProvider(BaseLLMProvider):
    """Provider that fails once before succeeding."""

    def __init__(self) -> None:
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
        return "flaky"

    async def _generate_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> LLMResponse:
        from llm.interfaces import LLMResponse
        self.calls += 1
        if self.calls == 1:
            raise LLMProviderError("temporary")
        return LLMResponse(content="ok")

    async def _stream_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> AsyncIterable[str]:
        return
        yield


@pytest.mark.asyncio
async def test_base_provider_retry_logic_retries_llm_errors() -> None:
    provider = FlakyProvider()

    result = await provider.generate("hello")

    assert result.content == "ok"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_openai_provider_parses_generate_response(monkeypatch) -> None:
    provider = OpenAIProvider(
        LLMConfig(provider="openai", model="test", api_key="key", retry_backoff_seconds=0)
    )

    async def mock_post_json(url: str, payload: dict, headers: dict) -> dict:
        return {"choices": [{"message": {"content": "hello"}}]}

    monkeypatch.setattr(provider, "_http_post_json", mock_post_json)

    result = await provider.generate("prompt")
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_ollama_provider_parses_generate_response(monkeypatch) -> None:
    provider = OllamaProvider(
        LLMConfig(provider="ollama", model="test", retry_backoff_seconds=0)
    )

    async def mock_post_json(url: str, payload: dict, headers: dict) -> dict:
        return {"message": {"content": "hello"}}

    monkeypatch.setattr(provider, "_http_post_json", mock_post_json)

    result = await provider.generate("prompt")
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_openai_provider_parses_stream_response(monkeypatch) -> None:
    provider = OpenAIProvider(
        LLMConfig(provider="openai", model="test", api_key="key", retry_backoff_seconds=0)
    )

    async def mock_stream_sse(payload: dict) -> AsyncIterable[dict | str]:
        yield {"choices": [{"delta": {"content": "he"}}]}
        yield {"choices": [{"delta": {"content": "llo"}}]}
        yield "[DONE]"

    monkeypatch.setattr(provider, "_stream_sse", mock_stream_sse)

    chunks = [chunk async for chunk in provider.stream("prompt")]
    assert chunks == ["he", "llo"]


@pytest.mark.asyncio
async def test_ollama_provider_parses_generate_response(monkeypatch) -> None:
    provider = OllamaProvider(
        LLMConfig(provider="ollama", model="test", retry_backoff_seconds=0)
    )

    async def mock_post_json(url: str, payload: dict, headers: dict) -> dict:
        return {"message": {"content": "hello"}}

    monkeypatch.setattr(provider, "_http_post_json", mock_post_json)

    result = await provider.generate("prompt")
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_ollama_provider_parses_stream_response(monkeypatch) -> None:
    provider = OllamaProvider(
        LLMConfig(provider="ollama", model="test", retry_backoff_seconds=0)
    )

    async def mock_stream_jsonl(payload: dict) -> AsyncIterable[dict]:
        yield {"message": {"content": "he"}, "done": False}
        yield {"message": {"content": "llo"}, "done": True}

    monkeypatch.setattr(provider, "_stream_jsonl", mock_stream_jsonl)

    chunks = [chunk async for chunk in provider.stream("prompt")]
    assert chunks == ["he", "llo"]


@pytest.mark.asyncio
async def test_openai_provider_requires_api_key() -> None:
    provider = OpenAIProvider(
        LLMConfig(provider="openai", model="test", max_retries=0)
    )

    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
        await provider.generate("prompt")
