"""Tests for LLM provider factory switching."""

from config.settings import Settings
from llm import OllamaProvider, OpenAIProvider, build_llm_provider


def test_factory_builds_openai_provider() -> None:
    """Factory should create OpenAIProvider when configured."""
    provider = build_llm_provider(
        Settings(llm_provider="openai", llm_model="gpt-test", openai_api_key="key")
    )

    assert isinstance(provider, OpenAIProvider)


def test_factory_builds_ollama_provider() -> None:
    """Factory should create OllamaProvider when configured."""
    provider = build_llm_provider(Settings(llm_provider="ollama", llm_model="llama-test"))

    assert isinstance(provider, OllamaProvider)
