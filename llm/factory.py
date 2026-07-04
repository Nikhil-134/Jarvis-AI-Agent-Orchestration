"""Factory for configured LLM providers."""

from config.settings import Settings
from llm.base import LLMConfig
from llm.ollama_provider import OllamaProvider
from llm.openai_provider import OpenAIProvider


def build_llm_provider(settings: Settings) -> OpenAIProvider | OllamaProvider:
    """Create an LLM provider from application settings."""
    config = LLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
    )

    if settings.llm_provider == "openai":
        return OpenAIProvider(config)
    if settings.llm_provider == "ollama":
        return OllamaProvider(config)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
