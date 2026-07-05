"""Factory for configured LLM providers using the provider registry.

Importing provider modules here ensures their ``@register_provider``
decorators fire before any registry lookup.
"""

# Eager-import provider modules so the decorators register them.
import llm.ollama_provider  # noqa: F401
import llm.openai_provider  # noqa: F401

from config.settings import Settings
from llm.base import LLMConfig
from llm.registry import get_provider_registry


def build_llm_provider(settings: Settings) -> object:
    """Create an LLM provider from application settings.

    Uses the :class:`ProviderRegistry` to look up the provider class
    by name, so adding a new provider does not require modifying this
    factory — the provider just needs the ``@register_provider``
    decorator.
    """
    config = LLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
    )

    registry = get_provider_registry()
    provider_cls = registry.get(settings.llm_provider)
    return provider_cls(config)
