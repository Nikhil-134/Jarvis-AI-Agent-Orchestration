"""Self-registering provider registry for Jarvis LLM providers.

Usage::

    from llm.registry import register_provider, ProviderRegistry

    @register_provider("ollama")
    class OllamaProvider(BaseLLMProvider):
        ...
"""

import logging
from typing import Any

from llm.interfaces import ILLMProvider, IProviderRegistry

_logger = logging.getLogger(__name__)


class ProviderRegistry(IProviderRegistry):
    """Thread-safe registry of LLM provider classes."""

    def __init__(self) -> None:
        self._providers: dict[str, type[ILLMProvider]] = {}

    def register(self, name: str, provider_cls: type[ILLMProvider]) -> None:
        """Register a provider class under a case-insensitive name."""
        key = name.lower()
        if key in self._providers:
            _logger.warning("Overwriting registered provider '%s'", key)
        self._providers[key] = provider_cls
        _logger.debug("Registered LLM provider '%s' -> %s", key, provider_cls.__name__)

    def get(self, name: str) -> type[ILLMProvider]:
        """Return the provider class registered under *name*."""
        key = name.lower()
        if key not in self._providers:
            raise ValueError(f"Unknown LLM provider: {name}. Available: {list(self._providers)}")
        return self._providers[key]

    def all(self) -> dict[str, type[ILLMProvider]]:
        """Return all registered providers mapped by lower-case name."""
        return dict(self._providers)


# Module-level singleton so the decorator below works without manual wiring.
_registry: ProviderRegistry | None = None


def _ensure_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def register_provider(name: str = "") -> Any:
    """Decorator that registers an :class:`ILLMProvider` subclass.

    If *name* is empty the lower-case class name with ``provider``
    stripped (e.g. ``OpenAIProvider`` → ``openai``) is used.
    """
    def decorator(cls: type[ILLMProvider]) -> type[ILLMProvider]:
        registry = _ensure_registry()
        key = name.lower() if name else cls.__name__.lower().replace("provider", "").strip("_")
        registry.register(key, cls)
        return cls
    return decorator


def get_provider_registry() -> ProviderRegistry:
    """Return the global provider registry singleton."""
    return _ensure_registry()
