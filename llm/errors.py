"""LLM provider exception types."""


class LLMError(Exception):
    """Base exception for LLM provider failures."""


class LLMProviderError(LLMError):
    """Raised when a provider returns an invalid or failed response."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM provider request times out."""
