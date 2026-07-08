"""Internet retrieval layer (roadmap #9).

A lightweight, fail-safe, context-only retrieval subsystem. It supplies fresh
public facts (weather/news/current events/recent releases) to the local model
as *temporary context* — it never reasons, never stores durably, and only
activates when the freshness router says local sources cannot answer.

Public surface::

    from knowledge.internet import (
        InternetKnowledgeService, build_internet_service,
        IRetrievalProvider, RetrievalResult, needs_internet,
    )
"""

from __future__ import annotations

from knowledge.internet.cache import TTLCache
from knowledge.internet.http_client import (
    DEFAULT_ALLOWED_HOSTS,
    HttpClientError,
    SafeHttpClient,
)
from knowledge.internet.interfaces import IRetrievalProvider, RetrievalResult
from knowledge.internet.providers import DuckDuckGoProvider, WikipediaProvider
from knowledge.internet.router import is_memory_query, needs_internet
from knowledge.internet.service import InternetKnowledgeService


def build_internet_service(
    *,
    enabled: bool = True,
    timeout: float = 6.0,
    overall_timeout: float = 8.0,
    cache_ttl_seconds: float = 300.0,
    min_interval_seconds: float = 1.0,
) -> InternetKnowledgeService | None:
    """Assemble the default DuckDuckGo + Wikipedia service, or None if disabled.

    Wiring lives here (composition root helper) so callers depend only on the
    service, not on the concrete providers/HTTP client (Dependency Inversion).
    """
    if not enabled:
        return None
    http = SafeHttpClient(timeout=timeout)
    providers: list[IRetrievalProvider] = [
        DuckDuckGoProvider(http),
        WikipediaProvider(http),
    ]
    cache: TTLCache = TTLCache(ttl_seconds=cache_ttl_seconds)
    return InternetKnowledgeService(
        providers,
        cache=cache,
        overall_timeout=overall_timeout,
        min_interval_seconds=min_interval_seconds,
    )


__all__ = [
    "InternetKnowledgeService",
    "build_internet_service",
    "IRetrievalProvider",
    "RetrievalResult",
    "DuckDuckGoProvider",
    "WikipediaProvider",
    "SafeHttpClient",
    "HttpClientError",
    "DEFAULT_ALLOWED_HOSTS",
    "TTLCache",
    "needs_internet",
    "is_memory_query",
]
