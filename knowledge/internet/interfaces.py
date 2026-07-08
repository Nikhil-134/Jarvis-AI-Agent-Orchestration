"""Contracts for the internet retrieval layer.

The layer is deliberately tiny and plugin-shaped: a provider is anything that
can turn a query string into zero-or-more :class:`RetrievalResult` snippets.
That keeps the service open for extension (add a provider) but closed for
modification (SOLID / OCP), and makes every provider trivially fakeable in
tests without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A single, self-contained fact snippet retrieved from a public source.

    It carries *only* what the LLM needs as context plus provenance for the
    user. It is intentionally immutable and free of any HTML/markup — providers
    return plain text only.
    """

    source: str          # provider name, e.g. "wikipedia" / "duckduckgo"
    title: str           # short label for the snippet
    snippet: str         # the plain-text fact(s)
    url: str = ""        # canonical public URL (provenance), may be empty
    score: float = 0.5   # provider-assigned confidence in [0, 1]
    metadata: dict = field(default_factory=dict)

    def is_useful(self) -> bool:
        """True if the snippet actually carries information worth injecting."""
        return bool(self.snippet and self.snippet.strip())


@runtime_checkable
class IRetrievalProvider(Protocol):
    """A pluggable public-information provider (DuckDuckGo, Wikipedia, ...).

    Implementations MUST:
      * never raise — return ``[]`` on any error/timeout (fail safe);
      * perform only JSON HTTP calls to whitelisted hosts (no scraping);
      * be cancellation-friendly (honour the ambient asyncio timeout).
    """

    @property
    def name(self) -> str:
        """Stable provider identifier used in results and logs."""
        ...

    async def retrieve(self, query: str, *, max_results: int = 3) -> list[RetrievalResult]:
        """Return up to *max_results* fact snippets for *query* (never raises)."""
        ...
