"""Concrete retrieval providers — DuckDuckGo and Wikipedia.

Both hit **public, key-free JSON APIs only** through the shared
:class:`SafeHttpClient`. Neither scrapes HTML, follows arbitrary links, or
contacts any host outside the whitelist. Each provider is defensive: any
error, timeout, or unexpected payload yields ``[]`` rather than an exception,
so a flaky network never breaks a conversation (fail safe).

The response shapes are extracted conservatively — we read only the specific
plain-text fields we understand and cap their length. We never echo raw markup.
"""

from __future__ import annotations

import logging

from knowledge.internet.http_client import HttpClientError, SafeHttpClient
from knowledge.internet.interfaces import RetrievalResult

_logger = logging.getLogger(__name__)

_MAX_SNIPPET_CHARS = 1200


def _clip(text: str, limit: int = _MAX_SNIPPET_CHARS) -> str:
    text = (text or "").strip()
    return text[:limit]


class DuckDuckGoProvider:
    """DuckDuckGo Instant Answer API (https://api.duckduckgo.com).

    Good for definitions, entities, and "instant answers". Returns the
    ``Abstract``/``Answer``/``Definition`` plus a few related topics.
    """

    _ENDPOINT = "https://api.duckduckgo.com/"

    def __init__(self, http: SafeHttpClient) -> None:
        self._http = http

    @property
    def name(self) -> str:
        return "duckduckgo"

    async def retrieve(self, query: str, *, max_results: int = 3) -> list[RetrievalResult]:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "t": "jarvis-local",
        }
        try:
            data = await self._http.get_json(self._ENDPOINT, params=params)
        except HttpClientError as exc:
            _logger.debug("DuckDuckGo retrieval failed: %s", exc)
            return []
        except Exception:  # noqa: BLE001 - never propagate
            _logger.debug("DuckDuckGo retrieval raised unexpectedly", exc_info=True)
            return []

        results: list[RetrievalResult] = []

        # Primary instant answer (Abstract) or a direct Answer/Definition.
        heading = str(data.get("Heading") or query).strip()
        primary = (
            str(data.get("AbstractText") or "").strip()
            or str(data.get("Answer") or "").strip()
            or str(data.get("Definition") or "").strip()
        )
        if primary:
            results.append(RetrievalResult(
                source=self.name,
                title=heading or query,
                snippet=_clip(primary),
                url=str(data.get("AbstractURL") or data.get("DefinitionURL") or ""),
                score=0.8,
            ))

        # A few related topics for breadth (each is a {Text, FirstURL}).
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if not isinstance(topic, dict):
                continue
            text = str(topic.get("Text") or "").strip()
            if not text:
                continue
            results.append(RetrievalResult(
                source=self.name,
                title=text[:60],
                snippet=_clip(text),
                url=str(topic.get("FirstURL") or ""),
                score=0.5,
            ))

        return results[:max_results]


class WikipediaProvider:
    """Wikipedia action API (https://en.wikipedia.org/w/api.php).

    Uses a single ``generator=search`` + ``prop=extracts`` call to fetch the
    lead (intro) plain-text extract of the best-matching article — no second
    round-trip, no HTML.
    """

    _ENDPOINT = "https://en.wikipedia.org/w/api.php"

    def __init__(self, http: SafeHttpClient) -> None:
        self._http = http

    @property
    def name(self) -> str:
        return "wikipedia"

    async def retrieve(self, query: str, *, max_results: int = 3) -> list[RetrievalResult]:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": str(max(1, min(max_results, 3))),
        }
        try:
            data = await self._http.get_json(self._ENDPOINT, params=params)
        except HttpClientError as exc:
            _logger.debug("Wikipedia retrieval failed: %s", exc)
            return []
        except Exception:  # noqa: BLE001 - never propagate
            _logger.debug("Wikipedia retrieval raised unexpectedly", exc_info=True)
            return []

        pages = (data.get("query") or {}).get("pages") or {}
        if not isinstance(pages, dict):
            return []

        # Preserve the search ranking where available (index field).
        ordered = sorted(
            pages.values(),
            key=lambda p: p.get("index", 1_000) if isinstance(p, dict) else 1_000,
        )

        results: list[RetrievalResult] = []
        for page in ordered:
            if len(results) >= max_results:
                break
            if not isinstance(page, dict):
                continue
            extract = str(page.get("extract") or "").strip()
            title = str(page.get("title") or "").strip()
            if not extract:
                continue
            results.append(RetrievalResult(
                source=self.name,
                title=title or query,
                snippet=_clip(extract),
                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else "",
                score=0.7,
            ))

        return results[:max_results]
