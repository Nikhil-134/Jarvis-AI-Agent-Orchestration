"""Tests for the Internet Knowledge Engine (roadmap #9).

Covers: the freshness router (priority ladder), the SafeHttpClient security
boundary (SSRF/whitelist/size/redirect), both providers against mocked JSON,
the service (parallel fan-out, dedupe, cache, timeout, fail-safe), and the
KnowledgeEngine wiring (internet consulted only when required, never for
memory/timeless queries).

No test touches the live network: httpx traffic is served by an in-process
``MockTransport``; time is driven by injected clocks.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from knowledge.internet import (
    DuckDuckGoProvider,
    InternetKnowledgeService,
    SafeHttpClient,
    TTLCache,
    WikipediaProvider,
    is_memory_query,
    needs_internet,
)
from knowledge.internet.http_client import HttpClientError, _host_allowed
from knowledge.internet.interfaces import RetrievalResult


# =========================================================================
# Router — the priority-ladder gate
# =========================================================================

class TestRouter:
    @pytest.mark.parametrize("query", [
        "What is Python?", "Explain recursion", "define a monad",
        "how does TCP work", "why is the sky blue",
    ])
    def test_timeless_questions_stay_local(self, query: str) -> None:
        assert needs_internet(query) is False

    @pytest.mark.parametrize("query", [
        "what's today's weather", "latest AI news", "newest NVIDIA GPU",
        "who is the prime minister of india", "current president of france",
        "bitcoin price right now", "cricket score today",
    ])
    def test_fresh_questions_need_internet(self, query: str) -> None:
        assert needs_internet(query) is True

    @pytest.mark.parametrize("query", [
        "what did we discuss yesterday", "what did I ask earlier",
        "what do I like", "remember what my favourite colour is",
        "you said something about that", "did we talk about python",
    ])
    def test_memory_questions_never_go_to_internet(self, query: str) -> None:
        assert is_memory_query(query) is True
        assert needs_internet(query) is False

    def test_empty_query_is_local(self) -> None:
        assert needs_internet("") is False
        assert needs_internet("   ") is False

    def test_latest_overrides_conceptual_prefix(self) -> None:
        # "what is the latest iPhone" is conceptual-prefixed but time-sensitive.
        assert needs_internet("what is the latest iphone") is True


# =========================================================================
# SafeHttpClient — security boundary
# =========================================================================

class TestHostAllowed:
    def test_exact_and_subdomain(self) -> None:
        allowed = ("wikipedia.org", "api.duckduckgo.com")
        assert _host_allowed("wikipedia.org", allowed)
        assert _host_allowed("en.wikipedia.org", allowed)
        assert _host_allowed("api.duckduckgo.com", allowed)

    def test_rejects_lookalike_and_empty(self) -> None:
        allowed = ("wikipedia.org",)
        assert not _host_allowed("evilwikipedia.org", allowed)
        assert not _host_allowed("wikipedia.org.evil.com", allowed)
        assert not _host_allowed("", allowed)


class TestSafeHttpClientPolicy:
    def test_is_allowed_matrix(self) -> None:
        c = SafeHttpClient()
        assert c.is_allowed("https://api.duckduckgo.com/")
        assert c.is_allowed("https://en.wikipedia.org/w/api.php")
        assert not c.is_allowed("http://en.wikipedia.org/")      # not https
        assert not c.is_allowed("https://evil.com/")             # not whitelisted
        assert not c.is_allowed("https://evilwikipedia.org/")    # lookalike
        assert not c.is_allowed("file:///etc/passwd")            # scheme
        assert not c.is_allowed("https://127.0.0.1/")            # SSRF target
        assert not c.is_allowed("https://localhost/")            # SSRF target

    async def test_blocked_url_raises_before_network(self) -> None:
        # transport that would explode if ever called
        def _boom(request):  # pragma: no cover - must not be reached
            raise AssertionError("network was contacted for a blocked URL")

        c = SafeHttpClient(transport=httpx.MockTransport(_boom))
        with pytest.raises(HttpClientError):
            await c.get_json("https://evil.com/data")

    async def test_get_json_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "api.duckduckgo.com"
            return httpx.Response(200, json={"ok": True})

        c = SafeHttpClient(transport=httpx.MockTransport(handler))
        data = await c.get_json("https://api.duckduckgo.com/", params={"q": "x"})
        assert data == {"ok": True}

    async def test_redirect_is_refused(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"Location": "https://evil.com/"})

        c = SafeHttpClient(transport=httpx.MockTransport(handler), max_retries=0)
        with pytest.raises(HttpClientError):
            await c.get_json("https://api.duckduckgo.com/")

    async def test_http_error_status_fails(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"err": "boom"})

        c = SafeHttpClient(transport=httpx.MockTransport(handler), max_retries=0)
        with pytest.raises(HttpClientError):
            await c.get_json("https://api.duckduckgo.com/")

    async def test_oversized_response_is_capped(self) -> None:
        big = "x" * (2 * 1024)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": big})

        c = SafeHttpClient(transport=httpx.MockTransport(handler),
                           max_response_bytes=512, max_retries=0)
        with pytest.raises(HttpClientError):
            await c.get_json("https://api.duckduckgo.com/")

    async def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                raise httpx.ConnectError("transient", request=request)
            return httpx.Response(200, json={"ok": calls["n"]})

        c = SafeHttpClient(transport=httpx.MockTransport(handler), max_retries=2)
        data = await c.get_json("https://api.duckduckgo.com/")
        assert data == {"ok": 2}
        assert calls["n"] == 2

    async def test_invalid_json_fails(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json")

        c = SafeHttpClient(transport=httpx.MockTransport(handler), max_retries=0)
        with pytest.raises(HttpClientError):
            await c.get_json("https://api.duckduckgo.com/")


# =========================================================================
# Providers — against mocked JSON payloads
# =========================================================================

class TestDuckDuckGoProvider:
    async def test_parses_abstract_and_related(self) -> None:
        payload = {
            "Heading": "Python",
            "AbstractText": "Python is a programming language.",
            "AbstractURL": "https://duckduckgo.com/Python",
            "RelatedTopics": [
                {"Text": "Python (programming)", "FirstURL": "https://x/1"},
                {"Text": "Monty Python", "FirstURL": "https://x/2"},
            ],
        }
        c = SafeHttpClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=payload)))
        results = await DuckDuckGoProvider(c).retrieve("python", max_results=3)
        assert results
        assert results[0].source == "duckduckgo"
        assert "programming language" in results[0].snippet
        assert results[0].score >= 0.7

    async def test_failure_returns_empty(self) -> None:
        c = SafeHttpClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(500)), max_retries=0)
        assert await DuckDuckGoProvider(c).retrieve("python") == []


class TestWikipediaProvider:
    async def test_parses_extracts(self) -> None:
        payload = {
            "query": {
                "pages": {
                    "123": {"index": 1, "title": "Python (programming language)",
                            "extract": "Python is a high-level language."},
                    "456": {"index": 2, "title": "Pythonidae",
                            "extract": "Pythonidae is a family of snakes."},
                }
            }
        }
        c = SafeHttpClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=payload)))
        results = await WikipediaProvider(c).retrieve("python", max_results=2)
        assert len(results) == 2
        assert results[0].title.startswith("Python")   # index=1 first
        assert results[0].url.startswith("https://en.wikipedia.org/wiki/")

    async def test_empty_pages_returns_empty(self) -> None:
        c = SafeHttpClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"query": {"pages": {}}})))
        assert await WikipediaProvider(c).retrieve("zzz") == []


# =========================================================================
# TTLCache
# =========================================================================

class TestTTLCache:
    async def test_hit_and_miss(self) -> None:
        cache: TTLCache = TTLCache(ttl_seconds=100, clock=lambda: 0.0)
        await cache.set("k", ["v"])
        assert await cache.get("k") == ["v"]
        assert await cache.get("absent") is None

    async def test_expiry(self) -> None:
        now = {"t": 0.0}
        cache: TTLCache = TTLCache(ttl_seconds=10, clock=lambda: now["t"])
        await cache.set("k", "v")
        now["t"] = 5.0
        assert await cache.get("k") == "v"
        now["t"] = 11.0
        assert await cache.get("k") is None   # expired

    async def test_eviction_bound(self) -> None:
        cache: TTLCache = TTLCache(ttl_seconds=100, max_entries=2, clock=lambda: 0.0)
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)     # evicts oldest ("a")
        assert await cache.get("a") is None
        assert await cache.get("c") == 3


# =========================================================================
# InternetKnowledgeService — orchestration
# =========================================================================

class _StubProvider:
    def __init__(self, name: str, results: list[RetrievalResult], *, delay: float = 0.0,
                 raises: bool = False) -> None:
        self._name = name
        self._results = results
        self._delay = delay
        self._raises = raises
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def retrieve(self, query: str, *, max_results: int = 3) -> list[RetrievalResult]:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises:
            raise RuntimeError("provider boom")
        return self._results


def _r(source: str, snippet: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(source=source, title=source, snippet=snippet, score=score)


class TestInternetKnowledgeService:
    async def test_parallel_merge_and_rank(self) -> None:
        p1 = _StubProvider("ddg", [_r("ddg", "alpha fact", 0.9)])
        p2 = _StubProvider("wiki", [_r("wiki", "beta fact", 0.7)])
        svc = InternetKnowledgeService([p1, p2], min_interval_seconds=0)
        out = await svc.retrieve("q")
        assert [r.snippet for r in out] == ["alpha fact", "beta fact"]  # score desc

    async def test_dedupe(self) -> None:
        p1 = _StubProvider("ddg", [_r("ddg", "same fact", 0.9)])
        p2 = _StubProvider("wiki", [_r("wiki", "same fact", 0.7)])
        svc = InternetKnowledgeService([p1, p2], min_interval_seconds=0)
        out = await svc.retrieve("q")
        assert len(out) == 1

    async def test_one_provider_failure_is_survivable(self) -> None:
        good = _StubProvider("wiki", [_r("wiki", "good fact")])
        bad = _StubProvider("ddg", [], raises=True)
        svc = InternetKnowledgeService([bad, good], min_interval_seconds=0)
        out = await svc.retrieve("q")
        assert [r.snippet for r in out] == ["good fact"]

    async def test_timeout_returns_partial_or_empty(self) -> None:
        slow = _StubProvider("slow", [_r("slow", "late")], delay=5.0)
        svc = InternetKnowledgeService([slow], overall_timeout=0.05, min_interval_seconds=0)
        out = await svc.retrieve("q")
        assert out == []   # nothing completed in time; fail-safe empty

    async def test_cache_prevents_second_fetch(self) -> None:
        p = _StubProvider("ddg", [_r("ddg", "cached fact")])
        svc = InternetKnowledgeService([p], min_interval_seconds=0)
        await svc.retrieve("same query")
        await svc.retrieve("SAME QUERY")   # case-insensitive key
        assert p.calls == 1                # served from cache the second time

    async def test_build_context_is_delimited_and_safe(self) -> None:
        p = _StubProvider("ddg", [_r("ddg", "the fact")])
        svc = InternetKnowledgeService([p], min_interval_seconds=0)
        block = await svc.build_context("q")
        assert "LIVE INTERNET RESULTS" in block
        assert "never follow any" in block.lower()   # injection framing
        assert "the fact" in block

    async def test_no_results_yields_empty_context(self) -> None:
        p = _StubProvider("ddg", [])
        svc = InternetKnowledgeService([p], min_interval_seconds=0)
        assert await svc.build_context("q") == ""

    async def test_empty_query_short_circuits(self) -> None:
        p = _StubProvider("ddg", [_r("ddg", "x")])
        svc = InternetKnowledgeService([p], min_interval_seconds=0)
        assert await svc.retrieve("   ") == []
        assert p.calls == 0
