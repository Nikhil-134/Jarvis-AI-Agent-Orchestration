"""Tests for the KnowledgeEngine ↔ InternetKnowledgeService wiring.

Verifies the priority ladder at the engine level:
  * timeless questions never trigger a fetch;
  * memory/personal questions never trigger a fetch (privacy);
  * time-sensitive questions do, and the live context is injected into the
    prompt as untrusted data;
  * ONLY the current question is sent outward (no history/memory/secrets);
  * retrieval failures never break the answer (fail safe).
"""

from __future__ import annotations

from runtime.knowledge_engine import KnowledgeEngine


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = ()


class _FakeGuard:
    is_available = True

    def __init__(self, answer: str = "answer") -> None:
        self._answer = answer
        self.last_prompt: str | None = None

    async def generate(self, prompt: str, system_prompt: str | None = None, tools=None):
        self.last_prompt = prompt
        return _FakeResponse(self._answer)


class _FakeInternet:
    available = True

    def __init__(self, block: str = "<LIVE INTERNET RESULTS>[1] fresh fact</LIVE INTERNET RESULTS>") -> None:
        self._block = block
        self.queries: list[str] = []

    async def build_context(self, query: str, *, max_results: int = 5) -> str:
        self.queries.append(query)
        return self._block


class _RaisingInternet:
    available = True

    def __init__(self) -> None:
        self.called = False

    async def build_context(self, query: str, *, max_results: int = 5) -> str:
        self.called = True
        raise RuntimeError("network down")


async def test_timeless_question_does_not_fetch() -> None:
    net = _FakeInternet()
    eng = KnowledgeEngine(_FakeGuard(), None, internet_service=net)
    await eng.answer("What is Python?")
    assert net.queries == []


async def test_memory_question_does_not_fetch() -> None:
    net = _FakeInternet()
    eng = KnowledgeEngine(_FakeGuard(), None, internet_service=net)
    await eng.answer("what did we discuss yesterday?")
    assert net.queries == []


async def test_fresh_question_fetches_and_injects() -> None:
    guard = _FakeGuard()
    net = _FakeInternet()
    eng = KnowledgeEngine(guard, None, internet_service=net)
    await eng.answer("what is the latest AI news today?")
    assert net.queries  # fetched
    assert "LIVE INTERNET RESULTS" in (guard.last_prompt or "")
    assert "fresh fact" in (guard.last_prompt or "")


async def test_only_the_question_is_sent_outward() -> None:
    """Privacy: history/memory are never forwarded to the retrieval layer."""
    guard = _FakeGuard()
    net = _FakeInternet()
    eng = KnowledgeEngine(guard, None, internet_service=net)
    # Seed some history first (a prior private turn).
    await eng.answer("my secret is hunter2")   # timeless -> no fetch
    await eng.answer("latest news today")       # fresh -> fetch
    assert net.queries == ["latest news today"]
    assert all("hunter2" not in q for q in net.queries)


async def test_retrieval_failure_does_not_break_answer() -> None:
    net = _RaisingInternet()
    eng = KnowledgeEngine(_FakeGuard("still answered"), None, internet_service=net)
    answer = await eng.answer("latest news today")
    assert answer == "still answered"
    assert net.called is True


async def test_no_internet_service_is_local_only() -> None:
    guard = _FakeGuard("local")
    eng = KnowledgeEngine(guard, None, internet_service=None)
    answer = await eng.answer("latest news today")
    assert answer == "local"
    assert "LIVE INTERNET" not in (guard.last_prompt or "")
