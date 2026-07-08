"""Production Stabilization Cycle — regression tests.

Each test here reproduces a specific runtime contract violation that was
observed in production. They are written to FAIL against the pre-fix code and
PASS after the fix, and they use only fast in-process fakes (no network, no
Ollama, no ChromaDB) so they run in CI.

Bugs covered:
  1. ``'NoneType' object has no attribute 'strip'/'split'`` — a provider that
     returns ``None`` content crashed the planner / decomposer. Contract now
     enforced at :class:`~llm.interfaces.LLMResponse` and every consumer.
  2. Memory routing — identity/preference questions must consult persistent
     memory BEFORE the LLM and must never leave the machine.
  4. Internet routing — weather / news / "latest" / current office-holders must
     reach the KnowledgeEngine (→ InternetKnowledgeService); timeless questions
     stay local.
  5. Internal exceptions/messages ("memory_id is required", "not installed",
     stack traces) must never reach the user.
  6/7. Every subsystem boundary validates inputs and never assumes ``None`` is
     a string/list/object.
"""

from __future__ import annotations

import pytest

from agents.contracts import AgentResult, AgentTask
from agents.jarvis_agent import JarvisPrimeAgent
from agents.planner import PlannerAgent
from llm.base import BaseLLMProvider, LLMConfig
from llm.interfaces import LLMResponse
from runtime.conversation_runtime import ConversationRuntime
from runtime.intent_engine import IntentEngine
from runtime.knowledge_engine import KnowledgeEngine
from runtime.llm_guard import LLMGuard
from runtime.response_composer import RuntimeResponseComposer
from tools.builtins import register_all_builtins
from tools.intent_detector import IntentDetector
from tools.manager import ToolManager
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _NoneProvider(BaseLLMProvider):
    """A provider that returns ``None`` content — the exact production trigger."""

    @property
    def name(self) -> str:
        return "none-provider"

    async def _generate_once(self, prompt, system_prompt, tools):
        return LLMResponse(content=None)  # contract violation source

    async def _stream_once(self, prompt, system_prompt, tools):
        yield ""


class _NoneGuard:
    """LLM guard whose generate() yields None content."""

    is_available = True

    async def generate(self, prompt, system_prompt=None, tools=None):
        return LLMResponse(content=None)


def _intent_engine() -> tuple[IntentDetector, IntentEngine]:
    reg = ToolRegistry()
    register_all_builtins(reg)
    detector = IntentDetector(ToolManager(registry=reg))
    return detector, IntentEngine(detector)


# ===========================================================================
# Bug 1 / 7 — LLMResponse contract: None content is never propagated
# ===========================================================================

class TestLLMResponseContract:

    def test_none_content_becomes_empty_string(self) -> None:
        assert LLMResponse(content=None).content == ""

    def test_non_string_content_is_stringified(self) -> None:
        assert LLMResponse(content=123).content == "123"

    def test_none_tool_calls_becomes_empty_tuple(self) -> None:
        assert LLMResponse(content="x", tool_calls=None).tool_calls == ()

    def test_content_is_always_stripable(self) -> None:
        # The exact failing operation from production must not raise.
        assert LLMResponse(content=None).content.strip() == ""

    async def test_generate_text_never_returns_none(self) -> None:
        provider = _NoneProvider(LLMConfig(provider="none", model="x"))
        text = await provider.generate_text("hi")
        assert text == ""
        # And the reported crash site is safe:
        assert text.split(",") == [""]


# ===========================================================================
# Bug 1 — planner / jarvis survive a None-content provider
# ===========================================================================

class TestNoneContentDoesNotCrash:

    async def test_planner_handles_none_content(self) -> None:
        provider = _NoneProvider(LLMConfig(provider="none", model="x"))
        planner = PlannerAgent(llm_provider=provider)
        result = await planner.handle(
            AgentTask(task_type="plan", payload={"goal": "remember my name is Nikhil"})
        )
        assert result.success
        # No None leaks into data["response"].
        assert isinstance(result.data.get("response"), str)
        assert result.data["response"].strip() != ""

    async def test_jarvis_decompose_handles_none_content(self) -> None:
        provider = _NoneProvider(LLMConfig(provider="none", model="x"))
        jarvis = JarvisPrimeAgent(llm_provider=provider)
        # This goal previously hit `response.split(",")` on a None value.
        result = await jarvis.handle(
            AgentTask(
                task_type="jarvis.process",
                payload={"goal": "back up my files and deploy the build", "task_type": "plan"},
            )
        )
        assert isinstance(result.data.get("response"), str)

    @pytest.mark.parametrize("text", ["remember my name is Nikhil", "Call me boss"])
    async def test_pipeline_never_crashes_on_none_content(self, text: str) -> None:
        detector, _ = _intent_engine()
        runtime = ConversationRuntime(
            orchestrator=None, intent_detector=detector, llm_guard=_NoneGuard(),
        )
        # Must return a clean string, never raise AttributeError.
        out = await runtime.process(text)
        assert isinstance(out, str)


# ===========================================================================
# Bug 4 — internet routing: current_info reaches the KnowledgeEngine
# ===========================================================================

class TestInternetRouting:

    def setup_method(self) -> None:
        self.detector, self.engine = _intent_engine()
        self.runtime = ConversationRuntime(
            orchestrator=None, intent_detector=self.detector, llm_guard=_NoneGuard(),
        )

    @pytest.mark.parametrize("query", [
        "Weather in Bangalore",
        "Latest AI news",
        "What is the news today",
        "Who is the current Prime Minister",
    ])
    def test_time_sensitive_goes_to_knowledge_engine(self, query: str) -> None:
        intent = self.engine.classify(query)
        assert self.runtime._is_knowledge_intent(intent), (
            f"{query!r} must reach the KnowledgeEngine / InternetKnowledgeService"
        )

    @pytest.mark.parametrize("query", ["What is Python?", "Explain recursion", "What is recursion?"])
    def test_timeless_stays_local(self, query: str) -> None:
        from knowledge.internet.router import needs_internet
        assert not needs_internet(query), f"{query!r} must stay local"

    def test_math_still_routes_to_tool_not_knowledge(self) -> None:
        intent = self.engine.classify("Solve 23*(18+7)")
        assert not self.runtime._is_knowledge_intent(intent)


# ===========================================================================
# Bug 2 — memory queries consult persistent memory before the LLM, never net
# ===========================================================================

class _RecordingGuard:
    is_available = True

    def __init__(self) -> None:
        self.last_prompt = ""

    async def generate(self, prompt, system_prompt=None, tools=None):
        self.last_prompt = prompt
        return LLMResponse(content="natural answer")


class _FakeProfileMemory:
    """Stand-in PersistentMemoryService exposing a user profile."""

    def __init__(self, profile: dict[str, str]) -> None:
        self._profile = profile

    async def get_profile(self) -> dict[str, str]:
        return dict(self._profile)


class _FakeSemanticMemory:
    """Stand-in MemoryService whose enrich_prompt returns fixed memories."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = contents

    async def enrich_prompt(self, prompt, top_k=5, per_memory_chars=500, max_context_length=2000):
        class _M:
            def __init__(self, c: str) -> None:
                self.content = c
        return prompt, [_M(c) for c in self._contents]


class TestMemoryFirstRecall:

    async def test_profile_injected_before_llm_for_identity_query(self) -> None:
        guard = _RecordingGuard()
        eng = KnowledgeEngine(
            guard, None,
            persistent_memory=_FakeProfileMemory({"name": "Nikhil"}),
        )
        await eng.answer("Who am I?")
        # The durable fact was placed into the prompt BEFORE generation.
        assert "Nikhil" in guard.last_prompt

    async def test_semantic_memory_injected_for_preference_query(self) -> None:
        guard = _RecordingGuard()
        eng = KnowledgeEngine(
            guard, _FakeSemanticMemory(["User: Remember I like Python\nJARVIS: Noted."]),
        )
        await eng.answer("What do I like?")
        assert "Python" in guard.last_prompt

    async def test_memory_query_never_hits_internet(self) -> None:
        called = {"net": False}

        class _Net:
            available = True

            async def build_context(self, query, *, max_results=5):
                called["net"] = True
                return "should-not-be-used"

        eng = KnowledgeEngine(_RecordingGuard(), None, internet_service=_Net())
        await eng.answer("what is my name?")
        assert called["net"] is False

    async def test_no_memory_still_answers_naturally(self) -> None:
        guard = _RecordingGuard()
        eng = KnowledgeEngine(guard, None, persistent_memory=_FakeProfileMemory({}))
        out = await eng.answer("Who am I?")
        assert out == "natural answer"


# ===========================================================================
# Bug 5 — internal failure messages never reach the user
# ===========================================================================

class TestNoInternalLeaks:

    def setup_method(self) -> None:
        self.composer = RuntimeResponseComposer()

    @pytest.mark.parametrize("internal_msg", [
        "memory_id is required for retrieval.",
        "Query is required for search.",
        "Browser automation requires a browser engine (not installed).",
        "pyautogui not installed",
        "JarvisPrimeAgent cannot handle task type: foo",
    ])
    async def test_failed_result_message_is_not_leaked(self, internal_msg: str) -> None:
        result = AgentResult(
            agent_name="memory", task_id="t", success=False, message=internal_msg,
        )
        out = await self.composer.compose("goal", result)
        assert internal_msg not in out
        assert out.strip() != ""

    async def test_user_facing_response_is_preserved(self) -> None:
        result = AgentResult(
            agent_name="jarvis", task_id="t", success=True,
            message="", data={"response": "Here is your answer."},
        )
        out = await self.composer.compose("goal", result)
        assert "Here is your answer." in out

    async def test_no_stack_trace_leaks(self) -> None:
        result = AgentResult(
            agent_name="x", task_id="t", success=False,
            message="Traceback (most recent call last):\n  ...\nAttributeError: 'NoneType'",
        )
        out = await self.composer.compose("goal", result)
        assert "Traceback" not in out
        assert "AttributeError" not in out
