"""Tests for cross-session persistence wiring in the conversation runtime.

These verify roadmap #8: the runtime records every meaningful turn into the
PersistentMemoryService, and does so *honestly* — a storage failure is logged
but never breaks the user-facing reply, and the layer's own store-worthiness
gate is respected.

The tests drive ``ConversationRuntime`` directly with lightweight fakes (no
LLM, no ChromaDB) so they are fast and deterministic. End-to-end restart
recall with real backends is covered by ``test_persistent_memory.py`` and the
manual restart proof recorded in PROJECT_BRAIN.
"""

from __future__ import annotations

import pytest

from runtime.conversation_runtime import ConversationRuntime


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakePersistentMemory:
    """Records the turns handed to ``record_turn``."""

    def __init__(self) -> None:
        self.turns: list[tuple[str, str, str]] = []

    async def record_turn(self, session_id: str, user_text: str, assistant_text: str, **_):
        self.turns.append((session_id, user_text, assistant_text))
        return "mem-id"


class RaisingPersistentMemory:
    """A persistent layer whose write always fails (durability outage)."""

    def __init__(self) -> None:
        self.called = False

    async def record_turn(self, *args, **kwargs):
        self.called = True
        raise RuntimeError("simulated store outage")


class LearningPersistentMemory(FakePersistentMemory):
    """Persistent layer that also records auto-learned preference text."""

    def __init__(self) -> None:
        super().__init__()
        self.learned: list[str] = []

    async def learn_preferences(self, user_text: str):
        self.learned.append(user_text)
        return ["name"] if "call me" in user_text.lower() else []


class RaisingLearnMemory(FakePersistentMemory):
    """learn_preferences always fails — must not break the reply."""

    async def learn_preferences(self, user_text: str):
        raise RuntimeError("simulated learn outage")


class FakeKnowledge:
    """Stands in for the KnowledgeEngine's direct-LLM path."""

    available = True

    def __init__(self, answer_text: str = "Paris is the capital of France.") -> None:
        self._answer = answer_text

    async def answer(self, text: str) -> str:
        return self._answer


def _runtime(persistent, answer: str = "Paris is the capital of France.") -> ConversationRuntime:
    """Build a ConversationRuntime whose knowledge path is a deterministic fake."""
    rt = ConversationRuntime(
        orchestrator=None,
        persistent_memory=persistent,
        session_id="test-session",
    )
    # Replace the real (LLM-backed) knowledge engine with a fake so the
    # knowledge fast-path is exercised without a model.
    rt.knowledge = FakeKnowledge(answer)
    return rt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_turn_is_recorded() -> None:
    mem = FakePersistentMemory()
    rt = _runtime(mem)

    reply = await rt.process("What is the capital of France?")

    assert "Paris" in reply
    assert len(mem.turns) == 1
    session_id, user_text, assistant_text = mem.turns[0]
    assert session_id == "test-session"
    assert user_text == "What is the capital of France?"
    assert "Paris" in assistant_text


@pytest.mark.asyncio
async def test_greeting_fast_path_is_not_recorded() -> None:
    """Canned personality greetings never reach the persistent layer."""
    mem = FakePersistentMemory()
    rt = _runtime(mem)

    reply = await rt.process("hello")

    assert reply  # a greeting came back
    assert mem.turns == []  # but nothing was persisted


@pytest.mark.asyncio
async def test_storage_failure_does_not_break_reply() -> None:
    """A failing persistent write is swallowed for the user but the answer stands."""
    mem = RaisingPersistentMemory()
    rt = _runtime(mem)

    reply = await rt.process("What is the capital of France?")

    assert "Paris" in reply  # user still gets the real answer
    assert mem.called is True  # and we genuinely attempted the write


@pytest.mark.asyncio
async def test_no_persistent_layer_is_a_noop() -> None:
    """With persistence disabled (None), the pipeline behaves exactly as before."""
    rt = _runtime(None)
    reply = await rt.process("What is the capital of France?")
    assert "Paris" in reply  # no crash, normal answer


@pytest.mark.asyncio
async def test_empty_input_records_nothing() -> None:
    mem = FakePersistentMemory()
    rt = _runtime(mem)
    reply = await rt.process("   ")
    assert reply == ""
    assert mem.turns == []


# ---------------------------------------------------------------------------
# Cycle 9 — automatic preference learning wired into the persist path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preferences_learned_on_turn() -> None:
    mem = LearningPersistentMemory()
    rt = _runtime(mem, answer="Sure thing.")

    await rt.process("call me Boss")

    assert mem.learned == ["call me Boss"]  # user text handed to the learner


@pytest.mark.asyncio
async def test_auto_learn_can_be_disabled() -> None:
    mem = LearningPersistentMemory()
    rt = ConversationRuntime(
        orchestrator=None,
        persistent_memory=mem,
        session_id="test-session",
        auto_learn_preferences=False,
    )
    rt.knowledge = FakeKnowledge("Sure thing.")

    await rt.process("call me Boss")

    assert mem.learned == []  # disabled → learner never called
    assert len(mem.turns) == 1  # but the turn is still recorded


@pytest.mark.asyncio
async def test_learn_failure_does_not_break_reply() -> None:
    mem = RaisingLearnMemory()
    rt = _runtime(mem, answer="Sure thing.")

    reply = await rt.process("call me Boss")

    assert reply == "Sure thing."  # a failing learner never breaks the reply
    assert len(mem.turns) == 1  # the turn still recorded


@pytest.mark.asyncio
async def test_minimal_fake_without_learn_is_safe() -> None:
    """A persistent layer with only record_turn must not error (getattr guard)."""
    mem = FakePersistentMemory()  # no learn_preferences attribute
    rt = _runtime(mem, answer="Sure thing.")

    reply = await rt.process("call me Boss")

    assert reply == "Sure thing."
    assert len(mem.turns) == 1
