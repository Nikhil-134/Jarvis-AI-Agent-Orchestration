"""Integration tests for the production startup flow.

These tests exercise the real ``build_orchestrator()`` path from
``main.py``, verifying that the MemoryManager is correctly initialised
at application startup rather than lazily inside every operation.

Why the previous test suite missed the runtime bug
---------------------------------------------------
Unit tests for ``MemoryManager`` call ``await mm.initialize()``
explicitly in their fixtures (e.g. ``test_memory_manager.py``).
The orchestrator tests (``test_orchestrator.py``) never involve memory
at all.  Neither file exercises the actual production wiring in
``main.build_orchestrator``, which created a ``MemoryManager`` but
never called ``.initialize()``.

These integration tests close that gap by testing the exact same
code path that ``python main.py`` uses.
"""

from __future__ import annotations

import os

import pytest

from collections.abc import AsyncIterable

from agents import MemoryAgent, PlannerAgent, ToolAgent, VoiceAgent
from agents.contracts import AgentTask
from llm import BaseLLMProvider, LLMConfig
from memory import MemoryManager, MemoryService
from memory.document_store import SQLiteDocumentStore
from memory.models import MemoryItem
from memory.vector_store import ChromaVectorStore
from orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_build_orchestrator_initialises_memory(tmp_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production startup path must initialise MemoryManager so that
    agents never encounter an uninitialised backend at runtime."""
    vector_path = str(tmp_path / "int_vectors")
    doc_path = str(tmp_path / "int_documents.db")

    monkeypatch.setenv("MEMORY_VECTOR_STORE_PATH", vector_path)
    monkeypatch.setenv("MEMORY_DOCUMENT_STORE_PATH", doc_path)
    monkeypatch.setenv("MEMORY_ENABLED", "true")

    from main import build_orchestrator

    orchestrator = await build_orchestrator()

    try:
        health = await orchestrator.health_check()
        assert health["initialized"] is True
        assert health["started"] is True

        memory_agent = next(
            a for a in orchestrator.agents.values() if a.name == "memory"
        )
        result = await memory_agent.handle(
            AgentTask(task_type="memory.stats")
        )
        assert result.success is True
        assert result.data["status"] == "completed"
        assert "stats" in result.data
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_memory_store_and_search_via_agent(tmp_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """After startup, agents must be able to store and retrieve memories."""
    vector_path = str(tmp_path / "store_vectors")
    doc_path = str(tmp_path / "store_documents.db")

    monkeypatch.setenv("MEMORY_VECTOR_STORE_PATH", vector_path)
    monkeypatch.setenv("MEMORY_DOCUMENT_STORE_PATH", doc_path)
    monkeypatch.setenv("MEMORY_ENABLED", "true")

    from main import build_orchestrator

    orchestrator = await build_orchestrator()

    try:
        memory_agent = next(
            a for a in orchestrator.agents.values() if a.name == "memory"
        )

        store_result = await memory_agent.handle(
            AgentTask(
                task_type="memory.store",
                payload={"content": "JARVIS uses ChromaDB for vector search", "memory_type": "fact"},
            )
        )
        assert store_result.success is True
        assert store_result.data["status"] == "stored"

        search_result = await memory_agent.handle(
            AgentTask(
                task_type="memory.search",
                payload={"query": "ChromaDB vector search", "top_k": 5},
            )
        )
        assert search_result.success is True
        assert search_result.data["status"] == "completed"
        assert search_result.data["count"] >= 1
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_planner_enriches_prompt_from_memory(tmp_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The planner agent must be able to enrich a prompt with stored
    memories via its injected MemoryService."""
    vector_path = str(tmp_path / "planner_vectors")
    doc_path = str(tmp_path / "planner_documents.db")

    monkeypatch.setenv("MEMORY_VECTOR_STORE_PATH", vector_path)
    monkeypatch.setenv("MEMORY_DOCUMENT_STORE_PATH", doc_path)
    monkeypatch.setenv("MEMORY_ENABLED", "true")

    from main import build_orchestrator

    orchestrator = await build_orchestrator()

    try:
        memory_agent = next(
            a for a in orchestrator.agents.values() if a.name == "memory"
        )

        await memory_agent.handle(
            AgentTask(
                task_type="memory.store",
                payload={
                    "content": "The user works as a software engineer",
                    "memory_type": "fact",
                    "importance": 0.9,
                },
            )
        )

        planner_agent = next(
            a for a in orchestrator.agents.values() if a.name == "planner"
        )
        result = await planner_agent.handle(
            AgentTask(task_type="plan", payload={"goal": "test memory enrichment"})
        )
        assert result.success is True
        assert "memory_enriched" in result.data
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_orchestrator_lifecycle_with_memory(tmp_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full lifecycle: initialise, start, route health-check, stop."""
    vector_path = str(tmp_path / "life_vectors")
    doc_path = str(tmp_path / "life_documents.db")

    monkeypatch.setenv("MEMORY_VECTOR_STORE_PATH", vector_path)
    monkeypatch.setenv("MEMORY_DOCUMENT_STORE_PATH", doc_path)
    monkeypatch.setenv("MEMORY_ENABLED", "true")

    from main import build_orchestrator

    orchestrator = await build_orchestrator()

    try:
        health = await orchestrator.health_check()
        assert health["initialized"] is True
        assert health["started"] is True

        result = await orchestrator.route(AgentTask(task_type="plan"))
        assert result.success is True
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_uninitialised_memory_manager_raises(tmp_path: str) -> None:
    """An uninitialised MemoryManager must raise RuntimeError.

    This documents the contract: callers MUST invoke .initialize()
    before use.  Unit tests were doing this correctly, but the
    production startup path was not — which is exactly the bug
    these integration tests prevent from recurring.
    """
    from memory import MemoryManager

    mm = MemoryManager(
        dedup_threshold=0.95,
        importance_threshold=0.3,
    )

    with pytest.raises(RuntimeError, match="not initialised"):
        await mm.store(MemoryItem(content="test"))

    with pytest.raises(RuntimeError, match="not initialised"):
        await mm.search("test")

    with pytest.raises(RuntimeError, match="not initialised"):
        await mm.get_stats()


class MemoryRecallProvider(BaseLLMProvider):
    """A test LLM provider whose response depends on what it sees in the prompt.

    Simulates a real LLM that can answer from memory context.  Used to
    verify the two-phase planner correctly passes memory context through
    to the ``responder`` prompt.
    """

    def __init__(self) -> None:
        super().__init__(LLMConfig(provider="recall", model="recall"))

    @property
    def name(self) -> str:
        return "recall"

    async def _generate_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> LLMResponse:
        from llm.interfaces import LLMResponse
        # Return a conversational response based on what memory context
        # the planner includes in the responder prompt.
        if "Boss" in prompt or "Nikhil" in prompt:
            return LLMResponse(content="Hi Boss! Your name is Nikhil and your favourite programming language is Python.")
        if "software engineer" in prompt:
            return LLMResponse(content="You work as a software engineer.")
        return LLMResponse(content="I received your message.")

    async def _stream_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> AsyncIterable[str]:
        response = await self._generate_once(prompt, system_prompt, tools)
        yield response.content


@pytest.mark.asyncio
async def test_planner_response_never_exposes_internal_planning(tmp_path: str) -> None:
    """The planner must NEVER return plans, steps, instructions, or
    internal reasoning in the response field."""
    vector_store = ChromaVectorStore(str(tmp_path / "np_vectors"))
    doc_store = SQLiteDocumentStore(str(tmp_path / "np_documents.db"))
    mm = MemoryManager(vector_store=vector_store, document_store=doc_store, importance_threshold=0.0)
    await mm.initialize()
    memory_service = MemoryService(mm)

    provider = MemoryRecallProvider()
    agent = PlannerAgent(llm_provider=provider, memory_service=memory_service)

    forbidden_patterns = [
        "1.", "2.", "Step", "step", "Plan", "plan",
        "Instruction", "instruction", "First,", "Second,",
        "Actionable", "actionable", "Example", "example",
        "Conversation", "conversation:",
        "You are the Jarvis planning agent",
        "Create a concise execution plan",
        "Return actionable steps",
    ]

    result = await agent.handle(
        AgentTask(task_type="plan", payload={"goal": "What is my name?"})
    )

    assert result.success is True
    assert "response" in result.data

    response = result.data["response"]
    for pattern in forbidden_patterns:
        assert pattern not in response, (
            f"Response leaked internal planning: found {pattern!r} in {response!r}"
        )


@pytest.mark.asyncio
async def test_memory_recall_with_personal_info(tmp_path: str) -> None:
    """Full memory recall cycle: store personal facts, then verify the
    planner retrieves and includes them in its conversational response."""
    vector_store = ChromaVectorStore(str(tmp_path / "mr_vectors"))
    doc_store = SQLiteDocumentStore(str(tmp_path / "mr_documents.db"))
    mm = MemoryManager(vector_store=vector_store, document_store=doc_store, importance_threshold=0.0)
    await mm.initialize()
    memory_service = MemoryService(mm)

    memory_agent = MemoryAgent(memory_service=memory_service)

    # Store personal facts (simulating previous conversation turns)
    await memory_agent.handle(
        AgentTask(task_type="memory.store", payload={"content": "Hi, my name is Nikhil.", "memory_type": "fact", "importance": 0.9})
    )
    await memory_agent.handle(
        AgentTask(task_type="memory.store", payload={"content": "Call me Boss.", "memory_type": "preference", "importance": 0.9})
    )
    await memory_agent.handle(
        AgentTask(task_type="memory.store", payload={"content": "My favourite language is Python.", "memory_type": "preference", "importance": 0.9})
    )

    # Verify memories were stored
    stats = await memory_agent.handle(AgentTask(task_type="memory.stats"))
    assert stats.success is True
    assert stats.data["stats"]["vector_count"] >= 3

    # Query the planner with memory recall provider
    provider = MemoryRecallProvider()
    planner_agent = PlannerAgent(llm_provider=provider, memory_service=memory_service)

    result = await planner_agent.handle(
        AgentTask(task_type="plan", payload={"goal": "What is my name?"})
    )

    assert result.success is True
    assert result.data["memory_enriched"] is True
    assert result.data["memory_count"] >= 1

    response = result.data["response"]
    # The provider returns this when it sees "Nikhil" or "Boss" in the prompt
    assert "Nikhil" in response
    assert "Boss" in response or "boss" in response.lower()
    assert "Python" in response
