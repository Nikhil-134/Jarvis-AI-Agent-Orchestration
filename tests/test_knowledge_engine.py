"""Tests for the KnowledgeEngine direct-LLM path and the memory storage gate.

These guard the fix for the "general questions return empty / greeting
fallback" regression, where knowledge queries were routed through the
rule-based planner + specialist workflow and poisoned by low-value memories.
"""

from __future__ import annotations

import pytest

from memory.memory_service import MemoryService
from runtime.knowledge_engine import KnowledgeEngine


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = ()


class _FakeGuard:
    """Minimal LLMGuard stand-in that echoes a canned answer."""

    def __init__(self, answer: str = "A real answer.", available: bool = True) -> None:
        self._answer = answer
        self.is_available = available
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    async def generate(self, prompt: str, system_prompt: str | None = None, tools=None):
        self.last_prompt = prompt
        self.last_system = system_prompt
        return _FakeResponse(self._answer)


# --------------------------------------------------------------------------
# KnowledgeEngine
# --------------------------------------------------------------------------


async def test_answers_knowledge_question_directly():
    engine = KnowledgeEngine(_FakeGuard("Paris is the capital of France."))
    answer = await engine.answer("What is the capital of France?")
    assert answer == "Paris is the capital of France."


async def test_empty_input_returns_empty():
    engine = KnowledgeEngine(_FakeGuard())
    assert await engine.answer("   ") == ""


async def test_unavailable_guard_gives_graceful_message():
    engine = KnowledgeEngine(_FakeGuard(available=False))
    answer = await engine.answer("Explain photosynthesis")
    assert "language model" in answer.lower()


async def test_blank_llm_answer_falls_back():
    engine = KnowledgeEngine(_FakeGuard(answer="   "))
    answer = await engine.answer("Explain something")
    assert answer  # non-empty fallback
    assert "rephrase" in answer.lower()


async def test_history_is_carried_into_prompt():
    guard = _FakeGuard("Answer two.")
    engine = KnowledgeEngine(guard)
    await engine.answer("First question")
    await engine.answer("Second question")
    # The second prompt should contain the first turn's content.
    assert "First question" in guard.last_prompt
    assert "Second question" in guard.last_prompt


async def test_reset_clears_history():
    guard = _FakeGuard("ok")
    engine = KnowledgeEngine(guard)
    await engine.answer("remember this")
    engine.reset()
    await engine.answer("fresh start")
    assert "remember this" not in guard.last_prompt


async def test_no_tools_or_json_leak_in_prompt():
    guard = _FakeGuard("clean")
    engine = KnowledgeEngine(guard)
    await engine.answer("What is gravity?")
    # The knowledge path must never inject tool definitions or $placeholders.
    assert "$tool_results" not in guard.last_prompt
    assert "tool_calls" not in guard.last_prompt


# --------------------------------------------------------------------------
# Memory storage gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,response,expected",
    [
        ("q", "Paris is the capital of France.", True),
        ("q", "Yes.", True),
        ("q", "Hello! How can I help you today?", False),
        ("q", "I received your message. How can I assist you further?", False),
        ("q", "", False),
        ("", "answer", False),
        ("q", '{"name": "text", "arguments": {"x": 1}}', False),
        ("q", "OK", False),  # shorter than 3 chars
    ],
)
def test_storage_gate(query, response, expected):
    assert MemoryService._is_storeworthy(query, response) is expected
