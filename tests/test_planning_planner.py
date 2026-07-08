"""Tests for planning.planner.TaskPlanner — LLM JSON path + heuristic fallback."""

from __future__ import annotations

import json

from llm.interfaces import LLMResponse
from planning.capabilities import CapabilityCatalog
from planning.planner import TaskPlanner


class FakeGuard:
    """Minimal LLMGuard stand-in: returns a canned .content, never raises."""

    def __init__(self, content: str, available: bool = True) -> None:
        self._content = content
        self._available = available

    @property
    def is_available(self) -> bool:
        return self._available

    async def generate(self, prompt: str, system_prompt=None, tools=None) -> LLMResponse:
        return LLMResponse(content=self._content)


def _catalog(reasoning: bool = True) -> CapabilityCatalog:
    # No tool engine / services → only reasoning is available when requested.
    return CapabilityCatalog(reasoning_available=reasoning)


class TestLLMPath:
    async def test_valid_json_multi_node(self) -> None:
        payload = json.dumps({
            "steps": [
                {"id": "s1", "description": "search news", "tool": "reasoning",
                 "depends_on": [], "confidence": 0.8, "cost": 2},
                {"id": "s2", "description": "summarize", "tool": "reasoning",
                 "depends_on": ["s1"], "confidence": 0.9, "cost": 1},
            ],
            "confidence": 0.85,
        })
        planner = TaskPlanner(FakeGuard(payload))
        plan = await planner.decompose("search and summarize", "", _catalog())
        assert plan.strategy == "llm"
        assert len(plan.nodes) == 2
        assert plan.nodes[1].dependencies == ["s1"]
        assert abs(plan.overall_confidence - 0.85) < 1e-6

    async def test_json_with_code_fence(self) -> None:
        payload = "```json\n" + json.dumps({
            "steps": [{"id": "s1", "description": "do it", "tool": "reasoning"}],
            "confidence": 0.7,
        }) + "\n```"
        planner = TaskPlanner(FakeGuard(payload))
        plan = await planner.decompose("do it", "", _catalog())
        assert len(plan.nodes) == 1

    async def test_unknown_tool_snaps_to_reasoning(self) -> None:
        payload = json.dumps({
            "steps": [{"id": "s1", "description": "x", "tool": "telepathy"}],
            "confidence": 0.9,
        })
        planner = TaskPlanner(FakeGuard(payload))
        plan = await planner.decompose("x", "", _catalog())
        assert plan.nodes[0].required_tool == "reasoning"

    async def test_forward_dependency_dropped(self) -> None:
        # s1 depends on s2 which is declared later → dep dropped (no cycle).
        payload = json.dumps({
            "steps": [
                {"id": "s1", "description": "a", "tool": "reasoning", "depends_on": ["s2"]},
                {"id": "s2", "description": "b", "tool": "reasoning"},
            ],
            "confidence": 0.8,
        })
        planner = TaskPlanner(FakeGuard(payload))
        plan = await planner.decompose("a then b", "", _catalog())
        assert plan.nodes[0].dependencies == []

    async def test_noisy_confidence_coerced(self) -> None:
        payload = json.dumps({
            "steps": [{"id": "s1", "description": "x", "tool": "reasoning",
                       "confidence": "not-a-number", "cost": "big"}],
            "confidence": "high",
        })
        planner = TaskPlanner(FakeGuard(payload))
        plan = await planner.decompose("x", "", _catalog())
        # Falls back to per-node default confidence (0.7) → overall average.
        assert 0.0 <= plan.overall_confidence <= 1.0
        assert plan.nodes[0].estimated_cost == 1.0


class TestHeuristicFallback:
    async def test_garbage_response_uses_heuristic(self) -> None:
        planner = TaskPlanner(FakeGuard("I cannot help with that, sorry."))
        plan = await planner.decompose("search news and summarize it", "", _catalog())
        assert plan.strategy in ("heuristic", "single")
        assert len(plan.nodes) >= 1

    async def test_graceful_string_uses_heuristic(self) -> None:
        # Simulates LLMGuard's graceful fallback message (won't parse as JSON).
        graceful = "I'm having trouble contacting the language model."
        planner = TaskPlanner(FakeGuard(graceful))
        plan = await planner.decompose("do a and then b", "", _catalog())
        assert plan.strategy == "heuristic"
        assert len(plan.nodes) == 2
        # Sequential dependency chain preserved.
        assert plan.nodes[1].dependencies == [plan.nodes[0].id]

    async def test_no_guard_uses_heuristic(self) -> None:
        planner = TaskPlanner(None)
        plan = await planner.decompose("just explain recursion", "", _catalog())
        assert plan.strategy == "single"
        assert len(plan.nodes) == 1

    async def test_unavailable_guard_uses_heuristic(self) -> None:
        planner = TaskPlanner(FakeGuard("{}", available=False))
        plan = await planner.decompose("explain", "", _catalog())
        assert plan.strategy == "single"

    async def test_empty_goal_yields_empty_plan(self) -> None:
        planner = TaskPlanner(FakeGuard("{}"))
        plan = await planner.decompose("   ", "", _catalog())
        assert plan.is_empty
