"""Routing Engine — routes processed intents to the correct agent pipeline.

Routes user requests to the appropriate execution path based on intent
classification, context, and available agents.

For tool execution, the engine:
1. Identifies the specific tool from the user's goal
2. Routes directly to the ToolAgent with tool_name in payload
3. Includes raw output metadata for ResponseSynthesizer
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.contracts import AgentResult, AgentTask
from orchestrator import Orchestrator

from runtime.intent_engine import IntentResult

_logger = logging.getLogger(__name__)

# Pattern: (regex, tool_name, extract_expression)
# Ordered from most specific to least specific to avoid false positives.
_TOOL_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = [
    (re.compile(r"\b(what\s+is\s+the\s+(date|time)|current\s+(date|time)|datetime)\b", re.IGNORECASE), "datetime", False),
    (re.compile(r"\b(create|write|add|make)\s+(a\s+)?note\b", re.IGNORECASE), "notes", False),
    (re.compile(r"\b(read|show|get|open)\s+(the\s+)?note\b", re.IGNORECASE), "notes", False),
    (re.compile(r"\b(list|show|all)\s+(my\s+)?notes\b", re.IGNORECASE), "notes", False),
    (re.compile(r"\b(weather|temperature|forecast)\s+(.+)$", re.IGNORECASE), "weather", False),
    (re.compile(r"\b(generate|create)\s+(a\s+)?(uuid|guid)\b", re.IGNORECASE), "uuid", False),
    (re.compile(r"\b(encode|decode)\s+.+\s+(to|from)\s+base64\b", re.IGNORECASE), "base64", False),
    (re.compile(r"\b(sha256|md5|hash)\s+(of\s+)?(.+)$", re.IGNORECASE), "hash", False),
    (re.compile(r"\b(pretty\s+)?print\s+(json|the\s+json)\b", re.IGNORECASE), "json", False),
    (re.compile(r"\b(system\s+info|system\s+information)\b", re.IGNORECASE), "system_info", False),
    (re.compile(r"\b(screenshot|take\s+a\s+screenshot|capture\s+screen)\b", re.IGNORECASE), "screenshot", False),
    (re.compile(r"\b(run|execute)\s+(a\s+)?(shell|command|terminal)\b", re.IGNORECASE), "shell", False),
    (re.compile(r"\b(clipboard|copy\s+to\s+clipboard)\b", re.IGNORECASE), "clipboard", False),
    (re.compile(r"\b(notify|send\s+notification)\b", re.IGNORECASE), "notification", False),
    (re.compile(r"\b(search|find|look\s+up|google|browse)\s+(.+)$", re.IGNORECASE), "search", False),
    # Calculator: must contain digits or arithmetic operators to avoid matching "what is Python"
    (re.compile(r"(?:calculate|evaluate|compute|solve|math)\s*(.+)$", re.IGNORECASE), "calculator", True),
    (re.compile(r"\bwhat\s+is\s+([\d][\d\s+\-*/().^%]+)$", re.IGNORECASE), "calculator", True),
]


class RoutingEngine:
    """Routes intents to the correct orchestration pipeline.

    Instead of a single if/elif chain in main.py, routing is handled
    as a strategy pattern — each intent type maps to a handler method.

    For tool execution, detects the specific tool from the user's goal
    and routes directly to the ToolAgent with the correct tool_name.
    """

    def __init__(self, orchestrator: Orchestrator | None) -> None:
        self._orchestrator = orchestrator

    async def route(
        self,
        intent: IntentResult,
        goal: str,
    ) -> AgentResult:
        """Route *goal* based on *intent* classification.

        Returns the orchestrator's AgentResult for further processing.
        The result includes tool metadata (tool_name, tool_output) when
        applicable, so the ResponseSynthesizer can format the output.
        """
        if self._orchestrator is None:
            return AgentResult(
                agent_name="runtime",
                task_id="",
                success=False,
                message="Orchestrator is not available.",
                data={"status": "failed", "response": "System is not fully initialized."},
            )

        if intent.requires_conversation:
            return await self._route_conversation(goal)

        if intent.requires_vision:
            return await self._route_vision(goal)

        if intent.requires_browser:
            return await self._route_browser(goal, intent)

        if intent.requires_tool:
            return await self._route_tool(goal, intent)

        if intent.requires_planning:
            return await self._route_plan(goal, intent)

        if intent.requires_knowledge:
            return await self._route_knowledge(goal)

        return await self._route_plan(goal, intent)

    # ------------------------------------------------------------------
    # Tool detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_tool(goal: str) -> tuple[str, dict[str, Any]]:
        """Detect tool name and extract arguments from *goal*.

        Returns:
            Tuple of (tool_name, arguments_dict).
            tool_name is empty string if no tool was detected.
        """
        for pattern, tool_name, extract_expr in _TOOL_PATTERNS:
            match = pattern.search(goal)
            if match:
                if tool_name == "calculator" and extract_expr and match.lastindex and match.lastindex >= 1:
                    expression = match.group(match.lastindex).strip()
                    expression = expression.rstrip("?.,;!")
                    return tool_name, {"expression": expression}
                if tool_name == "search":
                    query = match.group(match.lastindex).strip() if match.lastindex else goal
                    return tool_name, {"query": query}
                if tool_name == "weather":
                    location = match.group(match.lastindex).strip() if match.lastindex else ""
                    return tool_name, {"location": location}
                if tool_name == "hash":
                    data = match.group(match.lastindex).strip() if match.lastindex else ""
                    return tool_name, {"data": data, "algorithm": "sha256"}
                return tool_name, {}
        return "", {}

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    async def _route_conversation(self, goal: str) -> AgentResult:
        """Greetings, thanks, goodbyes — fast path, no agents."""
        return await self._orchestrator.route(
            AgentTask(
                task_type="jarvis.process",
                payload={"goal": goal, "task_type": "conversation"},
            )
        )

    async def _route_vision(self, goal: str) -> AgentResult:
        """Vision tasks — images, OCR, screenshots."""
        return await self._orchestrator.route(
            AgentTask(
                task_type="jarvis.process",
                payload={"goal": goal, "task_type": "plan"},
            )
        )

    async def _route_browser(self, goal: str, intent: IntentResult) -> AgentResult:
        """Browser/search tasks — web navigation, content extraction."""
        payload: dict[str, Any] = {"goal": goal, "task_type": "plan"}
        if intent.secondary:
            payload["secondary_intents"] = [s.label for s in intent.secondary]
        return await self._orchestrator.route(
            AgentTask(task_type="jarvis.process", payload=payload)
        )

    # ------------------------------------------------------------------
    # Tool execution — detect tool and route directly to ToolAgent
    # ------------------------------------------------------------------

    async def _route_tool(self, goal: str, intent: IntentResult) -> AgentResult:
        """Tool execution tasks.

        Detects the specific tool from the goal and routes directly to
        the ToolAgent with the correct tool_name and arguments.
        """
        tool_name, arguments = self._detect_tool(goal)

        if not tool_name:
            # Fallback: let JarvisPrimeAgent figure it out via planner
            _logger.info("No tool detected from goal, routing to JarvisPrimeAgent: %s", goal[:60])
            return await self._orchestrator.route(
                AgentTask(
                    task_type="jarvis.process",
                    payload={"goal": goal, "task_type": "tool.execute"},
                )
            )

        _logger.info("Detected tool '%s' from goal: %s", tool_name, goal[:60])

        return await self._orchestrator.route(
            AgentTask(
                task_type="tool.execute",
                payload={
                    "goal": goal,
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
            )
        )

    # ------------------------------------------------------------------
    # Plan & Knowledge
    # ------------------------------------------------------------------

    async def _route_plan(self, goal: str, intent: IntentResult) -> AgentResult:
        """Planning/workflow tasks — coding, security, devops, agent-specific."""
        payload: dict[str, Any] = {"goal": goal, "task_type": "plan"}
        payload["intent_label"] = intent.primary.label
        payload["intent_confidence"] = intent.primary.confidence
        if intent.secondary:
            payload["secondary_intents"] = [s.label for s in intent.secondary]
        return await self._orchestrator.route(
            AgentTask(task_type="jarvis.process", payload=payload)
        )

    async def _route_knowledge(self, goal: str) -> AgentResult:
        """Knowledge questions — LLM direct, no specialists."""
        return await self._orchestrator.route(
            AgentTask(
                task_type="jarvis.process",
                payload={"goal": goal, "task_type": "knowledge"},
            )
        )
