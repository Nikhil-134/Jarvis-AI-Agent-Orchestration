"""Fallback Engine — graceful degradation when components fail.

Provides tiered fallback strategies so that failures in any layer
produce user-friendly responses instead of tracebacks.
"""

from __future__ import annotations

import logging

from agents.contracts import AgentResult

_logger = logging.getLogger(__name__)

_FALLBACK_MESSAGES = {
    "llm_timeout": (
        "I'm having trouble contacting the language model right now. "
        "Please check that your LLM provider (e.g., Ollama) is running and try again."
    ),
    "orchestrator_unavailable": (
        "The system is still initializing. Please wait a moment and try again."
    ),
    "tool_failed": (
        "I wasn't able to complete that operation. "
        "Please try again or rephrase your request."
    ),
    "agent_unavailable": (
        "The specialist agent for that task is not available right now."
    ),
    "memory_unavailable": (
        "I'm having trouble accessing my memory system. "
        "I'll try to answer based on my general knowledge."
    ),
    "unknown": (
        "I encountered an unexpected issue. "
        "Please try again or rephrase your request."
    ),
}


class FallbackEngine:
    """Provides graceful fallback responses when the primary path fails.

    Usage::

        engine = FallbackEngine()
        result = engine.on_llm_error("timeout", original_goal="What is AI?")
        # Returns a graceful AgentResult with no traceback
    """

    def on_llm_error(
        self,
        error_type: str = "unknown",
        original_goal: str = "",
        details: str = "",
    ) -> AgentResult:
        return AgentResult(
            agent_name="fallback",
            task_id="",
            success=True,
            message=_FALLBACK_MESSAGES.get(error_type, _FALLBACK_MESSAGES["unknown"]),
            data={
                "status": "completed",
                "response": _FALLBACK_MESSAGES.get(error_type, _FALLBACK_MESSAGES["unknown"]),
                "fallback_reason": error_type,
                "memory_enriched": False,
                "memory_count": 0,
            },
        )

    def on_orchestrator_error(
        self,
        error: str = "",
        original_goal: str = "",
    ) -> AgentResult:
        response = _FALLBACK_MESSAGES["orchestrator_unavailable"]
        return AgentResult(
            agent_name="fallback",
            task_id="",
            success=True,
            message=response,
            data={
                "status": "completed",
                "response": response,
                "fallback_reason": "orchestrator_unavailable",
            },
        )

    def on_tool_error(
        self,
        tool_name: str = "",
        error: str = "",
    ) -> AgentResult:
        msg = _FALLBACK_MESSAGES["tool_failed"]
        if tool_name:
            msg = f"I wasn't able to complete the {tool_name} operation. {_FALLBACK_MESSAGES['tool_failed']}"
        return AgentResult(
            agent_name="fallback",
            task_id="",
            success=True,
            message=msg,
            data={
                "status": "completed",
                "response": msg,
                "fallback_reason": "tool_failed",
            },
        )

    def on_exception(
        self,
        exc: Exception,
        original_goal: str = "",
    ) -> AgentResult:
        _logger.exception("Unhandled exception in runtime processing")
        msg = _FALLBACK_MESSAGES["unknown"]
        return AgentResult(
            agent_name="fallback",
            task_id="",
            success=True,
            message=msg,
            data={
                "status": "completed",
                "response": msg,
                "fallback_reason": "exception",
            },
        )

    def wrap(self, result: AgentResult | None, error_type: str = "unknown") -> AgentResult:
        """Wrap a failed result with a fallback message if needed."""
        if result is not None and result.success:
            return result
        return self.on_llm_error(error_type)
