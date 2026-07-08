"""Runtime Response Composer — final response assembly with formatting guarantees.

Every response passes through this component. Responsibilities:
- Markdown formatting
- No JSON, tracebacks, internal prompts, or function calls in output
- Consistent conversational style
- Proper error message formatting
"""

from __future__ import annotations

import logging

from agents.contracts import AgentResult
from agents.response_composer import ResponseComposer as LegacyResponseComposer
from core.response_guards import INTERNAL_MESSAGE_MARKERS, is_user_safe

from runtime.fallback_engine import FallbackEngine
from runtime.response_formatter import ResponseFormatter

_logger = logging.getLogger(__name__)

# A graceful, user-facing message for any failure whose only detail is an
# internal error. The real (internal) message is logged, never shown.
_GRACEFUL_FAILURE = (
    "I wasn't able to complete that just now. "
    "Could you rephrase it or try again in a moment?"
)

# The internal-machinery marker list now lives in ``core.response_guards`` so
# the runtime composer and the planning ResponseVerifier share one source of
# truth. Re-exported under the original name for backward compatibility.
_INTERNAL_MESSAGE_MARKERS = INTERNAL_MESSAGE_MARKERS


def _is_user_facing(text: str) -> bool:
    """Return True if *text* is safe to show a user (not internal machinery)."""
    return is_user_safe(text)


class RuntimeResponseComposer:
    """Wraps the legacy ResponseComposer with runtime-level formatting guarantees.

    Every response goes through:
    1. Legacy merge (multi-agent fusion)
    2. Artifact stripping (JSON, tracebacks, prompts)
    3. Markdown formatting
    4. Clean response validation
    """

    def __init__(
        self,
        legacy_composer: LegacyResponseComposer | None = None,
        formatter: ResponseFormatter | None = None,
        fallback: FallbackEngine | None = None,
    ) -> None:
        self._legacy = legacy_composer
        self._formatter = formatter or ResponseFormatter()
        self._fallback = fallback or FallbackEngine()

    async def compose(
        self,
        goal: str,
        agent_result: AgentResult,
        *,
        personality_response: str | None = None,
    ) -> str:
        """Compose the final user-facing response.

        If a *personality_response* is provided (for greetings, jokes, etc.),
        it takes priority over the agent result.
        """
        if personality_response:
            return self._finalize(personality_response)

        if not agent_result.success:
            # NEVER surface a raw internal failure message (e.g. "memory_id is
            # required", "Browser automation requires a browser engine (not
            # installed)"). Prefer a user-facing response the agent produced;
            # otherwise degrade gracefully and log the real reason internally.
            candidate = ""
            if agent_result.data:
                candidate = agent_result.data.get("response") or ""
            if _is_user_facing(candidate):
                return self._finalize(candidate)
            _logger.info(
                "Suppressed internal failure message from '%s': %s",
                agent_result.agent_name, agent_result.message,
            )
            return self._finalize(_GRACEFUL_FAILURE)

        raw_response = ""
        if agent_result.data:
            raw_response = agent_result.data.get("response") or agent_result.data.get("output") or ""

        # Only fall back to the agent's message if it is genuinely user-facing;
        # internal status strings must not leak into the reply.
        if not raw_response and _is_user_facing(agent_result.message):
            raw_response = agent_result.message

        if not raw_response:
            raw_response = "I've completed your request."

        return self._finalize(raw_response)

    async def compose_multi(
        self,
        goal: str,
        results: list[AgentResult],
    ) -> str:
        """Compose a response from multiple agent results (workflow output)."""
        if self._legacy and len(results) > 1:
            merged = await self._legacy.merge(goal, results)
            return self._finalize(merged)

        if len(results) == 1:
            return await self.compose(goal, results[0])

        if not results:
            return self._finalize("I received your request.")

        parts: list[str] = []
        for r in results:
            text = ""
            if r.data:
                text = r.data.get("response") or r.data.get("output") or ""
            if not text:
                text = r.message or ""
            if text:
                parts.append(self._formatter.format_tool_summary(r.agent_name, text))

        return self._finalize("\n\n".join(parts) if parts else "All tasks completed.")

    def _finalize(self, text: str) -> str:
        """Apply final formatting and safety checks."""
        formatted = self._formatter.format(text)

        if not self._formatter.is_response_clean(formatted):
            _logger.warning("Response contains internal artifacts, restripping")
            formatted = self._formatter.format(formatted)

        return formatted
