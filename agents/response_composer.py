"""Response composer — merges multi-agent outputs into one coherent response."""

from __future__ import annotations

import logging
from typing import Any

from agents.contracts import AgentResult
from llm import BaseLLMProvider, PromptManager

_logger = logging.getLogger(__name__)


class ResponseComposer:
    """Merges outputs from multiple specialist agents into one
    coherent natural-language response.

    Handles single-agent passthrough, LLM-based fusion for
    multi-agent results, and rule-based fallback merging.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager or PromptManager()

    async def merge(
        self,
        goal: str,
        results: list[AgentResult],
    ) -> str:
        """Merge *results* into a single response string.

        Args:
            goal: The original user request.
            results: All agent results (successful or failed).

        Returns:
            A single coherent response string.
        """
        if not results:
            return "I received your request. How can I assist you further?"

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if not successful:
            return self._format_all_failed(failed)

        if len(successful) == 1:
            return self._extract_single(successful[0])

        return await self._fuse_responses(goal, successful, failed)

    def _extract_single(self, result: AgentResult) -> str:
        """Extract a response string from a single successful result."""
        data = result.data or {}
        response = data.get("response") or data.get("output") or result.message
        if response and isinstance(response, str) and response.strip():
            return response.strip()
        return "Task completed."

    def _format_all_failed(self, failed: list[AgentResult]) -> str:
        """Format a response when all agents failed."""
        reasons: list[str] = []
        for r in failed:
            reason = r.data.get("error", r.message) if r.data else r.message
            if reason:
                reasons.append(f"  - {r.agent_name}: {reason}")
        if reasons:
            return "I couldn't complete your request. The following issues occurred:\n" + "\n".join(reasons)
        return "I ran into an issue processing your request. Please try again."

    async def _fuse_responses(
        self,
        goal: str,
        successful: list[AgentResult],
        failed: list[AgentResult],
    ) -> str:
        """Fuse multiple successful responses into one coherent response."""
        if self._llm_provider is None:
            return self._simple_merge(successful)

        combined = "\n\n".join(
            self._format_agent_result_for_llm(r)
            for r in successful
        )

        if failed:
            combined += "\n\nFailed agents:\n" + "\n".join(
                f"  - {r.agent_name}: {r.data.get('error', r.message) if r.data else r.message}"
                for r in failed
            )

        prompt = self._prompt_manager.render(
            "merger",
            goal=goal,
            agent_results=combined,
        )

        try:
            merged = await self._llm_provider.generate_text(
                prompt,
                system_prompt=(
                    "You are a response fusion engine. Combine the "
                    "results from multiple specialist agents into one "
                    "coherent, natural, conversational response. "
                    "Eliminate redundancy and resolve conflicts."
                ),
            )
            return merged
        except Exception:
            _logger.exception("LLM merge failed, using simple merge")
            return self._simple_merge(successful)

    def _simple_merge(self, results: list[AgentResult]) -> str:
        """Merge results without an LLM by concatenating agent outputs."""
        lines: list[str] = []
        for r in results:
            text = self._extract_single(r)
            if text:
                lines.append(f"**{r.agent_name.title()}**: {text}")
        return "\n\n".join(lines) if lines else "All tasks completed."

    @staticmethod
    def _format_agent_result_for_llm(result: AgentResult) -> str:
        """Format an agent result for LLM fusion prompt."""
        data = result.data or {}
        response = data.get("response") or data.get("output") or result.message
        return f"Agent '{result.agent_name}' result:\n{response}"
