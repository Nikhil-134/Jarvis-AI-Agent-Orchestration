"""Veronica agent — Jarvis's Code Engineering specialist.

Supports code generation, review, refactoring, and analysis tasks
using optional LLM, memory, and tool support.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_CODE_ENGINEERING, Capability
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider, ChatSession, LLMError, LLMResponse
from memory import MemoryService

_logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are Veronica, Jarvis's Code Engineering specialist. "
    "You generate, review, refactor, and analyze source code with precision and clarity."
)


class VeronicaAgent(Agent):
    """Agent responsible for code engineering tasks.

    Handles code generation, review, refactoring, and analysis.
    When an LLM provider is available, tasks are processed through
    a chat session.  A ``file_system`` tool in the tool engine is
    used to write generated files.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="veronica",
            supported_task_types=(
                "code.generate",
                "code.review",
                "code.refactor",
                "code.analyze",
            ),
        )
        self._memory_service = memory_service
        self._tool_engine = tool_engine
        self._chat_session = (
            ChatSession(llm_provider, system_prompt=_SYSTEM_PROMPT)
            if llm_provider
            else None
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> list[Capability]:
        """Return the capabilities this agent provides."""
        return [CAPABILITY_CODE_ENGINEERING]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"VeronicaAgent cannot handle task type: {task.task_type}",
            )

        _logger.info("Handling task '%s' (id=%s)", task.task_type, task.task_id)

        try:
            if task.task_type == "code.generate":
                return await self._handle_generate(task)
            if task.task_type == "code.review":
                return await self._handle_review(task)
            if task.task_type == "code.refactor":
                return await self._handle_refactor(task)
            if task.task_type == "code.analyze":
                return await self._handle_analyze(task)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Unhandled task type: {task.task_type}",
            )
        except Exception:
            _logger.exception("Task '%s' (id=%s) failed", task.task_type, task.task_id)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Internal error while processing task.",
                data={"status": "error"},
            )

    # ------------------------------------------------------------------
    # code.generate
    # ------------------------------------------------------------------

    async def _handle_generate(self, task: AgentTask) -> AgentResult:
        """Generate source code from a specification.

        Payload fields:
            specification (str):  Description of the code to generate.
            language (str, optional):  Target programming language.
            output_path (str, optional):  File path to write the result to.
        """
        specification = str(task.payload.get("specification", ""))
        language = str(task.payload.get("language", ""))
        output_path = str(task.payload.get("output_path", "")) or None

        if not specification:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No specification provided in payload.",
                data={"status": "error"},
            )

        if self._chat_session is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Code generation requires an LLM provider.",
                data={"status": "error"},
            )

        prompt_parts = [f"Generate source code for the following specification:\n\n{specification}"]
        if language:
            prompt_parts.append(f"\nTarget language: {language}")
        prompt_parts.append(
            "\nProvide only the source code in a single code block. "
            "Do not include explanations unless asked."
        )

        prompt = "\n".join(prompt_parts)

        try:
            _logger.info("Generating code (language=%s, language=%s)", language, bool(language))
            response: LLMResponse = await self._chat_session.send(prompt)
            code = self._extract_code_block(response.content) or response.content

            # Write to file if a tool engine with file_system is available
            if output_path and self._tool_engine is not None:
                try:
                    await self._tool_engine.execute(
                        "file_system",
                        operation="write",
                        path=output_path,
                        content=code,
                    )
                    _logger.info("Wrote generated code to '%s'", output_path)
                except Exception:
                    _logger.exception("Failed to write generated code to '%s'", output_path)

            if self._memory_service is not None:
                try:
                    await self._memory_service.store_fact(
                        f"Generated code for: {specification[:200]}", importance=0.5
                    )
                except Exception:
                    _logger.debug("Failed to store code generation fact in memory")

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=True,
                message="Code generated successfully.",
                data={
                    "status": "completed",
                    "code": code,
                    "language": language,
                    "output_path": output_path,
                },
            )
        except LLMError:
            _logger.exception("Code generation LLM call failed")
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Code generation failed due to an LLM error.",
                data={"status": "error"},
            )

    # ------------------------------------------------------------------
    # code.review
    # ------------------------------------------------------------------

    async def _handle_review(self, task: AgentTask) -> AgentResult:
        """Analyze provided code for issues, patterns, and improvements.

        Payload fields:
            code (str):  The source code to review.
            language (str, optional):  Programming language of the code.
        """
        code = str(task.payload.get("code", ""))
        language = str(task.payload.get("language", ""))

        if not code:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No code provided for review in payload.",
                data={"status": "error"},
            )

        result = await self._llm_analysis(
            "code review",
            code,
            language,
            (
                "Review the following code for:\n"
                "1. Bugs and logic errors\n"
                "2. Code quality and style issues\n"
                "3. Performance concerns\n"
                "4. Security vulnerabilities\n"
                "5. Maintainability and readability improvements\n\n"
                "Provide a structured report with severity levels (critical, warning, suggestion)."
            ),
        )
        return result

    # ------------------------------------------------------------------
    # code.refactor
    # ------------------------------------------------------------------

    async def _handle_refactor(self, task: AgentTask) -> AgentResult:
        """Suggest and apply refactoring to provided code.

        Payload fields:
            code (str):  The source code to refactor.
            language (str, optional):  Programming language of the code.
            goal (str, optional):  Specific refactoring goal (e.g. "extract method",
                "improve performance", "reduce duplication").
            output_path (str, optional):  File path to write the result to.
        """
        code = str(task.payload.get("code", ""))
        language = str(task.payload.get("language", ""))
        goal = str(task.payload.get("goal", "general improvement"))
        output_path = str(task.payload.get("output_path", "")) or None

        if not code:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No code provided for refactoring in payload.",
                data={"status": "error"},
            )

        prompt = (
            f"Refactor the following code with the goal: {goal}.\n\n"
            "Provide the refactored code in a single code block "
            "and a brief summary of the changes made."
        )

        result = await self._llm_analysis("code refactoring", code, language, prompt)

        # Write refactored code to file if requested
        if result.success and output_path and self._tool_engine is not None:
            refactored_code = (result.data or {}).get("code", "")
            if refactored_code:
                try:
                    await self._tool_engine.execute(
                        "file_system",
                        operation="write",
                        path=output_path,
                        content=refactored_code,
                    )
                    _logger.info("Wrote refactored code to '%s'", output_path)
                except Exception:
                    _logger.exception("Failed to write refactored code to '%s'", output_path)

        return result

    # ------------------------------------------------------------------
    # code.analyze
    # ------------------------------------------------------------------

    async def _handle_analyze(self, task: AgentTask) -> AgentResult:
        """Analyze code structure, complexity, dependencies.

        Payload fields:
            code (str):  The source code to analyze.
            language (str, optional):  Programming language of the code.
        """
        code = str(task.payload.get("code", ""))
        language = str(task.payload.get("language", ""))

        if not code:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No code provided for analysis in payload.",
                data={"status": "error"},
            )

        result = await self._llm_analysis(
            "code analysis",
            code,
            language,
            (
                "Analyze the following code for:\n"
                "1. Overall structure and architecture\n"
                "2. Cyclomatic complexity and cognitive complexity\n"
                "3. Dependency graph (imports, external libraries)\n"
                "4. Code metrics (lines, functions, classes)\n"
                "5. Potential bottlenecks and technical debt\n\n"
                "Provide a structured analysis report."
            ),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _llm_analysis(
        self,
        analysis_type: str,
        code: str,
        language: str,
        instruction: str,
    ) -> AgentResult:
        """Run a generic code analysis through the LLM and return the result."""
        if self._chat_session is None:
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=False,
                message=f"{analysis_type} requires an LLM provider.",
                data={"status": "error"},
            )

        prompt_parts = [instruction]
        if language:
            prompt_parts.append(f"\nLanguage: {language}")
        prompt_parts.append(f"\n```{language}\n{code}\n```")

        prompt = "\n".join(prompt_parts)

        try:
            _logger.info("Performing %s on %d characters of code", analysis_type, len(code))
            response: LLMResponse = await self._chat_session.send(prompt)
        except LLMError:
            _logger.exception("%s LLM call failed", analysis_type)
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=False,
                message=f"{analysis_type} failed due to an LLM error.",
                data={"status": "error"},
            )

        return AgentResult(
            agent_name=self.name,
            task_id="",
            success=True,
            message=f"{analysis_type} completed.",
            data={
                "status": "completed",
                "analysis": response.content,
                "code": code,
                "language": language,
            },
        )

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """Extract the first fenced code block from *text*.

        Returns ``None`` when no fenced code block is found.
        """
        for line in text.splitlines():
            if line.startswith("```"):
                start = text.index(line) + len(line)
                end = text.index("```", start)
                return text[start:end].strip()
        return None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, object]:
        base = await super().health_check()
        base["capabilities"] = [c.name for c in self.capabilities]
        base["has_llm"] = self._chat_session is not None
        base["has_tool_engine"] = self._tool_engine is not None
        base["has_memory_service"] = self._memory_service is not None
        return base
