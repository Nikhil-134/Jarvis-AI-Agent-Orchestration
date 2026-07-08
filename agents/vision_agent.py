"""Vision agent — Jarvis's Computer Vision specialist.

Handles image analysis, OCR, screenshot capture (stub), and natural
language image description.  Delegates to an LLM or falls back to
rule-based responses when no vision provider is available.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_VISION
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider, ChatSession, LLMError, LLMResponse
from memory import MemoryService

_logger = logging.getLogger(__name__)


class VisionAgent(Agent):
    """Agent responsible for computer vision tasks.

    Supports analyzing images, extracting text via OCR, capturing
    screenshots, and generating natural-language descriptions of image
    contents.

    When an :class:`BaseLLMProvider` is available, analysis and
    description tasks are delegated to the LLM with vision capabilities.
    OCR is attempted via the LLM first, falling back to a stub response.
    Screenshot capture is always a stub (not yet implemented).
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(name="vision", supported_task_types=(
            "vision.analyze",
            "vision.ocr",
            "vision.screenshot",
            "vision.describe",
        ))
        self._memory_service = memory_service
        self._tool_engine = tool_engine
        self._chat_session = (
            ChatSession(llm_provider, system_prompt="You are Jarvis, a vision-capable AI assistant.")
            if llm_provider
            else None
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> list[Any]:
        return [CAPABILITY_VISION]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"VisionAgent cannot handle task type: {task.task_type}",
            )

        image_path: str | None = task.payload.get("path")
        prompt: str | None = task.payload.get("prompt")

        try:
            if task.task_type == "vision.analyze":
                result = await self._analyze(image_path, prompt)
            elif task.task_type == "vision.ocr":
                result = await self._ocr(image_path)
            elif task.task_type == "vision.screenshot":
                result = self._screenshot()
            elif task.task_type == "vision.describe":
                result = await self._describe(image_path, prompt)
            else:
                result = AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message=f"Unknown vision task type: {task.task_type}",
                )
                return result

            return result
        except Exception:
            _logger.exception("Vision task '%s' failed", task.task_type)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Vision task '{task.task_type}' encountered an unexpected error.",
            )

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------

    async def _analyze(self, image_path: str | None, prompt: str | None) -> AgentResult:
        """Analyze an image and return a structured description / analysis."""
        if not image_path:
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=False,
                message="No image path provided for analysis.",
            )

        if self._chat_session is None:
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=True,
                message="Image analysis completed (fallback).",
                data={
                    "analysis": f"Received request to analyze image at '{image_path}'. "
                    "No vision LLM is configured — please provide an LLM provider for detailed analysis.",
                    "path": image_path,
                },
            )

        try:
            user_msg = f"Analyze this image: {image_path}"
            if prompt:
                user_msg += f"\n\nAdditional context: {prompt}"

            response: LLMResponse = await self._chat_session.send(user_msg)
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=True,
                message="Image analysis completed.",
                data={
                    "analysis": response.content,
                    "path": image_path,
                },
            )
        except LLMError:
            _logger.exception("LLM image analysis failed for '%s'", image_path)
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=False,
                message="Image analysis failed due to an LLM error.",
                data={"path": image_path},
            )

    async def _ocr(self, image_path: str | None) -> AgentResult:
        """Extract text from an image via OCR."""
        if not image_path:
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=False,
                message="No image path provided for OCR.",
            )

        if self._chat_session is None:
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=True,
                message="OCR completed (fallback).",
                data={
                    "text": f"OCR requested for '{image_path}'. "
                    "No vision LLM is configured — unable to extract text.",
                    "path": image_path,
                },
            )

        try:
            response: LLMResponse = await self._chat_session.send(
                f"Extract all text from this image using OCR: {image_path}"
            )
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=True,
                message="OCR completed.",
                data={
                    "text": response.content,
                    "path": image_path,
                },
            )
        except LLMError:
            _logger.exception("LLM OCR failed for '%s'", image_path)
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=False,
                message="OCR failed due to an LLM error.",
                data={"path": image_path},
            )

    def _screenshot(self) -> AgentResult:
        """Take a screenshot (stub — not yet implemented)."""
        _logger.warning("Screenshot capture is not yet implemented")
        return AgentResult(
            agent_name=self.name,
            task_id="",
            success=False,
            message="Screenshot capture is not yet implemented.",
            data={"status": "not_implemented"},
        )

    async def _describe(self, image_path: str | None, prompt: str | None) -> AgentResult:
        """Generate a natural language description of image contents."""
        if not image_path:
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=False,
                message="No image path provided for description.",
            )

        if self._chat_session is None:
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=True,
                message="Image description completed (fallback).",
                data={
                    "description": f"Received request to describe image at '{image_path}'. "
                    "No vision LLM is configured — unable to generate a description.",
                    "path": image_path,
                },
            )

        try:
            user_msg = f"Describe the contents of this image in natural language: {image_path}"
            if prompt:
                user_msg += f"\n\n{prompt}"

            response: LLMResponse = await self._chat_session.send(user_msg)
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=True,
                message="Image description completed.",
                data={
                    "description": response.content,
                    "path": image_path,
                },
            )
        except LLMError:
            _logger.exception("LLM image description failed for '%s'", image_path)
            return AgentResult(
                agent_name=self.name,
                task_id="",
                success=False,
                message="Image description failed due to an LLM error.",
                data={"path": image_path},
            )
