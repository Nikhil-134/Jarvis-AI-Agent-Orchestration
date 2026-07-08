"""Pepper agent — user experience and interaction."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_USER_EXPERIENCE
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider
from memory import MemoryService

_logger = logging.getLogger(__name__)


class PepperAgent(Agent):
    """Agent responsible for notifications, display, interaction, and speech."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="pepper",
            supported_task_types=("ux.notify", "ux.display", "ux.interact", "ux.speak"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Any]:
        return [CAPABILITY_USER_EXPERIENCE]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"PepperAgent cannot handle task type: {task.task_type}",
            )

        match task.task_type:
            case "ux.notify":
                return await self._notify(task)
            case "ux.display":
                return await self._display(task)
            case "ux.interact":
                return await self._interact(task)
            case "ux.speak":
                return await self._speak(task)
            case _:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message=f"Unknown task type: {task.task_type}",
                )

    async def _notify(self, task: AgentTask) -> AgentResult:
        title = task.payload.get("title", "Notification")
        message = task.payload.get("message", "")
        level = task.payload.get("level", "info")
        _logger.info("Sending notification: title=%s level=%s", title, level)
        try:
            import platform
            if platform.system() == "Windows":
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, message, title, 0)
        except Exception:
            _logger.warning("Desktop notification not available, logging instead")
            _logger.info("NOTIFICATION [%s] %s: %s", level, title, message)
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Notification sent.",
            data={"title": title, "message": message, "level": level},
        )

    async def _display(self, task: AgentTask) -> AgentResult:
        content = task.payload.get("content", "")
        format_type = task.payload.get("format", "text")
        _logger.info("Displaying content (format=%s): %s", format_type, content[:100])
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Content displayed.",
            data={"content": content, "format": format_type},
        )

    async def _interact(self, task: AgentTask) -> AgentResult:
        prompt = task.payload.get("prompt", "")
        options = task.payload.get("options", [])
        _logger.info("Handling interactive input: prompt=%s options=%s", prompt, options)
        response = task.payload.get("response", "")
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Interaction handled.",
            data={"prompt": prompt, "options": options, "response": response},
        )

    async def _speak(self, task: AgentTask) -> AgentResult:
        text = task.payload.get("text", "")
        _logger.info("ux.speak requested (%d chars) — delegating to voice subsystem", len(text))
        # PepperAgent has no TTS backend. Speech synthesis is owned by the voice
        # subsystem (VoiceAgent / VoicePipeline); do not fake a spoken result.
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=False,
            message="Speech synthesis is handled by the voice subsystem (VoiceAgent), "
            "not PepperAgent.",
            data={"text": text, "status": "unavailable"},
        )
