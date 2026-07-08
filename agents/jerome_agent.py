"""Jerome — DevOps & Deployment specialist for Jarvis.

Handles system configuration, application deployment, system administration,
and health monitoring with full shell and file-system tool support.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_DEVOPS, Capability
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider
from memory import MemoryService
from tools import ToolExecutionEngine

_logger = logging.getLogger(__name__)


class JeromeAgent(Agent):
    """Agent responsible for devops, deployment, and system administration tasks."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: ToolExecutionEngine | None = None,
    ) -> None:
        super().__init__(
            name="jerome",
            supported_task_types=(
                "devops.configure",
                "devops.deploy",
                "system.admin",
                "devops.monitor",
            ),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Capability]:
        return [CAPABILITY_DEVOPS]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"JeromeAgent cannot handle task type: {task.task_type}",
            )

        try:
            if task.task_type == "devops.configure":
                result = await self._handle_configure(task)
            elif task.task_type == "devops.deploy":
                result = await self._handle_deploy(task)
            elif task.task_type == "system.admin":
                result = await self._handle_admin(task)
            elif task.task_type == "devops.monitor":
                result = await self._handle_monitor(task)
            else:
                result = AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message=f"Unknown task type: {task.task_type}",
                )

            return result
        except Exception:
            _logger.exception("Failed to handle task %s (%s)", task.task_id, task.task_type)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="An unexpected error occurred while handling the task.",
            )

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------

    async def _handle_configure(self, task: AgentTask) -> AgentResult:
        """Configure system or application settings using the shell tool."""
        _logger.info("Handling devops.configure task %s", task.task_id)
        command = task.payload.get("command", "")
        if not command:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No command provided for configuration task.",
            )
        return await self._run_shell(task, command)

    async def _handle_deploy(self, task: AgentTask) -> AgentResult:
        """Deploy applications using the shell tool."""
        _logger.info("Handling devops.deploy task %s", task.task_id)
        command = task.payload.get("command", "")
        if not command:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No command provided for deployment task.",
            )
        return await self._run_shell(task, command)

    async def _handle_admin(self, task: AgentTask) -> AgentResult:
        """Perform system administration tasks using shell and file_system tools."""
        _logger.info("Handling system.admin task %s", task.task_id)
        command = task.payload.get("command", "")
        if not command:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No command provided for administration task.",
            )
        return await self._run_shell(task, command)

    async def _handle_monitor(self, task: AgentTask) -> AgentResult:
        """Monitor system health and performance."""
        _logger.info("Handling devops.monitor task %s", task.task_id)
        command = task.payload.get("command", "")
        if not command:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No command provided for monitoring task.",
            )
        return await self._run_shell(task, command)

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    async def _run_shell(self, task: AgentTask, command: str) -> AgentResult:
        """Execute a shell command via the tool engine and return the result."""
        if self._tool_engine is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No tool engine available to execute the command.",
            )

        try:
            result = await self._tool_engine.execute("shell", command=command)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=result.success,
                message="Command executed successfully." if result.success else "Command failed.",
                data={
                    "output": result.output,
                    "error": result.error,
                    "execution_time_ms": result.execution_time_ms,
                },
            )
        except Exception:
            _logger.exception("Shell command execution failed")
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Shell command execution raised an exception.",
            )
