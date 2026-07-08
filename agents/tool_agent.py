"""Tool agent — executes tool calls via the ToolExecutionEngine."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask
from memory import MemoryService
from tools import ToolExecutionEngine, ToolManager, ToolRegistry
from tools.builtins import register_all_builtins
from tools.engine import ToolResult

_logger = logging.getLogger(__name__)


class ToolAgent(Agent):
    """Agent responsible for tool execution.

    Supports task type ``tool.execute``.  The task payload must contain
    ``tool_name`` (str) and optionally ``arguments`` (dict).

    When no engine is provided a default is created with all built-in
    tools registered.

    When a :class:`MemoryService` is provided, important tool results
    are stored for future retrieval.
    """

    def __init__(
        self,
        engine: ToolExecutionEngine | None = None,
        auto_register_builtins: bool = True,
        memory_service: MemoryService | None = None,
        store_results: bool = True,
    ) -> None:
        super().__init__(name="tool", supported_task_types=("tool.execute",))
        if engine is not None:
            self._engine = engine
        else:
            registry = ToolRegistry()
            if auto_register_builtins:
                register_all_builtins(registry)
            self._engine = ToolExecutionEngine(registry=registry)
        self._memory_service = memory_service
        self._store_results = store_results

    @property
    def engine(self) -> ToolExecutionEngine:
        return self._engine

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"ToolAgent cannot handle task type: {task.task_type}",
            )

        tool_name = str(task.payload.get("tool_name", ""))
        arguments: dict[str, Any] = dict(task.payload.get("arguments", {}))
        run_id: str = task.payload.get("task_id", task.task_id)

        if not tool_name:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No tool_name provided in payload.",
                data={"status": "error", "tool_name": ""},
            )

        _logger.info(
            "ToolAgent executing '%s' (run_id=%s, args=%s)",
            tool_name,
            run_id,
            arguments,
        )

        result: ToolResult = await self._engine.execute(tool_name, **arguments)

        _logger.info(
            "ToolAgent '%s' completed in %.1f ms (success=%s)",
            tool_name,
            result.execution_time_ms,
            result.success,
        )

        if self._store_results and self._memory_service is not None and result.success and result.output:
            try:
                content = f"Tool '{tool_name}' returned: {result.output[:500]}"
                await self._memory_service.store_fact(content, importance=0.5)
            except Exception:
                _logger.debug("Failed to store tool result in memory")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=result.success,
            message=result.output if result.success else (result.error or "Tool execution failed."),
            data={
                "status": "completed" if result.success else "error",
                "tool_name": tool_name,
                "output": result.output,
                "execution_time_ms": result.execution_time_ms,
                "error": result.error,
            },
        )

    async def health_check(self) -> dict[str, object]:
        base = await super().health_check()
        base["tool_count"] = self._engine.registry.count
        base["tool_names"] = [s.name for s in self._engine.registry.list_specs()]
        return base
