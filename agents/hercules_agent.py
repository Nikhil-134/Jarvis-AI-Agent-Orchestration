"""Hercules agent — computation and data processing."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_COMPUTATION
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider
from memory import MemoryService

_logger = logging.getLogger(__name__)


class HerculesAgent(Agent):
    """Agent responsible for computational tasks and data transformation."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="hercules",
            supported_task_types=("compute.process", "data.transform", "batch.execute"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Any]:
        return [CAPABILITY_COMPUTATION]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"HerculesAgent cannot handle task type: {task.task_type}",
            )

        match task.task_type:
            case "compute.process":
                return await self._process(task)
            case "data.transform":
                return await self._transform(task)
            case "batch.execute":
                return await self._batch_execute(task)
            case _:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message=f"Unknown task type: {task.task_type}",
                )

    async def _process(self, task: AgentTask) -> AgentResult:
        operation = task.payload.get("operation", "")
        input_data = task.payload.get("input", "")
        parameters = task.payload.get("parameters", {})
        _logger.info("Processing computation: operation=%s", operation)
        if self._tool_engine is None or not hasattr(self._tool_engine, "execute"):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Computation requires a tool engine to execute the operation.",
                data={"status": "unavailable", "operation": operation},
            )
        try:
            result = await self._tool_engine.execute(operation, input=input_data, **parameters)
            output = result.output if result.success else (result.error or "unknown error")
            success = result.success
        except Exception as exc:
            _logger.exception("Computation failed")
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Computation failed: {exc}",
            )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=success,
            message="Computation completed." if success else "Computation failed.",
            data={"operation": operation, "output": output},
        )

    async def _transform(self, task: AgentTask) -> AgentResult:
        input_data = task.payload.get("input", "")
        target_format = task.payload.get("target_format", "json")
        _logger.info("Transforming data to format: %s", target_format)
        transformed = str(input_data)
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Data transformation completed.",
            data={"input_format": type(input_data).__name__, "target_format": target_format, "output": transformed},
        )

    async def _batch_execute(self, task: AgentTask) -> AgentResult:
        operations = task.payload.get("operations", [])
        _logger.info("Executing batch of %d operations", len(operations))

        if not operations:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No operations provided for batch execution.",
                data={"status": "error"},
            )

        if self._tool_engine is None or not hasattr(self._tool_engine, "execute"):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Batch execution requires a tool engine.",
                data={"status": "unavailable", "total": len(operations)},
            )

        # Execute each operation for real through the tool engine and report the
        # true per-operation outcome. Runs sequentially (no parallelism claimed).
        results: list[dict[str, Any]] = []
        succeeded = 0
        for op in operations:
            name = op.get("operation", "")
            params = op.get("parameters", {})
            try:
                res = await self._tool_engine.execute(name, **params)
                ok = res.success
                output = res.output if res.success else (res.error or "operation failed")
            except Exception as exc:  # noqa: BLE001 - report per-op failure, keep going
                _logger.exception("Batch operation '%s' failed", name)
                ok = False
                output = str(exc)
            if ok:
                succeeded += 1
            results.append({"operation": name, "success": ok, "output": output})

        all_ok = succeeded == len(results)
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=all_ok,
            message=f"Batch processing finished: {succeeded}/{len(results)} operation(s) succeeded.",
            data={"status": "completed" if all_ok else "error", "results": results, "total": len(results)},
        )
