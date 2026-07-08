"""Athena agent — Strategic Planning specialist."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_STRATEGY, Capability
from agents.contracts import AgentResult, AgentTask

_logger = logging.getLogger(__name__)


class AthenaAgent(Agent):
    """Agent responsible for strategic planning, task decomposition, and workflow design."""

    def __init__(
        self,
        llm_provider: Any | None = None,
        memory_service: Any | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="athena",
            supported_task_types=("strategy.plan", "task.decompose", "workflow.design"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Capability]:
        return [CAPABILITY_STRATEGY]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"AthenaAgent cannot handle task type: {task.task_type}",
            )

        handlers = {
            "strategy.plan": self._handle_strategy_plan,
            "task.decompose": self._handle_task_decompose,
            "workflow.design": self._handle_workflow_design,
        }

        handler = handlers.get(task.task_type)
        if handler is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Unknown task type: {task.task_type}",
            )

        try:
            return await handler(task)
        except Exception:
            _logger.exception("AthenaAgent failed to handle task %s", task.task_id)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Internal error while processing task.",
            )

    async def _handle_strategy_plan(self, task: AgentTask) -> AgentResult:
        goal = task.payload.get("goal", "")
        constraints = task.payload.get("constraints", [])
        _logger.info("Strategic planning for goal=%s", goal)

        steps = []
        if goal:
            lines = goal.split(".")
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line:
                    steps.append(f"Step {i}: {line}")

        if not steps:
            steps = ["Step 1: Define the objective", "Step 2: Gather requirements", "Step 3: Execute plan"]

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Strategic plan generated.",
            data={
                "status": "completed",
                "goal": goal,
                "steps": steps,
                "constraints": constraints,
                "step_count": len(steps),
            },
        )

    async def _handle_task_decompose(self, task: AgentTask) -> AgentResult:
        description = task.payload.get("description", "")
        parent_id = task.payload.get("parent_id", "")
        _logger.info("Decomposing task: %s", description)

        sub_tasks = []
        if description:
            parts = description.split(";")
            for i, part in enumerate(parts, 1):
                part = part.strip()
                if part:
                    sub_tasks.append({
                        "id": f"{parent_id}.{i}" if parent_id else str(i),
                        "description": part,
                        "status": "pending",
                    })

        if not sub_tasks:
            sub_tasks.append({
                "id": "1",
                "description": description or "Unnamed task",
                "status": "pending",
            })

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Task decomposition completed.",
            data={
                "status": "completed",
                "parent_task": parent_id,
                "sub_tasks": sub_tasks,
                "sub_task_count": len(sub_tasks),
            },
        )

    async def _handle_workflow_design(self, task: AgentTask) -> AgentResult:
        objective = task.payload.get("objective", "")
        stages = task.payload.get("stages", 3)
        _logger.info("Designing workflow for objective=%s with %d stages", objective, stages)

        workflow = []
        for i in range(1, stages + 1):
            workflow.append({
                "stage": i,
                "name": f"Stage {i}",
                "tasks": [],
                "status": "pending",
            })

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Workflow design completed.",
            data={
                "status": "completed",
                "objective": objective,
                "workflow": workflow,
                "stage_count": len(workflow),
            },
        )
