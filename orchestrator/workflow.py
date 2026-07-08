"""Workflow engine — multi-agent task coordination for Jarvis Prime Orchestrator.

Supports sequential, parallel, conditional, retry, timeout, and
cancellation semantics via a dependency-graph execution model.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agents.contracts import AgentResult, AgentTask

_logger = logging.getLogger(__name__)


class WorkflowStep:
    """A single step in a multi-agent workflow.

    Each step targets one agent with one task.  Steps can be sequential
    (waiting for previous step) or parallel (executed concurrently).
    Supports retry, timeout, and conditional execution.
    """

    def __init__(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        agent_name: str | None = None,
        depends_on: list[str] | None = None,
        *,
        max_retries: int = 0,
        timeout_seconds: float | None = None,
        condition: Callable[[dict[str, AgentResult]], bool] | None = None,
    ) -> None:
        self.step_id: str = task_type
        self.task_type: str = task_type
        self.payload: dict[str, Any] = payload or {}
        self.agent_name: str | None = agent_name
        self.depends_on: list[str] = depends_on or []
        self.result: AgentResult | None = None
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.condition = condition
        self.skipped: bool = False


class WorkflowPlan:
    """A planned multi-agent workflow with ordered/parallel steps."""

    def __init__(self, goal: str) -> None:
        self.goal: str = goal
        self.steps: list[WorkflowStep] = []
        self._metadata: dict[str, Any] = {}

    def add_step(self, step: WorkflowStep) -> WorkflowStep:
        self.steps.append(step)
        return step

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    @property
    def completed_steps(self) -> list[WorkflowStep]:
        return [s for s in self.steps if s.result is not None]

    @property
    def failed_steps(self) -> list[WorkflowStep]:
        return [s for s in self.steps if s.result is not None and not s.result.success]

    @property
    def skipped_steps(self) -> list[WorkflowStep]:
        return [s for s in self.steps if s.skipped]

    @property
    def all_succeeded(self) -> bool:
        return len(self.failed_steps) == 0 and len(self.completed_steps) == len(self.steps)


class WorkflowEngine:
    """Executes multi-agent workflow plans with retry, timeout, and cancellation.

    Supports:
    - Sequential and parallel execution based on dependency declarations
    - Automatic retry with configurable max_retries per step
    - Timeout per step
    - Conditional step execution
    - Cancellation via asyncio.CancelledError propagation
    """

    def __init__(
        self,
        route_fn: Callable[[AgentTask], Awaitable[AgentResult]],
        default_retries: int = 0,
        default_timeout: float | None = None,
    ) -> None:
        self._route = route_fn
        self._default_retries = default_retries
        self._default_timeout = default_timeout
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        _logger.info("Workflow execution cancelled")

    async def execute(self, plan: WorkflowPlan) -> list[AgentResult]:
        if plan.is_empty:
            return []

        self._cancelled = False
        completed: dict[str, AgentResult] = {}
        results: list[AgentResult] = []

        remaining = list(plan.steps)
        while remaining and not self._cancelled:
            ready: list[WorkflowStep] = []
            still_pending: list[WorkflowStep] = []

            for step in remaining:
                if not self._step_dependencies_satisfied(step, completed):
                    still_pending.append(step)
                    continue

                if step.condition is not None and not step.condition(completed):
                    step.skipped = True
                    _logger.info("Step '%s' skipped (condition not met)", step.task_type)
                    completed[step.step_id] = AgentResult(
                        agent_name="workflow",
                        task_id=step.step_id,
                        success=True,
                        message="Skipped",
                        data={"skipped": True},
                    )
                    continue

                ready.append(step)

            if not ready and still_pending:
                _logger.warning(
                    "Workflow deadlocked: %d steps pending with unsatisfied dependencies",
                    len(still_pending),
                )
                break

            remaining = still_pending

            step_results = await asyncio.gather(
                *(self._execute_step(step) for step in ready),
                return_exceptions=True,
            )

            for step, result in zip(ready, step_results, strict=False):
                if isinstance(result, asyncio.CancelledError):
                    self._cancelled = True
                    failed = AgentResult(
                        agent_name="workflow",
                        task_id=step.step_id,
                        success=False,
                        message="Workflow cancelled",
                        data={"cancelled": True},
                    )
                    step.result = failed
                    completed[step.step_id] = failed
                elif isinstance(result, Exception):
                    _logger.error("Workflow step '%s' failed: %s", step.task_type, result)
                    failed = AgentResult(
                        agent_name="workflow",
                        task_id=step.step_id,
                        success=False,
                        message=str(result),
                        data={"error": str(result)},
                    )
                    step.result = failed
                    completed[step.step_id] = failed
                else:
                    step.result = result
                    completed[step.step_id] = result
                results.append(step.result)

        return results

    def _step_dependencies_satisfied(
        self, step: WorkflowStep, completed: dict[str, AgentResult]
    ) -> bool:
        return all(dep in completed for dep in step.depends_on)

    async def _execute_step(self, step: WorkflowStep) -> AgentResult:
        max_retries = step.max_retries if step.max_retries > 0 else self._default_retries
        timeout = step.timeout_seconds or self._default_timeout

        for attempt in range(max_retries + 1):
            try:
                task = AgentTask(
                    task_type=step.task_type,
                    payload=step.payload,
                )
                _logger.debug(
                    "Workflow executing step '%s' (agent=%s, attempt=%d/%d)",
                    step.task_type,
                    step.agent_name or "auto",
                    attempt + 1,
                    max_retries + 1,
                )

                if timeout is not None:
                    result = await asyncio.wait_for(
                        self._route(task),
                        timeout=timeout,
                    )
                else:
                    result = await self._route(task)

                return result

            except asyncio.TimeoutError:
                _logger.warning(
                    "Step '%s' timed out (attempt %d/%d)",
                    step.task_type, attempt + 1, max_retries + 1,
                )
                if attempt < max_retries:
                    continue
                return AgentResult(
                    agent_name="workflow",
                    task_id=step.step_id,
                    success=False,
                    message=f"Step timed out after {timeout}s",
                    data={"error": f"Timeout after {timeout}s", "step": step.task_type},
                )
            except Exception as exc:
                _logger.warning(
                    "Step '%s' failed (attempt %d/%d): %s",
                    step.task_type, attempt + 1, max_retries + 1, exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                return AgentResult(
                    agent_name="workflow",
                    task_id=step.step_id,
                    success=False,
                    message=str(exc),
                    data={"error": str(exc), "step": step.task_type},
                )

        return AgentResult(
            agent_name="workflow",
            task_id=step.step_id,
            success=False,
            message="Step execution exhausted retries",
            data={"error": "Max retries exceeded"},
        )
