"""Steve agent — Testing & QA specialist."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_TESTING, Capability
from agents.contracts import AgentResult, AgentTask

_logger = logging.getLogger(__name__)


class SteveAgent(Agent):
    """Agent responsible for running tests, generating test cases, and analyzing coverage."""

    def __init__(
        self,
        llm_provider: Any | None = None,
        memory_service: Any | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="steve",
            supported_task_types=("test.run", "test.create", "test.analyze", "coverage.report"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Capability]:
        return [CAPABILITY_TESTING]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"SteveAgent cannot handle task type: {task.task_type}",
            )

        handlers = {
            "test.run": self._handle_test_run,
            "test.create": self._handle_test_create,
            "test.analyze": self._handle_test_analyze,
            "coverage.report": self._handle_coverage_report,
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
            _logger.exception("SteveAgent failed to handle task %s", task.task_id)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Internal error while processing task.",
            )

    async def _handle_test_run(self, task: AgentTask) -> AgentResult:
        test_path = task.payload.get("test_path", "")
        args = task.payload.get("args", "")
        _logger.info("Running tests at %s with args=%s", test_path, args)

        if self._tool_engine is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Running tests requires a tool engine to execute pytest.",
                data={"status": "unavailable", "test_path": test_path},
            )

        cmd = f"python -m pytest {test_path} {args} -v 2>&1"
        result = await self._tool_engine.execute("shell", command=cmd)
        output = result.output if result.success else (result.error or "Tests failed")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            # A failing test run must not be reported as a success.
            success=result.success,
            message="Test execution completed." if result.success else "Test execution reported failures.",
            data={
                "status": "completed" if result.success else "error",
                "test_path": test_path,
                "output": output,
            },
        )

    async def _handle_test_create(self, task: AgentTask) -> AgentResult:
        specification = task.payload.get("specification", "")
        module_name = task.payload.get("module_name", "module")
        _logger.info("Generating test cases for %s", module_name)

        test_cases = []
        lines = specification.strip().split("\n")
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                test_cases.append({
                    "id": f"test_{i}",
                    "description": line,
                    "status": "generated",
                })

        if not test_cases:
            test_cases.append({
                "id": "test_1",
                "description": f"Verify {module_name} basic functionality",
                "status": "generated",
            })

        template = (
            f'"""Tests for {module_name}."""\n\n'
            f"import pytest\nfrom {module_name} import *\n\n\n"
        )
        for tc in test_cases:
            template += f"def test_{tc['id']}():\n    \"\"\"{tc['description']}\"\"\"\n    pass\n\n\n"

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Test cases generated.",
            data={
                "status": "completed",
                "module_name": module_name,
                "test_cases": test_cases,
                "test_count": len(test_cases),
                "generated_code": template,
            },
        )

    async def _handle_test_analyze(self, task: AgentTask) -> AgentResult:
        test_output = task.payload.get("test_output", "")
        _logger.info("Analyzing test results")

        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        for line in test_output.split("\n"):
            if line.startswith("PASSED") or "PASSED" in line:
                passed += 1
            elif "FAILED" in line:
                failed += 1
            elif "ERROR" in line:
                errors += 1
            elif "SKIPPED" in line:
                skipped += 1

        total = passed + failed + errors + skipped
        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": round((passed / total * 100) if total > 0 else 0, 1),
        }

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Test analysis completed.",
            data={
                "status": "completed",
                "summary": summary,
            },
        )

    async def _handle_coverage_report(self, task: AgentTask) -> AgentResult:
        target = task.payload.get("target", ".")
        _logger.info("Generating coverage report for %s", target)

        if self._tool_engine is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Coverage reporting requires a tool engine to execute pytest.",
                data={"status": "unavailable", "target": target},
            )

        cmd = f"python -m pytest --cov={target} --cov-report=term-missing 2>&1"
        result = await self._tool_engine.execute("shell", command=cmd)
        raw = result.output if result.success else (result.error or "Coverage failed")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=result.success,
            message="Coverage report generated." if result.success else "Coverage run reported failures.",
            data={
                "status": "completed" if result.success else "error",
                "target": target,
                "coverage": {"raw_output": raw},
            },
        )
