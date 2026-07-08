"""Tests for planning.executor.TaskExecutor — concurrency + resilience."""

from __future__ import annotations

import asyncio

from planning.executor import TaskExecutor
from planning.models import NodeResult, Plan, RetryPolicy, TaskNode, TaskStatus
from planning.scratchpad import ReasoningScratchpad


class ScriptedInvoker:
    """Invoker whose behaviour per node id is scripted for the test."""

    def __init__(self, behaviours: dict) -> None:
        # behaviours[node_id] = callable(node) -> NodeResult | 'sleep'/'raise'
        self._behaviours = behaviours
        self.max_concurrent = 0
        self._active = 0

    async def invoke(self, node: TaskNode, context: str = "") -> NodeResult:
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            behaviour = self._behaviours.get(node.id, "ok")
            if behaviour == "ok":
                await asyncio.sleep(0.01)
                return NodeResult(node.id, TaskStatus.SUCCEEDED, output=f"{node.id}-out",
                                  backend="reasoning", attempts=1)
            if behaviour == "hang":
                await asyncio.sleep(10)  # will hit the executor timeout
                return NodeResult(node.id, TaskStatus.SUCCEEDED)
            if behaviour == "fail":
                return NodeResult(node.id, TaskStatus.FAILED, error="nope",
                                  backend="reasoning")
            if callable(behaviour):
                return await behaviour(node)
            return NodeResult(node.id, TaskStatus.SUCCEEDED)
        finally:
            self._active -= 1


def _plan(nodes: list[TaskNode]) -> Plan:
    return Plan(goal="g", nodes=tuple(nodes), overall_confidence=0.8)


class TestConcurrency:
    async def test_independent_nodes_run_in_parallel(self) -> None:
        nodes = [TaskNode(id=f"n{i}", description="x") for i in range(4)]
        inv = ScriptedInvoker({})
        ex = TaskExecutor(inv, max_parallel=4, task_timeout_seconds=5)
        metrics = await ex.execute(_plan(nodes))
        assert metrics.succeeded == 4
        assert inv.max_concurrent >= 2  # genuinely concurrent

    async def test_semaphore_caps_parallelism(self) -> None:
        nodes = [TaskNode(id=f"n{i}", description="x") for i in range(6)]
        inv = ScriptedInvoker({})
        ex = TaskExecutor(inv, max_parallel=2, task_timeout_seconds=5)
        await ex.execute(_plan(nodes))
        assert inv.max_concurrent <= 2


class TestDependencies:
    async def test_dependency_order_respected(self) -> None:
        order: list[str] = []

        async def record(node):
            order.append(node.id)
            return NodeResult(node.id, TaskStatus.SUCCEEDED, output="x")

        nodes = [
            TaskNode(id="a", description="a"),
            TaskNode(id="b", description="b", dependencies=["a"]),
        ]
        inv = ScriptedInvoker({"a": record, "b": record})
        ex = TaskExecutor(inv, task_timeout_seconds=5)
        await ex.execute(_plan(nodes))
        assert order == ["a", "b"]

    async def test_failed_critical_skips_dependents(self) -> None:
        nodes = [
            TaskNode(id="a", description="a"),
            TaskNode(id="b", description="b", dependencies=["a"]),
        ]
        inv = ScriptedInvoker({"a": "fail"})
        ex = TaskExecutor(inv, task_timeout_seconds=5)
        metrics = await ex.execute(_plan(nodes))
        assert metrics.failed == 1
        assert metrics.skipped == 1


class TestTimeoutAndRetry:
    async def test_timeout_marks_timed_out(self) -> None:
        node = TaskNode(id="a", description="a", retry_policy=RetryPolicy.none())
        inv = ScriptedInvoker({"a": "hang"})
        ex = TaskExecutor(inv, task_timeout_seconds=0.05)
        metrics = await ex.execute(_plan([node]))
        assert metrics.timed_out == 1

    async def test_retry_then_succeed(self) -> None:
        attempts = {"n": 0}

        async def flaky(node):
            attempts["n"] += 1
            if attempts["n"] < 2:
                return NodeResult(node.id, TaskStatus.FAILED, error="transient")
            return NodeResult(node.id, TaskStatus.SUCCEEDED, output="ok")

        node = TaskNode(id="a", description="a",
                        retry_policy=RetryPolicy(max_retries=2, backoff_seconds=0.0))
        inv = ScriptedInvoker({"a": flaky})
        ex = TaskExecutor(inv, task_timeout_seconds=5)
        metrics = await ex.execute(_plan([node]))
        assert metrics.succeeded == 1
        assert attempts["n"] == 2

    async def test_retry_count_recorded(self) -> None:
        node = TaskNode(id="a", description="a",
                        retry_policy=RetryPolicy(max_retries=2, backoff_seconds=0.0))
        inv = ScriptedInvoker({"a": "fail"})
        ex = TaskExecutor(inv, task_timeout_seconds=5)
        metrics = await ex.execute(_plan([node]))
        assert metrics.failed == 1
        assert metrics.total_attempts == 3  # 1 + 2 retries


class TestCancellation:
    async def test_cancel_mid_run(self) -> None:
        async def slow(node):
            await asyncio.sleep(0.2)
            return NodeResult(node.id, TaskStatus.SUCCEEDED)

        nodes = [TaskNode(id=f"n{i}", description="x") for i in range(3)]
        inv = ScriptedInvoker({n.id: slow for n in nodes})
        ex = TaskExecutor(inv, max_parallel=1, task_timeout_seconds=5)

        async def canceller():
            await asyncio.sleep(0.05)
            ex.cancel()

        metrics, _ = await asyncio.gather(ex.execute(_plan(nodes)), canceller())
        # Not all three could have finished before cancel.
        assert metrics.succeeded < 3
        assert metrics.cancelled >= 1


class TestScratchpadAndSafety:
    async def test_results_recorded_in_scratchpad(self) -> None:
        sp = ReasoningScratchpad()
        node = TaskNode(id="a", description="a")
        inv = ScriptedInvoker({})
        ex = TaskExecutor(inv, task_timeout_seconds=5, scratchpad=sp)
        await ex.execute(_plan([node]))
        assert sp.result_for("a").success

    async def test_empty_plan_returns_zero_metrics(self) -> None:
        ex = TaskExecutor(ScriptedInvoker({}), task_timeout_seconds=5)
        metrics = await ex.execute(Plan(goal="g", nodes=()))
        assert metrics.total == 0

    async def test_dependency_context_passed_downstream(self) -> None:
        seen: dict[str, str] = {}

        async def capture(node):
            seen[node.id] = node.description  # placeholder
            return NodeResult(node.id, TaskStatus.SUCCEEDED, output=f"{node.id}-out")

        class CtxInvoker(ScriptedInvoker):
            async def invoke(self, node, context=""):
                seen[node.id] = context
                return NodeResult(node.id, TaskStatus.SUCCEEDED, output=f"{node.id}-out")

        nodes = [
            TaskNode(id="a", description="a"),
            TaskNode(id="b", description="b", dependencies=["a"]),
        ]
        sp = ReasoningScratchpad()
        ex = TaskExecutor(CtxInvoker({}), task_timeout_seconds=5, scratchpad=sp)
        await ex.execute(_plan(nodes))
        assert "a-out" in seen["b"]  # downstream saw upstream output
