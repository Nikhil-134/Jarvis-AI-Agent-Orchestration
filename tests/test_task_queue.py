"""Tests for asynchronous task queue processing."""

import asyncio

from agents import AgentResult, AgentTask
from orchestrator import TaskQueue


def test_task_queue_processes_enqueued_tasks() -> None:
    """TaskQueue should process tasks and retain results by task id."""
    async def run() -> AgentResult | None:
        async def handler(task: AgentTask) -> AgentResult:
            return AgentResult("worker", task.task_id, True, "done")

        queue = TaskQueue(handler)
        await queue.start()
        task = AgentTask(task_type="demo")
        task_id = await queue.enqueue(task)
        await queue.join()
        result = queue.get_result(task_id)
        await queue.stop()
        return result

    result = asyncio.run(run())

    assert result is not None
    assert result.success is True
    assert result.agent_name == "worker"
