"""Tests for asynchronous task queue processing."""

import pytest

from agents import AgentResult, AgentTask
from orchestrator import TaskQueue


@pytest.mark.asyncio
async def test_task_queue_processes_enqueued_tasks() -> None:
    async def handler(task: AgentTask) -> AgentResult:
        return AgentResult("worker", task.task_id, True, "done")

    queue = TaskQueue(handler)
    await queue.start()
    task = AgentTask(task_type="demo")
    task_id = await queue.enqueue(task)
    await queue.join()
    result = queue.get_result(task_id)
    await queue.stop()

    assert result is not None
    assert result.success is True
    assert result.agent_name == "worker"
