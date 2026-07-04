"""Asynchronous task queue for orchestrated agent work."""

import asyncio
from collections.abc import Awaitable, Callable

from agents.contracts import AgentResult, AgentTask

TaskHandler = Callable[[AgentTask], Awaitable[AgentResult]]


class TaskQueue:
    """Async queue that routes tasks through a provided task handler."""

    def __init__(self, handler: TaskHandler, worker_count: int = 1) -> None:
        """Initialize the queue with an async task handler."""
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")

        self._handler = handler
        self._worker_count = worker_count
        self._queue: asyncio.Queue[AgentTask] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._results: dict[str, AgentResult] = {}
        self._running = False

    @property
    def is_running(self) -> bool:
        """Return whether queue workers are active."""
        return self._running

    async def start(self) -> None:
        """Start queue workers."""
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(), name=f"jarvis-task-worker-{index}")
            for index in range(self._worker_count)
        ]

    async def stop(self) -> None:
        """Stop queue workers after cancelling pending worker loops."""
        if not self._running:
            return

        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def enqueue(self, task: AgentTask) -> str:
        """Add a task to the queue and return its task id."""
        await self._queue.put(task)
        return task.task_id

    async def join(self) -> None:
        """Wait until all queued tasks have been processed."""
        await self._queue.join()

    def get_result(self, task_id: str) -> AgentResult | None:
        """Return a processed task result by task id."""
        return self._results.get(task_id)

    async def _worker(self) -> None:
        """Continuously process queued tasks until cancelled."""
        while True:
            task = await self._queue.get()
            try:
                self._results[task.task_id] = await self._handler(task)
            finally:
                self._queue.task_done()


if __name__ == "__main__":
    async def demo_handler(task: AgentTask) -> AgentResult:
        return AgentResult("demo", task.task_id, True, "processed")

    async def demo() -> None:
        queue = TaskQueue(demo_handler)
        await queue.start()
        task_id = await queue.enqueue(AgentTask(task_type="demo"))
        await queue.join()
        print(queue.get_result(task_id))
        await queue.stop()

    asyncio.run(demo())
