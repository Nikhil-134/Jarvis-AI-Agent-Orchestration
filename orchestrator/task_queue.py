"""Asynchronous task queue for orchestrated agent work."""

import asyncio
import logging

from agents.contracts import AgentResult, AgentTask
from orchestrator.interfaces import ITaskQueue, TaskHandler


class TaskQueue(ITaskQueue):
    """Async queue that routes tasks through a provided task handler.

    Implements :class:`ITaskQueue`.  Workers process queued tasks
    concurrently via the supplied *handler* (typically
    :meth:`Orchestrator.route`).

    Usage::

        queue = TaskQueue(handler, worker_count=4)
        await queue.start()
        task_id = await queue.enqueue(task)
        await queue.join()
        result = queue.get_result(task_id)
        await queue.stop()
    """

    def __init__(self, handler: TaskHandler, worker_count: int = 1) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")

        self._handler = handler
        self._worker_count = worker_count
        self._queue: asyncio.Queue[AgentTask] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._results: dict[str, AgentResult] = {}
        self._running = False
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(), name=f"jarvis-task-worker-{index}")
            for index in range(self._worker_count)
        ]
        self._logger.info("Started %d task queue worker(s)", self._worker_count)

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._logger.info("Task queue stopped")

    async def enqueue(self, task: AgentTask) -> str:
        await self._queue.put(task)
        self._logger.debug("Enqueued task '%s' of type '%s'", task.task_id, task.task_type)
        return task.task_id

    async def join(self) -> None:
        await self._queue.join()

    def get_result(self, task_id: str) -> AgentResult | None:
        return self._results.get(task_id)

    async def _worker(self) -> None:
        while True:
            task = await self._queue.get()
            try:
                self._results[task.task_id] = await self._handler(task)
            except Exception:
                self._logger.exception("Task '%s' failed in worker", task.task_id)
            finally:
                self._queue.task_done()
