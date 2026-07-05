"""Memory agent implementation — full CRUD operations on the memory system."""

import logging

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask
from memory import MemoryService
from memory.models import MemoryItem, MemoryType

_logger = logging.getLogger(__name__)


class MemoryAgent(Agent):
    """Agent responsible for memory-related tasks.

    Supports:
    - ``memory.store`` — Store a new memory
    - ``memory.retrieve`` — Retrieve a memory by id
    - ``memory.search`` — Semantic search across memories
    - ``memory.forget`` — Delete a memory by id
    - ``memory.stats`` — Return memory system statistics
    """

    def __init__(self, memory_service: MemoryService | None = None) -> None:
        super().__init__(
            name="memory",
            supported_task_types=(
                "memory.store",
                "memory.retrieve",
                "memory.search",
                "memory.forget",
                "memory.stats",
            ),
        )
        self._memory_service = memory_service

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"MemoryAgent cannot handle task type: {task.task_type}",
            )

        if self._memory_service is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="MemoryService is not configured.",
                data={"status": "unavailable"},
            )

        try:
            return await self._dispatch(task)
        except Exception as exc:
            _logger.exception("Memory operation failed for task %s", task.task_id)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Memory operation failed: {exc}",
                data={"status": "error"},
            )

    async def _dispatch(self, task: AgentTask) -> AgentResult:
        payload = task.payload
        manager = self._memory_service.manager

        if task.task_type == "memory.store":
            content = str(payload.get("content", ""))
            memory_type = MemoryType(payload.get("memory_type", "conversation"))
            importance = float(payload.get("importance", 0.5))

            item = MemoryItem(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=payload.get("metadata", {}),
            )
            memory_id = await manager.store(item)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=True,
                message="Memory stored successfully.",
                data={"status": "stored", "memory_id": memory_id},
            )

        if task.task_type == "memory.retrieve":
            memory_id = str(payload.get("memory_id", ""))
            if not memory_id:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message="memory_id is required for retrieval.",
                )

            item = await manager.retrieve(memory_id)
            if item is None:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=True,
                    message="Memory not found.",
                    data={"status": "not_found"},
                )

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=True,
                message="Memory retrieved.",
                data={"status": "retrieved", "memory": item.to_document()},
            )

        if task.task_type == "memory.search":
            query = str(payload.get("query", ""))
            top_k = int(payload.get("top_k", 5))

            if not query:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message="Query is required for search.",
                )

            results = await manager.search(query, top_k=top_k)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=True,
                message=f"Found {len(results)} memory results.",
                data={
                    "status": "completed",
                    "results": [r.to_document() for r in results],
                    "count": len(results),
                },
            )

        if task.task_type == "memory.forget":
            memory_id = str(payload.get("memory_id", ""))
            if not memory_id:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message="memory_id is required for deletion.",
                )

            deleted = await manager.forget(memory_id)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=True,
                message=f"Memory {'deleted' if deleted else 'not found'}.",
                data={"status": "deleted" if deleted else "not_found"},
            )

        if task.task_type == "memory.stats":
            stats = await manager.get_stats()
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=True,
                message="Memory stats retrieved.",
                data={"status": "completed", "stats": stats},
            )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=False,
            message=f"Unknown memory task type: {task.task_type}",
        )
