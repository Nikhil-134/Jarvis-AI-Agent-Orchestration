"""Memory agent placeholder implementation."""

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask


class MemoryAgent(Agent):
    """Agent responsible for memory-related tasks."""

    def __init__(self) -> None:
        super().__init__(name="memory", supported_task_types=("memory.store", "memory.retrieve"))

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"MemoryAgent cannot handle task type: {task.task_type}",
            )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Memory placeholder completed.",
            data={"status": "not_implemented"},
        )
