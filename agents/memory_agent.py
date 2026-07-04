"""Memory agent placeholder implementation."""

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask


class MemoryAgent(Agent):
    """Agent responsible for memory-related tasks."""

    def __init__(self) -> None:
        """Initialize the memory agent."""
        super().__init__(name="memory", supported_task_types=("memory.store", "memory.retrieve"))

    def handle(self, task: AgentTask) -> AgentResult:
        """Return a deterministic memory placeholder result."""
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


if __name__ == "__main__":
    demo_task = AgentTask(task_type="memory.retrieve", payload={"key": "demo"})
    print(MemoryAgent().handle(demo_task))
