"""Tool agent placeholder implementation."""

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask


class ToolAgent(Agent):
    """Agent responsible for tool execution requests."""

    def __init__(self) -> None:
        """Initialize the tool agent."""
        super().__init__(name="tool", supported_task_types=("tool.execute",))

    def handle(self, task: AgentTask) -> AgentResult:
        """Return a deterministic tool placeholder result."""
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"ToolAgent cannot handle task type: {task.task_type}",
            )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Tool execution placeholder completed.",
            data={"status": "not_implemented"},
        )


if __name__ == "__main__":
    demo_task = AgentTask(task_type="tool.execute", payload={"tool_name": "demo"})
    print(ToolAgent().handle(demo_task))
