"""Tool agent placeholder implementation."""

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask


class ToolAgent(Agent):
    """Agent responsible for tool execution requests."""

    def __init__(self) -> None:
        super().__init__(name="tool", supported_task_types=("tool.execute",))

    async def handle(self, task: AgentTask) -> AgentResult:
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
