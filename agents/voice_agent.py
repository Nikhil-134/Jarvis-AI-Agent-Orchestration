"""Voice agent placeholder implementation."""

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask


class VoiceAgent(Agent):
    """Agent responsible for voice input and output tasks."""

    def __init__(self) -> None:
        super().__init__(name="voice", supported_task_types=("voice.input", "voice.output"))

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"VoiceAgent cannot handle task type: {task.task_type}",
            )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Voice placeholder completed.",
            data={"status": "not_implemented"},
        )
