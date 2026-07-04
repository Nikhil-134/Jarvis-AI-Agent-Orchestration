"""Voice agent placeholder implementation."""

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask


class VoiceAgent(Agent):
    """Agent responsible for voice input and output tasks."""

    def __init__(self) -> None:
        """Initialize the voice agent."""
        super().__init__(name="voice", supported_task_types=("voice.input", "voice.output"))

    def handle(self, task: AgentTask) -> AgentResult:
        """Return a deterministic voice placeholder result."""
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


if __name__ == "__main__":
    demo_task = AgentTask(task_type="voice.output", payload={"text": "demo"})
    print(VoiceAgent().handle(demo_task))
