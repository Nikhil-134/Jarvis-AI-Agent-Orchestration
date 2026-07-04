"""Planner agent implementation."""

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider, ChatSession, LLMError, PromptManager


class PlannerAgent(Agent):
    """Agent responsible for planning tasks with optional LLM support."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        """Initialize the planner agent."""
        super().__init__(name="planner", supported_task_types=("plan",))
        self._prompt_manager = prompt_manager or PromptManager()
        self._chat_session = (
            ChatSession(llm_provider, system_prompt="You are the Jarvis planning agent.")
            if llm_provider
            else None
        )

    def handle(self, task: AgentTask) -> AgentResult:
        """Plan a task, using an LLM provider when one is configured."""
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"PlannerAgent cannot handle task type: {task.task_type}",
            )

        if self._chat_session is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=True,
                message="Planning placeholder completed.",
                data={"status": "not_implemented", "steps": []},
            )

        goal = str(task.payload.get("goal", "No goal provided."))
        prompt = self._prompt_manager.render("planner", goal=goal)
        try:
            plan = self._chat_session.send(prompt)
        except LLMError as exc:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Planning LLM request failed: {exc}",
                data={"status": "llm_error"},
            )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Planning completed with LLM provider.",
            data={"status": "completed", "plan": plan},
        )


if __name__ == "__main__":
    demo_task = AgentTask(task_type="plan", payload={"goal": "demo"})
    print(PlannerAgent().handle(demo_task))
