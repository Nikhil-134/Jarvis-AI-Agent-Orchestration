"""Task orchestration and agent routing."""

import asyncio
import logging
from collections.abc import Iterable

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask
from orchestrator.context import SharedContext
from orchestrator.exceptions import (
    AgentAlreadyRegisteredError,
    AgentNotRegisteredError,
    NoAgentForTaskError,
)
from orchestrator.message_bus import MessageBus
from orchestrator.task_queue import TaskQueue


class Orchestrator:
    """Registers agents and routes tasks to capable handlers."""

    def __init__(
        self,
        agents: Iterable[Agent] | None = None,
        context: SharedContext | None = None,
        message_bus: MessageBus | None = None,
        task_queue: TaskQueue | None = None,
    ) -> None:
        """Initialize the orchestrator with an optional set of agents."""
        self._agents: dict[str, Agent] = {}
        self.context = context or SharedContext()
        self.message_bus = message_bus or MessageBus()
        self.task_queue = task_queue or TaskQueue(self.route_async)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False
        self._started = False

        for agent in agents or ():
            self.register(agent)

    @property
    def agents(self) -> dict[str, Agent]:
        """Return a copy of the registered agent mapping."""
        return dict(self._agents)

    def register(self, agent: Agent) -> None:
        """Register an agent by its unique name."""
        if agent.name in self._agents:
            raise AgentAlreadyRegisteredError(f"Agent already registered: {agent.name}")

        agent.bind_runtime(self.context, self.message_bus)
        self._agents[agent.name] = agent
        self._logger.info("Registered agent '%s'", agent.name)

    def unregister(self, agent_name: str) -> Agent:
        """Remove and return a registered agent by name."""
        try:
            agent = self._agents.pop(agent_name)
        except KeyError as exc:
            raise AgentNotRegisteredError(f"Agent is not registered: {agent_name}") from exc

        self._logger.info("Unregistered agent '%s'", agent_name)
        return agent

    def route(self, task: AgentTask) -> AgentResult:
        """Route a task to the first registered agent that can handle it."""
        self._logger.info("Routing task '%s' of type '%s'", task.task_id, task.task_type)

        for agent in self._agents.values():
            if agent.can_handle(task):
                self._logger.info("Task '%s' routed to agent '%s'", task.task_id, agent.name)
                return agent.handle(task)

        self._logger.warning("No agent found for task type '%s'", task.task_type)
        raise NoAgentForTaskError(f"No registered agent can handle task type: {task.task_type}")

    async def route_async(self, task: AgentTask) -> AgentResult:
        """Asynchronously route a task while preserving the Phase 1 sync handler contract."""
        return await asyncio.to_thread(self.route, task)

    async def enqueue(self, task: AgentTask) -> str:
        """Enqueue a task for asynchronous processing."""
        return await self.task_queue.enqueue(task)

    async def initialize(self) -> None:
        """Initialize all registered agents."""
        for agent in self._agents.values():
            await agent.initialize()
        self._initialized = True
        self._logger.info("Orchestrator initialized")

    async def start(self) -> None:
        """Start the orchestrator, agents, and task queue."""
        if not self._initialized:
            await self.initialize()

        for agent in self._agents.values():
            await agent.start()
        await self.task_queue.start()
        self._started = True
        self._logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the task queue and all registered agents."""
        await self.task_queue.stop()
        for agent in self._agents.values():
            await agent.stop()
        self._started = False
        self._logger.info("Orchestrator stopped")

    async def health_check(self) -> dict[str, object]:
        """Return orchestrator and agent health information."""
        agent_health = {
            agent.name: await agent.health_check()
            for agent in self._agents.values()
        }
        return {
            "initialized": self._initialized,
            "started": self._started,
            "agent_count": len(self._agents),
            "task_queue_running": self.task_queue.is_running,
            "agents": agent_health,
        }


if __name__ == "__main__":
    from agents import MemoryAgent, PlannerAgent, ToolAgent, VoiceAgent
    from config import configure_logging

    configure_logging()
    orchestrator = Orchestrator([PlannerAgent(), MemoryAgent(), ToolAgent(), VoiceAgent()])
    print(orchestrator.route(AgentTask(task_type="plan", payload={"goal": "demo"})))
