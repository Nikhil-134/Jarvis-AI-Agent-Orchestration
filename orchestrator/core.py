"""Task orchestration and agent routing with DIP and middleware."""

import logging
from collections.abc import Iterable

from agents.contracts import AgentResult, AgentTask
from agents.interfaces import IAgent
from orchestrator.exceptions import (
    AgentAlreadyRegisteredError,
    AgentNotRegisteredError,
    NoAgentForTaskError,
)
from orchestrator.interfaces import IEventBus, ISharedContext, ITaskQueue
from orchestrator.middleware import MiddlewarePipeline
from orchestrator.message_bus import MessageBus
from orchestrator.context import SharedContext
from orchestrator.task_queue import TaskQueue


class Orchestrator:
    """Registers agents and routes tasks to capable handlers.

    Follows the Dependency Inversion Principle: accepts interfaces for
    context, event bus, and task queue.  Sensible defaults are provided
    that work out-of-the-box for single-process deployments.

    A :class:`MiddlewarePipeline` allows before/after/on-error hooks to
    be registered for cross-cutting concerns (logging, metrics, auth).
    """

    def __init__(
        self,
        agents: Iterable[IAgent] | None = None,
        context: ISharedContext | None = None,
        event_bus: IEventBus | None = None,
        task_queue: ITaskQueue | None = None,
    ) -> None:
        self._agents: dict[str, IAgent] = {}
        self.context = context or SharedContext()
        self.event_bus = event_bus or MessageBus()
        self.middleware = MiddlewarePipeline()
        self.task_queue = task_queue or TaskQueue(self._route_async_handler)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False
        self._started = False

        for agent in agents or ():
            self.register(agent)

        # Register built-in logging middleware
        self._install_default_middleware()

    def _install_default_middleware(self) -> None:
        """Attach default logging hooks to the middleware pipeline."""

        async def log_before(task: AgentTask) -> None:
            self._logger.info("Routing task '%s' of type '%s'", task.task_id, task.task_type)

        async def log_after(task: AgentTask, result: AgentResult) -> None:
            self._logger.info(
                "Task '%s' completed by '%s': success=%s",
                task.task_id,
                result.agent_name,
                result.success,
            )

        async def log_error(task: AgentTask, exc: Exception) -> None:
            self._logger.error("Task '%s' failed: %s", task.task_id, exc)

        self.middleware.add_before(log_before)
        self.middleware.add_after(log_after)
        self.middleware.add_on_error(log_error)

    @property
    def agents(self) -> dict[str, IAgent]:
        return dict(self._agents)

    def register(self, agent: IAgent) -> None:
        if agent.name in self._agents:
            raise AgentAlreadyRegisteredError(f"Agent already registered: {agent.name}")

        agent.bind_runtime(self.context, self.event_bus)
        self._agents[agent.name] = agent
        self._logger.info("Registered agent '%s'", agent.name)

    def unregister(self, agent_name: str) -> IAgent:
        try:
            agent = self._agents.pop(agent_name)
        except KeyError as exc:
            raise AgentNotRegisteredError(f"Agent is not registered: {agent_name}") from exc

        self._logger.info("Unregistered agent '%s'", agent_name)
        return agent

    async def route(self, task: AgentTask) -> AgentResult:
        """Route a task through middleware to the first matching agent.

        All steps (before-hooks, agent lookup, agent handling) are
        wrapped in a single try/except so that middleware on-error
        hooks fire regardless of where the failure occurs.
        """
        try:
            await self.middleware.run_before(task)
            agent = self._find_agent(task)
            self._logger.debug("Task '%s' routed to agent '%s'", task.task_id, agent.name)
            result = await agent.handle(task)
        except Exception as exc:
            await self.middleware.run_on_error(task, exc)
            raise

        await self.middleware.run_after(task, result)
        return result

    def _find_agent(self, task: AgentTask) -> IAgent:
        """Return the first registered agent that can handle *task*."""
        for agent in self._agents.values():
            if agent.can_handle(task):
                return agent
        raise NoAgentForTaskError(f"No registered agent can handle task type: {task.task_type}")

    async def _route_async_handler(self, task: AgentTask) -> AgentResult:
        """Handler used by the :class:`TaskQueue` — delegates to :meth:`route`."""
        return await self.route(task)

    async def enqueue(self, task: AgentTask) -> str:
        return await self.task_queue.enqueue(task)

    async def initialize(self) -> None:
        for agent in self._agents.values():
            await agent.initialize()
        self._initialized = True
        self._logger.info("Orchestrator initialized")

    async def start(self) -> None:
        if not self._initialized:
            await self.initialize()

        for agent in self._agents.values():
            await agent.start()
        await self.task_queue.start()
        self._started = True
        self._logger.info("Orchestrator started")

    async def stop(self) -> None:
        await self.task_queue.stop()
        for agent in self._agents.values():
            await agent.stop()
        self._started = False
        self._logger.info("Orchestrator stopped")

    async def health_check(self) -> dict[str, object]:
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



