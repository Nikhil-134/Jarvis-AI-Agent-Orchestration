"""Middleware pipeline for orchestrator task routing."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agents.contracts import AgentResult, AgentTask

BeforeHook = Callable[[AgentTask], Awaitable[None]]
AfterHook = Callable[[AgentTask, AgentResult], Awaitable[None]]
ErrorHook = Callable[[AgentTask, Exception], Awaitable[None]]


class MiddlewarePipeline:
    """Pipeline of before/after/on-error hooks executed during task routing.

    Hooks run in registration order:

    - *Before* hooks receive the task before routing.
    - *After* hooks receive the task and result after a successful route.
    - *On error* hooks receive the task and exception when routing fails.
    """

    def __init__(self) -> None:
        self._before: list[BeforeHook] = []
        self._after: list[AfterHook] = []
        self._on_error: list[ErrorHook] = []
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add_before(self, hook: BeforeHook) -> None:
        """Register a hook that runs before task routing."""
        self._before.append(hook)
        self._logger.debug("Registered before-task hook %s", getattr(hook, "__name__", hook))

    def add_after(self, hook: AfterHook) -> None:
        """Register a hook that runs after a successful route."""
        self._after.append(hook)
        self._logger.debug("Registered after-task hook %s", getattr(hook, "__name__", hook))

    def add_on_error(self, hook: ErrorHook) -> None:
        """Register a hook that runs when routing raises an exception."""
        self._on_error.append(hook)
        self._logger.debug("Registered on-error hook %s", getattr(hook, "__name__", hook))

    async def run_before(self, task: AgentTask) -> None:
        """Execute all before-task hooks in order."""
        for hook in self._before:
            await hook(task)

    async def run_after(self, task: AgentTask, result: AgentResult) -> None:
        """Execute all after-task hooks in order."""
        for hook in self._after:
            await hook(task, result)

    async def run_on_error(self, task: AgentTask, exc: Exception) -> None:
        """Execute all on-error hooks in order."""
        for hook in self._on_error:
            await hook(task, exc)

    def clear(self) -> None:
        """Remove all registered hooks."""
        self._before.clear()
        self._after.clear()
        self._on_error.clear()
