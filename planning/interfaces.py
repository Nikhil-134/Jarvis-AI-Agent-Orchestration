"""Interface definitions for the Planning & Task Execution subsystem.

These abstract base classes are the Dependency-Inversion seams: the
:class:`~planning.coordinator.PlanningCoordinator` depends on these
abstractions, not on concrete implementations, so any part can be swapped or
faked in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from planning.models import (
    ExecutionMetrics,
    NodeResult,
    Plan,
    TaskNode,
    VerificationResult,
)


class ICapabilityCatalog(ABC):
    """A catalogue of the capabilities the planner may assign to tasks."""

    @abstractmethod
    def describe_for_planner(self) -> str:
        """Return a human-readable list of available capabilities for a prompt."""

    @abstractmethod
    def is_available(self, capability: str) -> bool:
        """Return whether *capability* can actually be executed right now."""

    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """Return the known capability names."""


class ITaskPlanner(ABC):
    """Decomposes a goal into an executable :class:`Plan`."""

    @abstractmethod
    async def decompose(
        self,
        goal: str,
        memory_context: str,
        catalog: ICapabilityCatalog,
    ) -> Plan:
        """Return a :class:`Plan` for *goal*. Never raises."""


class IToolInvoker(ABC):
    """Executes a single :class:`TaskNode` by delegating to a real backend."""

    @abstractmethod
    async def invoke(self, node: TaskNode, context: str = "") -> NodeResult:
        """Run *node* and return its :class:`NodeResult`. Never raises."""


class ITaskExecutor(ABC):
    """Executes a :class:`Plan`'s task graph with concurrency + resilience."""

    @abstractmethod
    async def execute(self, plan: Plan) -> ExecutionMetrics:
        """Execute *plan* and return aggregate metrics. Never raises."""

    @abstractmethod
    def cancel(self) -> None:
        """Request cooperative cancellation of an in-progress execution."""


class IResponseVerifier(ABC):
    """Validates a final response before it is shown to the user."""

    @abstractmethod
    def verify(
        self,
        response: str,
        plan: Plan,
        results: list[NodeResult],
    ) -> VerificationResult:
        """Return a :class:`VerificationResult`. Never raises."""
