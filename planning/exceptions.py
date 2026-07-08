"""Exceptions for the Planning & Task Execution subsystem.

These are raised only by :meth:`TaskGraph.validate` (a programming/plan-shape
error caught by the coordinator, which then degrades gracefully).  The
executor, invoker, planner, verifier, and coordinator themselves never raise —
they return typed results.
"""

from __future__ import annotations


class PlanningError(Exception):
    """Base class for planning-subsystem errors."""


class GraphValidationError(PlanningError):
    """The task graph failed structural validation."""


class UnknownDependencyError(GraphValidationError):
    """A task declared a dependency on a task id that does not exist."""


class GraphCycleError(GraphValidationError):
    """The task graph contains a dependency cycle."""
