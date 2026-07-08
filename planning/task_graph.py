"""TaskGraph — a validated dependency DAG of :class:`TaskNode` objects.

Responsibilities
----------------
* Hold the nodes of a plan and their dependency edges.
* :meth:`validate` — reject unknown dependencies and cycles (Kahn's algorithm)
  before any execution begins.
* :meth:`ready_nodes` — given the set of completed node ids, return the nodes
  whose dependencies are all satisfied and which are still pending.
* :meth:`skip_unreachable` — when a critical node fails, transitively mark its
  dependents ``SKIPPED`` so they never run.
* :meth:`progress` / :attr:`is_complete` — progress tracking.

This does **no** scheduling or I/O — that is the executor's job.  Keeping the
topology logic here (rather than duplicating it inside the executor) keeps the
graph math in one testable place.
"""

from __future__ import annotations

import logging

from planning.exceptions import GraphCycleError, UnknownDependencyError
from planning.models import TaskNode, TaskStatus

_logger = logging.getLogger(__name__)


class TaskGraph:
    """A dependency graph of task nodes with validation and progress tracking."""

    def __init__(self, nodes: list[TaskNode] | None = None) -> None:
        self._nodes: dict[str, TaskNode] = {}
        for node in nodes or []:
            self.add_node(node)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_node(self, node: TaskNode) -> TaskNode:
        """Add *node* to the graph. Duplicate ids are rejected."""
        if node.id in self._nodes:
            raise ValueError(f"Duplicate task id: {node.id!r}")
        self._nodes[node.id] = node
        return node

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> tuple[TaskNode, ...]:
        return tuple(self._nodes.values())

    def get(self, node_id: str) -> TaskNode | None:
        return self._nodes.get(node_id)

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def is_empty(self) -> bool:
        return not self._nodes

    # ------------------------------------------------------------------
    # Validation (Kahn topological sort → detects unknown deps + cycles)
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise if any dependency is unknown or the graph contains a cycle."""
        # 1. Unknown dependency check.
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise UnknownDependencyError(
                        f"Task {node.id!r} depends on unknown task {dep!r}"
                    )
                if dep == node.id:
                    raise GraphCycleError(f"Task {node.id!r} depends on itself")

        # 2. Cycle detection via Kahn's algorithm.
        indegree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for _dep in node.dependencies:
                indegree[node.id] += 1

        queue = [nid for nid, deg in indegree.items() if deg == 0]
        visited = 0
        # dependents[x] = nodes that depend on x
        dependents: dict[str, list[str]] = {nid: [] for nid in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                dependents[dep].append(node.id)

        while queue:
            current = queue.pop()
            visited += 1
            for child in dependents[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if visited != len(self._nodes):
            raise GraphCycleError(
                "Task graph contains a cycle (topological sort incomplete)"
            )

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def ready_nodes(self, completed_ids: set[str]) -> list[TaskNode]:
        """Return pending nodes whose dependencies are all in *completed_ids*.

        Results are ordered by descending priority then ascending id so the
        scheduler is deterministic.
        """
        ready = [
            node
            for node in self._nodes.values()
            if node.status is TaskStatus.PENDING
            and all(dep in completed_ids for dep in node.dependencies)
        ]
        ready.sort(key=lambda n: (-n.priority, n.id))
        return ready

    @property
    def pending_nodes(self) -> list[TaskNode]:
        return [n for n in self._nodes.values() if n.status is TaskStatus.PENDING]

    def mark_status(self, node_id: str, status: TaskStatus) -> None:
        node = self._nodes.get(node_id)
        if node is not None:
            node.status = status

    def skip_unreachable(self, failed_id: str) -> list[str]:
        """Mark every still-pending transitive dependent of *failed_id* SKIPPED.

        Returns the list of node ids that were skipped.  Used when a *critical*
        node fails so downstream work that can never satisfy its dependencies is
        not attempted.
        """
        skipped: list[str] = []
        # Breadth-first over the dependents graph.
        frontier = [failed_id]
        seen: set[str] = {failed_id}
        while frontier:
            current = frontier.pop()
            for node in self._nodes.values():
                if node.id in seen:
                    continue
                if current in node.dependencies and node.status is TaskStatus.PENDING:
                    node.status = TaskStatus.SKIPPED
                    skipped.append(node.id)
                    seen.add(node.id)
                    frontier.append(node.id)
        if skipped:
            _logger.info("Skipped %d unreachable task(s) after %r failed: %s",
                         len(skipped), failed_id, skipped)
        return skipped

    def cancel_all(self) -> list[str]:
        """Mark all non-terminal nodes CANCELLED. Returns the affected ids."""
        cancelled: list[str] = []
        for node in self._nodes.values():
            if not node.status.is_terminal:
                node.status = TaskStatus.CANCELLED
                cancelled.append(node.id)
        return cancelled

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when every node has reached a terminal state."""
        return all(n.status.is_terminal for n in self._nodes.values())

    def progress(self) -> tuple[int, int]:
        """Return ``(terminal_count, total_count)``."""
        total = len(self._nodes)
        done = sum(1 for n in self._nodes.values() if n.status.is_terminal)
        return done, total

    def results(self) -> list:
        """Return the attached NodeResults for nodes that have one."""
        return [n.result for n in self._nodes.values() if n.result is not None]
