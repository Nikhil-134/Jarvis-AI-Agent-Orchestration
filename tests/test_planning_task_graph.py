"""Tests for planning.task_graph.TaskGraph — validation and scheduling."""

from __future__ import annotations

import pytest

from planning.exceptions import GraphCycleError, UnknownDependencyError
from planning.models import TaskNode, TaskStatus
from planning.task_graph import TaskGraph


def _node(nid: str, deps: list[str] | None = None, priority: int = 0) -> TaskNode:
    return TaskNode(id=nid, description=f"do {nid}", dependencies=deps, priority=priority)


class TestValidation:
    def test_valid_dag_passes(self) -> None:
        g = TaskGraph([_node("a"), _node("b", ["a"]), _node("c", ["a", "b"])])
        g.validate()  # should not raise

    def test_unknown_dependency_raises(self) -> None:
        g = TaskGraph([_node("a"), _node("b", ["missing"])])
        with pytest.raises(UnknownDependencyError):
            g.validate()

    def test_self_dependency_raises(self) -> None:
        g = TaskGraph([_node("a", ["a"])])
        with pytest.raises(GraphCycleError):
            g.validate()

    def test_cycle_raises(self) -> None:
        g = TaskGraph([_node("a", ["c"]), _node("b", ["a"]), _node("c", ["b"])])
        with pytest.raises(GraphCycleError):
            g.validate()

    def test_duplicate_id_rejected(self) -> None:
        g = TaskGraph([_node("a")])
        with pytest.raises(ValueError):
            g.add_node(_node("a"))


class TestReadyNodes:
    def test_roots_ready_first(self) -> None:
        g = TaskGraph([_node("a"), _node("b", ["a"])])
        ready = g.ready_nodes(set())
        assert [n.id for n in ready] == ["a"]

    def test_dependent_ready_after_completion(self) -> None:
        g = TaskGraph([_node("a"), _node("b", ["a"])])
        ready = g.ready_nodes({"a"})
        assert "b" in [n.id for n in ready]

    def test_priority_ordering(self) -> None:
        g = TaskGraph([_node("low", priority=1), _node("high", priority=9)])
        ready = g.ready_nodes(set())
        assert [n.id for n in ready] == ["high", "low"]

    def test_running_node_not_ready_again(self) -> None:
        g = TaskGraph([_node("a")])
        g.mark_status("a", TaskStatus.RUNNING)
        assert g.ready_nodes(set()) == []


class TestSkipUnreachable:
    def test_skips_transitive_dependents(self) -> None:
        g = TaskGraph([_node("a"), _node("b", ["a"]), _node("c", ["b"]), _node("d")])
        skipped = g.skip_unreachable("a")
        assert set(skipped) == {"b", "c"}
        assert g.get("b").status is TaskStatus.SKIPPED
        assert g.get("c").status is TaskStatus.SKIPPED
        # Independent node d is untouched.
        assert g.get("d").status is TaskStatus.PENDING

    def test_only_pending_are_skipped(self) -> None:
        g = TaskGraph([_node("a"), _node("b", ["a"])])
        g.mark_status("b", TaskStatus.SUCCEEDED)
        skipped = g.skip_unreachable("a")
        assert skipped == []


class TestProgress:
    def test_progress_counts_terminal(self) -> None:
        g = TaskGraph([_node("a"), _node("b")])
        assert g.progress() == (0, 2)
        g.mark_status("a", TaskStatus.SUCCEEDED)
        assert g.progress() == (1, 2)
        assert not g.is_complete
        g.mark_status("b", TaskStatus.FAILED)
        assert g.is_complete

    def test_cancel_all_marks_nonterminal(self) -> None:
        g = TaskGraph([_node("a"), _node("b")])
        g.mark_status("a", TaskStatus.SUCCEEDED)
        cancelled = g.cancel_all()
        assert cancelled == ["b"]
        assert g.get("b").status is TaskStatus.CANCELLED
        assert g.get("a").status is TaskStatus.SUCCEEDED


class TestNodeIdentity:
    def test_hash_and_eq_by_id(self) -> None:
        a1 = _node("a")
        a2 = _node("a")
        assert a1 == a2 and hash(a1) == hash(a2)
        assert len({a1, a2}) == 1
