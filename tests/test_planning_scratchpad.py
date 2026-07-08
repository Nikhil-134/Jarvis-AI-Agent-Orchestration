"""Tests for planning.scratchpad.ReasoningScratchpad — TTL + isolation."""

from __future__ import annotations

from planning.models import NodeResult, TaskStatus
from planning.scratchpad import ReasoningScratchpad


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestKeyValue:
    def test_set_get(self) -> None:
        sp = ReasoningScratchpad()
        sp.set("k", 123)
        assert sp.get("k") == 123

    def test_missing_returns_default(self) -> None:
        sp = ReasoningScratchpad()
        assert sp.get("nope", "d") == "d"

    def test_delete(self) -> None:
        sp = ReasoningScratchpad()
        sp.set("k", 1)
        sp.delete("k")
        assert sp.get("k") is None


class TestTTL:
    def test_entry_expires(self) -> None:
        clock = _Clock()
        sp = ReasoningScratchpad(ttl_seconds=10.0, clock=clock)
        sp.set("k", "v")
        clock.t = 9.0
        assert sp.get("k") == "v"
        clock.t = 10.0
        assert sp.get("k") is None

    def test_no_ttl_never_expires(self) -> None:
        clock = _Clock()
        sp = ReasoningScratchpad(ttl_seconds=None, clock=clock)
        sp.set("k", "v")
        clock.t = 1e9
        assert sp.get("k") == "v"

    def test_purge_expired(self) -> None:
        clock = _Clock()
        sp = ReasoningScratchpad(ttl_seconds=5.0, clock=clock)
        sp.set("a", 1)
        sp.set("b", 2)
        clock.t = 6.0
        assert sp.purge_expired() == 2
        assert sp.size == 0


class TestResults:
    def test_record_and_recall(self) -> None:
        sp = ReasoningScratchpad()
        r = NodeResult(node_id="s1", status=TaskStatus.SUCCEEDED, output="hi")
        sp.record_result(r)
        assert sp.result_for("s1") is r
        assert sp.successful_outputs() == ["hi"]

    def test_failed_output_excluded(self) -> None:
        sp = ReasoningScratchpad()
        sp.record_result(NodeResult(node_id="s1", status=TaskStatus.FAILED, output="x"))
        assert sp.successful_outputs() == []


class TestIsolation:
    def test_two_scratchpads_are_independent(self) -> None:
        a = ReasoningScratchpad()
        b = ReasoningScratchpad()
        a.set("k", "a-value")
        assert b.get("k") is None
