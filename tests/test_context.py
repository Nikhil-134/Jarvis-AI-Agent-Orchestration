"""Tests for shared agent context."""

from orchestrator import SharedContext


def test_shared_context_sets_gets_deletes_and_snapshots_values() -> None:
    """SharedContext should provide basic key-value operations."""
    context = SharedContext({"existing": 1})

    context.set("new", {"value": True})
    context.delete("existing")

    assert context.get("existing") is None
    assert context.get("new") == {"value": True}
    assert context.snapshot() == {"new": {"value": True}}
