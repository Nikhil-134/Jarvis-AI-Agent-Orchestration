"""ReasoningScratchpad — ephemeral working memory for multi-step reasoning.

Requirement #6: a working memory for multi-step reasoning *during* a
conversation that does **not** pollute long-term memory.

This is deliberately a *different* system from :class:`memory.WorkingMemory`
(which is an LRU cache of durable ``MemoryItem`` objects inside the
``MemoryManager``).  The scratchpad:

* holds free-form key/value scratch state plus the :class:`NodeResult` of every
  executed task, keyed by node id;
* is TTL-scoped with an **injectable clock** so tests are deterministic;
* is created **fresh per** :meth:`PlanningCoordinator.run` so one goal's
  intermediate results never leak into another;
* is **never** written to the vector store or document store — nothing here is
  persisted.  Only the final, verified answer is (optionally) recorded by the
  runtime's existing ``_persist_turn`` path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from planning.models import NodeResult


@dataclass(slots=True)
class _Entry:
    value: Any
    expires_at: float | None


class ReasoningScratchpad:
    """Ephemeral, per-run scratch store for intermediate reasoning state."""

    def __init__(
        self,
        *,
        ttl_seconds: float | None = 3600.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock or time.monotonic
        self._store: dict[str, _Entry] = {}
        self._results: dict[str, NodeResult] = {}

    # ------------------------------------------------------------------
    # Key/value scratch state
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, *, ttl_seconds: float | None = ...) -> None:  # type: ignore[assignment]
        """Store *value* under *key*.

        Pass ``ttl_seconds`` to override the default TTL for this entry;
        ``None`` means never expire.
        """
        ttl = self._ttl if ttl_seconds is ... else ttl_seconds
        expires = None if ttl is None else self._clock() + ttl
        self._store[key] = _Entry(value=value, expires_at=expires)

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return default
        if self._expired(entry):
            self._store.pop(key, None)
            return default
        return entry.value

    def has(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    # ------------------------------------------------------------------
    # Task result store
    # ------------------------------------------------------------------

    def record_result(self, result: NodeResult) -> None:
        """Store the outcome of an executed node."""
        self._results[result.node_id] = result

    def result_for(self, node_id: str) -> NodeResult | None:
        return self._results.get(node_id)

    def all_results(self) -> list[NodeResult]:
        return list(self._results.values())

    def successful_outputs(self) -> list[str]:
        """Return the outputs of every successful node (in insertion order)."""
        return [r.output for r in self._results.values() if r.success and r.output]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def _expired(self, entry: _Entry) -> bool:
        return entry.expires_at is not None and self._clock() >= entry.expires_at

    def purge_expired(self) -> int:
        """Drop expired entries; return how many were removed."""
        expired_keys = [k for k, e in self._store.items() if self._expired(e)]
        for k in expired_keys:
            self._store.pop(k, None)
        return len(expired_keys)

    def clear(self) -> None:
        self._store.clear()
        self._results.clear()

    @property
    def size(self) -> int:
        return len(self._store)


_MISSING = object()
