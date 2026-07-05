"""Shared runtime context for Jarvis agents."""

from threading import RLock
from typing import Any

from orchestrator.interfaces import ISharedContext


class SharedContext(ISharedContext):
    """Thread-safe key-value context shared by all agents.

    Implements :class:`ISharedContext`.  Uses a re-entrant lock so that
    agent code holding the lock can safely call other context methods.
    """

    def __init__(self, initial_data: dict[str, Any] | None = None) -> None:
        self._data = dict(initial_data or {})
        self._lock = RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)
