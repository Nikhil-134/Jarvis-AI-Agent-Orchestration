"""Shared runtime context for Jarvis agents."""

from threading import RLock
from typing import Any


class SharedContext:
    """Thread-safe key-value context shared by all agents."""

    def __init__(self, initial_data: dict[str, Any] | None = None) -> None:
        """Initialize the shared context with optional seed data."""
        self._data = dict(initial_data or {})
        self._lock = RLock()

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value from the context."""
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> None:
        """Delete a value from the context when it exists."""
        with self._lock:
            self._data.pop(key, None)

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of all context data."""
        with self._lock:
            return dict(self._data)


if __name__ == "__main__":
    context = SharedContext({"status": "ready"})
    print(context.snapshot())
