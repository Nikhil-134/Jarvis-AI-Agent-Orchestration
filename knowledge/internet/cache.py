"""A tiny async-safe TTL cache for retrieval results.

Internet facts are volatile, so entries live only briefly (default 5 min). The
cache exists to (a) avoid hammering public APIs on repeated/duplicate queries
and (b) keep latency low within a conversation. It is bounded in size (LRU-ish
eviction of the oldest entry) so it cannot grow without limit.

Time comes from an injectable ``clock`` (defaults to ``time.monotonic``) so
expiry is deterministic under test without patching globals.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Callable, Generic, TypeVar

_V = TypeVar("_V")


class TTLCache(Generic[_V]):
    """Async-safe, bounded, time-to-live cache keyed by string."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_entries: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._store: "OrderedDict[str, tuple[float, _V]]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> _V | None:
        """Return the cached value for *key*, or None if missing/expired."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                # Expired — evict and miss.
                self._store.pop(key, None)
                return None
            # Mark as recently used.
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: _V) -> None:
        """Store *value* under *key* with the configured TTL."""
        async with self._lock:
            expires_at = self._clock() + self._ttl
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            # Evict oldest entries beyond the cap.
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Current number of entries (may include not-yet-purged expired ones)."""
        return len(self._store)
