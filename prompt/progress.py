"""Progress indicator for long-running prompt processing."""

from __future__ import annotations

import io
import sys
import time


class ChunkProgress:
    """Simple progress display for multi-chunk prompt processing.

    Shows a compact status line that is overwritten on each update::

        [3/7] Processing chunk 3…  (chunk_size=2.1 KB)

    When the total is 1 or fewer, nothing is displayed.
    """

    def __init__(self, total: int, *, stream: io.TextIOBase | None = None) -> None:
        self._total = total
        self._current = 0
        self._stream = stream or sys.stdout
        self._start_time: float | None = None
        self._finished = False

    def start(self) -> None:
        """Call when processing begins (resets timer)."""
        self._current = 0
        self._start_time = time.monotonic()
        self._finished = False

    def advance(self, chunk_size_chars: int = 0) -> None:
        """Increment the chunk counter and redraw the progress line."""
        if self._total <= 1:
            return
        self._current += 1
        kb = chunk_size_chars / 1024
        msg = f"\r\033[K[{self._current}/{self._total}] Processing chunk {self._current}…  (chunk_size={kb:.1f} KB)"
        self._stream.write(msg)
        self._stream.flush()

    def finish(self) -> None:
        """Clear the progress line when processing is complete."""
        if self._total <= 1 or self._finished:
            return
        elapsed = time.monotonic() - (self._start_time or 0)
        self._stream.write(f"\r\033[KAll {self._total} chunks processed in {elapsed:.1f}s.\n")
        self._stream.flush()
        self._finished = True
