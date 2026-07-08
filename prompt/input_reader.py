"""Input reader — reads user input without arbitrary length limits.

Supports three modes:
1. **Interactive** — single-line ``input()`` (backward compatible).
2. **File** — ``!file <path>`` reads content from a file.
3. **Paste** — ``!paste`` reads multi-line input until a delimiter.
4. **Piped** — auto-detects when stdin is redirected and reads all at once.
"""

from __future__ import annotations

import enum
import logging
import sys
from pathlib import Path
from typing import IO

_logger = logging.getLogger(__name__)

_PASTE_DELIMITER = "."
_PASTE_HELP = "Enter/paste your content.  End with a line containing only '.' (period)."


class InputMode(enum.Enum):
    """How the input was sourced."""

    INTERACTIVE = "interactive"
    FILE = "file"
    PASTE = "paste"
    PIPED = "piped"


class InputReader:
    """Reads user input without arbitrary length limits.

    Usage::

        reader = InputReader()
        content, mode = await reader.read()  # or reader.read_interactive()
    """

    def __init__(self, stdin: IO[str] | None = None) -> None:
        self._stdin = stdin or sys.stdin

    async def read(self) -> tuple[str, InputMode]:
        """Detect input source and read accordingly.

        In piped mode reads all stdin lines; otherwise presents the
        interactive prompt.
        """
        if self._is_piped():
            content = self._read_piped()
            if content:
                return (content, InputMode.PIPED)
        return await self.read_interactive()

    def _is_piped(self) -> bool:
        """Detect whether stdin has been redirected from a pipe or file."""
        try:
            return not self._stdin.isatty()
        except Exception:
            return False

    def _read_piped(self) -> str:
        """Read all available lines from piped stdin."""
        try:
            lines = self._stdin.readlines()
            return "".join(lines).strip()
        except (OSError, ValueError) as exc:
            _logger.warning("Failed to read piped input: %s", exc)
            return ""

    async def read_interactive(self, prompt: str = "\nYou: ") -> tuple[str, InputMode]:
        """Read one interactive turn, supporting special commands.

        Commands:
        - ``!file <path>`` — read content from a file
        - ``!paste``      — multi-line paste mode (end with ``.`` on its own line)
        - everything else — single-line input (backward compatible)
        """
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return ("exit", InputMode.INTERACTIVE)

        if not raw:
            return ("", InputMode.INTERACTIVE)
        if raw.startswith("!file "):
            return self._read_file(raw)
        if raw == "!paste":
            return self._read_paste()
        return (raw, InputMode.INTERACTIVE)

    def _read_file(self, raw: str) -> tuple[str, InputMode]:
        """Handle ``!file <path>`` command."""
        path_str = raw[len("!file "):].strip().strip("\"'")
        if not path_str:
            print("Usage: !file <path>")
            return ("", InputMode.INTERACTIVE)
        path = Path(path_str).expanduser()
        if not path.exists():
            print(f"File not found: {path}")
            return ("", InputMode.INTERACTIVE)
        try:
            content = path.read_text(encoding="utf-8")
            size = len(content)
            print(f"\nRead {size:,} characters from {path.name}")
            return (content, InputMode.FILE)
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Error reading file: {exc}")
            return ("", InputMode.INTERACTIVE)

    def _read_paste(self) -> tuple[str, InputMode]:
        """Handle ``!paste`` multi-line input mode."""
        print(_PASTE_HELP)
        lines: list[str] = []
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == _PASTE_DELIMITER:
                break
            lines.append(line)
        content = "\n".join(lines).strip()
        if content:
            print(f"\nRead {len(content):,} characters.")
        return (content, InputMode.PASTE)


async def read_long_input() -> tuple[str, InputMode]:
    """Convenience function for single-input scenarios."""
    reader = InputReader()
    return await reader.read()
