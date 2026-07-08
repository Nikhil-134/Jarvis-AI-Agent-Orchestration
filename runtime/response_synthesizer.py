"""Response Synthesizer — converts raw tool output into natural language.

Every tool execution result flows through this component. It determines
the tool type from the tool name and formats the output appropriately:

- Calculator   → "The answer is:\n\n{output}"
- Browser      → Summarized with citations
- Notes        → Formatted markdown / confirmation / table
- Text         → Clean text (no JSON)
- File system  → Confirmation message
- Generic      → Clean display preserving markdown/tables/code/lists

Usage::

    synthesizer = ResponseSynthesizer()
    result = synthesizer.synthesize("calculator", "24")
    # Returns: "The answer is:\n\n24"
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    """Converts raw tool output into natural language responses.

    Dispatches to tool-specific formatters based on *tool_name*.
    Preserves markdown tables, code blocks, and lists in all outputs.
    """

    # Known tool aliases for matching
    _TOOL_ALIASES: dict[str, str] = {
        "calc": "calculator",
        "math": "calculator",
        "search": "browser",
        "web": "browser",
        "read": "file_system",
        "write": "file_system",
    }

    def synthesize(self, tool_name: str, output: str) -> str:
        """Convert raw tool *output* into a natural language string.

        Args:
            tool_name: Name of the tool that produced the output.
            output: Raw output string from the tool.

        Returns:
            A clean, formatted natural language response.
        """
        if not output:
            return self._format_empty(tool_name)

        tool_key = self._TOOL_ALIASES.get(tool_name, tool_name)
        formatter = self._get_formatter(tool_key)
        try:
            result = formatter(output)
            _logger.debug(
                "Synthesized '%s' output: %d chars → %d chars",
                tool_name, len(output), len(result),
            )
            return result
        except Exception as exc:
            _logger.warning("Formatter error for '%s': %s", tool_name, exc)
            return self._format_generic(output)

    def _get_formatter(self, tool_name: str):
        """Return the formatter function for *tool_name*."""
        formatters: dict[str, Any] = {
            "calculator": self._format_calculator,
            "browser": self._format_browser,
            "text": self._format_text,
            "file_system": self._format_file_system,
            "notes": self._format_notes,
            "weather": self._format_weather,
            "datetime": self._format_datetime,
            "uuid": self._format_uuid,
            "json": self._format_json,
            "base64": self._format_base64,
            "hash": self._format_hash,
            "clipboard": self._format_clipboard,
            "system_info": self._format_system_info,
            "notification": self._format_notification,
            "screenshot": self._format_screenshot,
            "shell": self._format_shell,
        }
        return formatters.get(tool_name, self._format_generic)

    # ------------------------------------------------------------------
    # Calculator
    # ------------------------------------------------------------------

    def _format_calculator(self, output: str) -> str:
        display = output.strip()
        # A pure integer (possibly a very large one, e.g. 2**5000 or
        # factorial(100)) is kept exact — never routed through float(), which
        # would overflow to inf and then raise on int(inf).
        if not display.lstrip("-").isdigit():
            try:
                val = float(display)
                if val == int(val) and abs(val) < 1e15:
                    display = str(int(val))
            except (ValueError, TypeError, OverflowError):
                pass
        return f"The answer is:\n\n{display}"

    # ------------------------------------------------------------------
    # Browser / Search
    # ------------------------------------------------------------------

    def _format_browser(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        return cleaned

    # ------------------------------------------------------------------
    # Text tool
    # ------------------------------------------------------------------

    def _format_text(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        return cleaned

    # ------------------------------------------------------------------
    # File system
    # ------------------------------------------------------------------

    def _format_file_system(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        return cleaned

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def _format_notes(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        if cleaned and cleaned != output:
            return cleaned

        lines = output.strip().split("\n")
        if len(lines) >= 2 and len(lines[0]) < 100:
            title = lines[0].strip("*# \t")
            content = "\n".join(lines[1:]).strip()
            return f"# {title}\n\n{content}" if content else f"# {title}"
        return output

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def _format_weather(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        return cleaned

    # ------------------------------------------------------------------
    # Datetime
    # ------------------------------------------------------------------

    def _format_datetime(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        return f"The current date and time is:\n\n{cleaned}"

    # ------------------------------------------------------------------
    # UUID
    # ------------------------------------------------------------------

    def _format_uuid(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        return cleaned

    # ------------------------------------------------------------------
    # JSON tool
    # ------------------------------------------------------------------

    def _format_json(self, output: str) -> str:
        try:
            parsed = json.loads(output)
            pretty = json.dumps(parsed, indent=2)
            return f"```json\n{pretty}\n```"
        except (json.JSONDecodeError, ValueError, TypeError):
            return output

    # ------------------------------------------------------------------
    # Base64
    # ------------------------------------------------------------------

    def _format_base64(self, output: str) -> str:
        return output

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    def _format_hash(self, output: str) -> str:
        return output

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    def _format_clipboard(self, output: str) -> str:
        return f"Clipboard content:\n\n{output}"

    # ------------------------------------------------------------------
    # System info
    # ------------------------------------------------------------------

    def _format_system_info(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        return cleaned

    # ------------------------------------------------------------------
    # Notification
    # ------------------------------------------------------------------

    def _format_notification(self, output: str) -> str:
        return "Notification sent."

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def _format_screenshot(self, output: str) -> str:
        return "Screenshot captured."

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    def _format_shell(self, output: str) -> str:
        return f"```\n{output}\n```"

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------

    def _format_generic(self, output: str) -> str:
        cleaned = self._try_parse_json(output)
        if cleaned and cleaned != output:
            return cleaned
        return output

    # ------------------------------------------------------------------
    # Empty
    # ------------------------------------------------------------------

    @staticmethod
    def _format_empty(tool_name: str) -> str:
        return f"The {tool_name} tool completed with no output."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_json(text: str) -> str:
        """If *text* is JSON, extract meaningful fields and format them.

        Handles both JSON objects and arrays. Falls back to original text.
        """
        stripped = text.strip()
        if not stripped:
            return text

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    return _format_json_dict(data)
            except (json.JSONDecodeError, ValueError):
                pass

        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                items = json.loads(stripped)
                if isinstance(items, list) and items:
                    lines: list[str] = []
                    for item in items:
                        if isinstance(item, dict):
                            lines.append(_format_json_dict(item))
                        else:
                            lines.append(str(item))
                    return "\n\n".join(lines)
            except (json.JSONDecodeError, ValueError):
                pass

        return text

    @staticmethod
    def _try_parse_dict(text: str) -> str:
        """Fallback: if text looks like a Python dict repr, extract fields."""
        stripped = text.strip()
        if stripped.startswith("{") and "':" in stripped:
            try:
                import ast
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    return _format_json_dict(parsed)
            except (ValueError, SyntaxError, MemoryError):
                pass
        return text


def _format_json_dict(data: dict[str, Any]) -> str:
    """Format a dict into human-readable lines.

    Extracts known keys (output, result, response, error, status) and
    formats them cleanly. Falls back to key: value lines.
    """
    if "output" in data and data["output"]:
        val = data["output"]
        if isinstance(val, str):
            return val
        return str(val)
    if "result" in data and data["result"]:
        return str(data["result"])
    if "response" in data and data["response"]:
        return str(data["response"])
    if "error" in data and data["error"]:
        return f"Error: {data['error']}"

    lines: list[str] = []
    for k, v in data.items():
        if k in ("success", "status", "execution_time_ms", "tool_name"):
            continue
        if isinstance(v, dict):
            v = _format_json_dict(v)
        elif isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        lines.append(f"{k}: {v}")
    return "\n".join(lines) if lines else str(data)
