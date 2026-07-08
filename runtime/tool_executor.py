"""Tool Executor — parses tool requests, executes them, and returns natural language.

Ensures raw tool JSON, function call payloads, and internal execution details
NEVER reach the user.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

from agents.contracts import AgentResult, AgentTask
from orchestrator import Orchestrator

_logger = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    """A tool execution failed with a user-facing error."""


class ToolExecutor:
    """Executes tool calls and returns clean natural language results.

    Wraps the existing ToolAgent and ToolExecutionEngine to ensure:
    - Raw JSON is never returned to the user
    - Tool call payloads are hidden
    - Results are formatted as natural language
    - Errors are user-friendly

    Usage::

        executor = ToolExecutor(orchestrator)
        result = await executor.execute("calculator", {"expression": "2+2"})
        # Returns: "The result of 2+2 is 4."
    """

    def __init__(self, orchestrator: Orchestrator | None) -> None:
        self._orchestrator = orchestrator

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """Execute a tool and return a natural language result string.

        Never returns raw JSON or internal execution details.
        """
        if self._orchestrator is None:
            return self._format_error(f"The {tool_name} tool is not available.")

        try:
            result: AgentResult = await self._orchestrator.route(
                AgentTask(
                    task_type="tool.execute",
                    payload={
                        "tool_name": tool_name,
                        "arguments": arguments or {},
                    },
                )
            )

            return self._format_result(tool_name, result)
        except Exception as exc:
            _logger.exception("Tool execution failed: %s", tool_name)
            return self._format_error(f"Failed to execute {tool_name}: {exc}")

    @staticmethod
    def _format_result(tool_name: str, result: AgentResult) -> str:
        """Convert an AgentResult into clean natural language."""
        if not result.success:
            # Log the real error internally; show the user a graceful, generic
            # message so internal details (missing deps, stack traces, "X is
            # required") never leak.
            internal = ""
            if result.data:
                internal = result.data.get("error") or ""
            internal = internal or result.message or "unknown error"
            _logger.info("Tool '%s' failed internally: %s", tool_name, internal)
            return (
                f"I wasn't able to complete the {tool_name} operation just now. "
                "Please try again or rephrase your request."
            )

        output = ""
        if result.data:
            output = result.data.get("response") or result.data.get("output") or ""

        if not output:
            output = result.message or ""

        if not output:
            return f"The {tool_name} tool completed successfully."

        cleaned = _clean_tool_output(output)
        return cleaned

    @staticmethod
    def _format_error(message: str) -> str:
        return f"I'm sorry, but I encountered an issue: {message}"

    @staticmethod
    def format_tool_summary(tool_name: str, output: str) -> str:
        """Format a tool result as a brief summary for inclusion in a response."""
        cleaned = _clean_tool_output(output)
        preview = cleaned[:300] if len(cleaned) > 300 else cleaned
        return f"**{tool_name}**: {preview}"

    async def execute_and_format(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """Execute a tool and format the result as a concise summary."""
        result = await self.execute(tool_name, arguments)
        return self.format_tool_summary(tool_name, result)


def _clean_tool_output(output: str) -> str:
    """Remove raw JSON, dict representations, and internal metadata from output.

    Ensures users never see Python dict repr, JSON payloads, or internal
    tool call structures.
    """
    if not output:
        return ""

    # If the output starts with a JSON object or array, try to pretty-print
    stripped = output.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                if "success" in parsed and "output" in parsed:
                    return str(parsed.get("output", ""))
                lines = [f"{k}: {v}" for k, v in parsed.items() if k not in ("success",)]
                return "\n".join(lines)
        except (json.JSONDecodeError, ValueError):
            pass

    # If output looks like a Python dict repr
    if stripped.startswith("{") and "':" in stripped:
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict):
                return str(parsed.get("output", parsed.get("result", str(parsed))))
        except (ValueError, SyntaxError, MemoryError):
            pass

    # Remove common internal prefixes
    prefixes_to_strip = [
        "Tool execution result:",
        "Tool result:",
        "Function call result:",
        "Agent result:",
        "Raw output:",
    ]
    for prefix in prefixes_to_strip:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].strip()

    return stripped
