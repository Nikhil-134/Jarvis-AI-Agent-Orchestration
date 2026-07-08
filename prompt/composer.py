"""Response composer — wraps raw results in natural conversational text."""

from __future__ import annotations

from typing import Any

from agents.contracts import AgentResult


_TOOL_FRIENDLY: dict[str, str] = {
    "calculator": "calculation result",
    "datetime": "current date and time",
    "uuid": "generated UUID",
    "base64": "Base64 operation result",
    "hash": "hash result",
    "json": "JSON operation result",
    "text": "text processing result",
    "shell": "command output",
    "system_info": "system information",
    "file_system": "file system operation result",
}


def compose(result: AgentResult) -> str:
    """Wrap *result* in natural conversational text suitable for CLI display.

    Never exposes internal fields (plan, status, memory flags) or raw JSON.
    """
    if not result.success:
        return _compose_failure(result)

    data = result.data or {}

    # Planner response — the "response" key contains LLM-generated text
    if "response" in data:
        response = data["response"]
        if isinstance(response, str) and response.strip():
            return response.strip()
        if isinstance(response, dict) and "response" in response:
            text = response["response"]
            if isinstance(text, str) and text.strip():
                return text.strip()

    # Tool response with tool_name metadata
    tool_name = data.get("tool_name", result.data.get("tool_name", ""))
    if tool_name and data.get("output"):
        friendly = _TOOL_FRIENDLY.get(tool_name, f"{tool_name} result")
        return f"Here is the {friendly}:\n\n{data['output'].strip()}"

    # Generic fallback — never show internal messages
    if data:
        output = data.get("output", data.get("message", ""))
        if isinstance(output, str) and output.strip():
            return output.strip()

    # Last resort: use result.message if it doesn't look internal
    msg = (result.message or "").strip()
    if msg and msg != "Planning completed." and not msg.startswith("Tool '") and "completed" not in msg.lower():
        return msg

    return _compose_success_without_output(result)


def _compose_failure(result: AgentResult) -> str:
    """Compose a user-facing error response."""
    data = result.data or {}
    error = data.get("error", result.message or "")
    return f"I ran into an issue: {error}"


def _compose_success_without_output(result: AgentResult) -> str:
    """Compose a generic success response when there's no specific output."""
    data = result.data or {}
    tool_name = data.get("tool_name", "")
    if tool_name:
        return f"The {tool_name} operation completed successfully."
    return "Done. Is there anything else I can help with?"
