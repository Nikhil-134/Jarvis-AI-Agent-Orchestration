"""JSON tool — validate, pretty-print, and convert JSON data."""

from __future__ import annotations

import json
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec


class JsonTool(ITool):
    """JSON operations: validate, pretty_print, convert."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="json",
            description="JSON operations: validate, pretty_print, convert (dict to JSON or JSON to dict).",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation: validate, pretty_print, convert.",
                        "enum": ["validate", "pretty_print", "convert"],
                    },
                    "data": {
                        "type": "string",
                        "description": "The JSON string or Python dict representation to process.",
                    },
                    "indent": {
                        "type": "integer",
                        "description": "Indentation level for pretty_print (default: 2).",
                    },
                },
                "required": ["operation", "data"],
            },
        )

    @property
    def category(self) -> str:
        return "utility"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        operation = str(kwargs.get("operation", ""))
        data_str = str(kwargs.get("data", ""))

        if not operation:
            return {"success": False, "output": "", "error": "No operation specified."}
        if not data_str:
            return {"success": False, "output": "", "error": "No data provided."}

        if operation == "validate":
            try:
                parsed = json.loads(data_str)
                return {
                    "success": True,
                    "output": "Valid JSON",
                    "data": {"valid": True, "type": type(parsed).__name__},
                }
            except json.JSONDecodeError as exc:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Invalid JSON: {exc}",
                    "data": {"valid": False, "error": str(exc)},
                }

        if operation == "pretty_print":
            indent = int(kwargs.get("indent", 2))
            try:
                parsed = json.loads(data_str)
                formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
                return {
                    "success": True,
                    "output": formatted,
                    "data": {"indent": indent, "original_length": len(data_str), "formatted_length": len(formatted)},
                }
            except json.JSONDecodeError as exc:
                return {"success": False, "output": "", "error": f"Invalid JSON: {exc}"}

        if operation == "convert":
            try:
                parsed = json.loads(data_str)
                result: dict[str, Any] = {"success": True, "data": {"type": type(parsed).__name__}}
                if isinstance(parsed, dict):
                    result["output"] = "\n".join(f"{k}: {v}" for k, v in parsed.items())
                elif isinstance(parsed, list):
                    result["output"] = "\n".join(str(item) for item in parsed)
                else:
                    result["output"] = str(parsed)
                return result
            except json.JSONDecodeError:
                return {"success": False, "output": "", "error": "Data is not valid JSON. Provide a valid JSON string."}

        return {"success": False, "output": "", "error": f"Unknown operation: {operation}"}
