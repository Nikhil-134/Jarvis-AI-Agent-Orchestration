"""Base64 tool — encode and decode Base64 data."""

from __future__ import annotations

import base64
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec


class Base64Tool(ITool):
    """Base64 encode and decode operations."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="base64",
            description="Encode or decode Base64 data.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation: encode or decode.",
                        "enum": ["encode", "decode"],
                    },
                    "data": {
                        "type": "string",
                        "description": "The input data to encode or decode.",
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
        operation = str(kwargs.get("operation", "")).lower()
        data = str(kwargs.get("data", ""))

        if not operation:
            return {"success": False, "output": "", "error": "No operation specified."}
        if not data:
            return {"success": False, "output": "", "error": "No data provided."}

        try:
            if operation == "encode":
                encoded = base64.b64encode(data.encode("utf-8")).decode("ascii")
                return {
                    "success": True,
                    "output": encoded,
                    "data": {"operation": "encode", "original_length": len(data), "encoded_length": len(encoded)},
                }
            elif operation == "decode":
                decoded = base64.b64decode(data).decode("utf-8")
                return {
                    "success": True,
                    "output": decoded,
                    "data": {"operation": "decode", "encoded_length": len(data), "decoded_length": len(decoded)},
                }
            else:
                return {"success": False, "output": "", "error": f"Unknown operation: {operation}"}
        except (base64.binascii.Error, UnicodeDecodeError) as exc:
            return {"success": False, "output": "", "error": f"Base64 {operation} failed: {exc}"}
        except Exception as exc:
            return {"success": False, "output": "", "error": f"Base64 {operation} failed: {exc}"}
