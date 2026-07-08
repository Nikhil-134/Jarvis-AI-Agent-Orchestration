"""UUID tool — generates UUIDs."""

from __future__ import annotations

import uuid
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec


class UuidTool(ITool):
    """Generates UUIDs in various formats."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="uuid",
            description="Generate UUIDs. Returns a version 4 UUID by default.",
            parameters={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of UUIDs to generate (default: 1, max: 100).",
                    },
                    "version": {
                        "type": "integer",
                        "description": "UUID version: 4 (random, default) or 1 (time-based).",
                        "enum": [1, 4],
                    },
                },
            },
        )

    @property
    def category(self) -> str:
        return "utility"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        count = min(int(kwargs.get("count", 1)), 100)
        version = int(kwargs.get("version", 4))

        if version not in (1, 4):
            return {"success": False, "output": "", "error": "Unsupported UUID version. Use 1 or 4."}

        uuids: list[str] = []
        for _ in range(count):
            if version == 1:
                uuids.append(str(uuid.uuid1()))
            else:
                uuids.append(str(uuid.uuid4()))

        output = "\n".join(uuids)
        return {
            "success": True,
            "output": output,
            "data": {"count": count, "version": version, "uuids": uuids},
        }
