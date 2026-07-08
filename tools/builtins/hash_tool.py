"""Hash tool — computes cryptographic hashes."""

from __future__ import annotations

import hashlib
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec


class HashTool(ITool):
    """Computes cryptographic hashes (SHA-256, MD5)."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="hash",
            description="Compute cryptographic hashes. Supports SHA-256 and MD5 algorithms.",
            parameters={
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "description": "Hash algorithm: sha256, md5.",
                        "enum": ["sha256", "md5"],
                    },
                    "data": {
                        "type": "string",
                        "description": "The input data to hash.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Output encoding: hex (default) or base64.",
                        "enum": ["hex", "base64"],
                    },
                },
                "required": ["algorithm", "data"],
            },
        )

    @property
    def category(self) -> str:
        return "utility"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        algorithm = str(kwargs.get("algorithm", "")).lower()
        data = str(kwargs.get("data", ""))
        encoding = str(kwargs.get("encoding", "hex")).lower()

        if not algorithm:
            return {"success": False, "output": "", "error": "No algorithm specified."}
        if not data:
            return {"success": False, "output": "", "error": "No data provided."}

        try:
            if algorithm == "sha256":
                h = hashlib.sha256(data.encode("utf-8"))
            elif algorithm == "md5":
                h = hashlib.md5(data.encode("utf-8"))
            else:
                return {"success": False, "output": "", "error": f"Unsupported algorithm: {algorithm}"}

            if encoding == "base64":
                import base64
                result = base64.b64encode(h.digest()).decode("ascii")
            else:
                result = h.hexdigest()

            return {
                "success": True,
                "output": result,
                "data": {"algorithm": algorithm, "encoding": encoding, "hash": result},
            }
        except Exception as exc:
            return {"success": False, "output": "", "error": f"Hash computation failed: {exc}"}
