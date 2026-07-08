"""Clipboard tool — read, write, and clear the system clipboard."""

from __future__ import annotations

from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False


class ClipboardTool(ITool):
    """Read, write, or clear the system clipboard."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="clipboard",
            description="Access the system clipboard. "
                        "Operations: read (get clipboard contents), "
                        "write (set clipboard contents), "
                        "clear (empty clipboard).",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The clipboard operation to perform.",
                        "enum": ["read", "write", "clear"],
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to write to the clipboard "
                                       "(required for write operation).",
                    },
                },
                "required": ["operation"],
            },
        )

    @property
    def category(self) -> str:
        return "utility"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.DANGEROUS

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not HAS_PYPERCLIP:
            return {
                "success": False,
                "output": "",
                "error": "pyperclip is not installed. Install with: pip install pyperclip",
            }

        operation: str = str(kwargs.get("operation", ""))

        if operation == "read":
            return self._read()
        elif operation == "write":
            text: str = str(kwargs.get("text", ""))
            return self._write(text)
        elif operation == "clear":
            return self._clear()
        else:
            return {
                "success": False,
                "output": "",
                "error": f"Unknown clipboard operation: {operation}. "
                         f"Must be read, write, or clear.",
            }

    def _read(self) -> dict[str, Any]:
        try:
            content = pyperclip.paste()
        except Exception as exc:
            return {"success": False, "output": "", "error": f"Clipboard read failed: {exc}"}
        return {
            "success": True,
            "output": content or "(clipboard is empty)",
            "data": {"length": len(content)},
        }

    def _write(self, text: str) -> dict[str, Any]:
        if not text:
            return {"success": False, "output": "", "error": "No text provided for write operation."}
        try:
            pyperclip.copy(text)
        except Exception as exc:
            return {"success": False, "output": "", "error": f"Clipboard write failed: {exc}"}
        return {
            "success": True,
            "output": f"Written {len(text)} characters to clipboard.",
            "data": {"length": len(text)},
        }

    def _clear(self) -> dict[str, Any]:
        try:
            pyperclip.copy("")
        except Exception as exc:
            return {"success": False, "output": "", "error": f"Clipboard clear failed: {exc}"}
        return {"success": True, "output": "Clipboard cleared."}
