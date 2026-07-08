"""Screenshot tool — captures the primary monitor display."""

from __future__ import annotations

import base64
import io
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec

try:
    from PIL import ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ScreenshotTool(ITool):
    """Capture a screenshot of the primary monitor.

    Returns a base64-encoded PNG by default, or saves to a file if
    ``save_path`` is provided.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="screenshot",
            description="Take a screenshot of the primary monitor. "
                        "Returns a base64-encoded PNG image. "
                        "Optionally saves to a file path instead.",
            parameters={
                "type": "object",
                "properties": {
                    "save_path": {
                        "type": "string",
                        "description": "Optional file path to save the screenshot to "
                                       "(e.g. '/tmp/screenshot.png'). "
                                       "If provided, returns the file path instead of base64.",
                    },
                },
            },
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.DANGEROUS

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not HAS_PIL:
            return {
                "success": False,
                "output": "",
                "error": "Pillow is not installed. Install with: pip install Pillow",
            }

        save_path: str | None = kwargs.get("save_path")

        try:
            img = ImageGrab.grab()
        except Exception as exc:
            return {
                "success": False,
                "output": "",
                "error": f"Screenshot failed: {exc}",
            }

        if save_path:
            try:
                img.save(save_path, "PNG")
                return {
                    "success": True,
                    "output": f"Screenshot saved to {save_path}",
                    "data": {"file_path": save_path},
                }
            except Exception as exc:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Failed to save screenshot: {exc}",
                }

        buffer = io.BytesIO()
        try:
            img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as exc:
            return {
                "success": False,
                "output": "",
                "error": f"Failed to encode screenshot: {exc}",
            }

        return {
            "success": True,
            "output": b64,
            "data": {
                "format": "base64",
                "mime_type": "image/png",
                "size_bytes": len(b64),
            },
        }
