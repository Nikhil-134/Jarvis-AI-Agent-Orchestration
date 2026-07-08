"""Desktop notification tool — sends OS-level notifications."""

from __future__ import annotations

from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec

try:
    from plyer import notification as plyer_notification
    HAS_PLYER = True
except ImportError:
    HAS_PLYER = False

try:
    from win10toast import ToastNotifier
    HAS_WIN10TOAST = True
except ImportError:
    HAS_WIN10TOAST = False


class NotificationTool(ITool):
    """Send a desktop notification.

    Uses ``plyer`` first, falls back to ``win10toast`` on Windows.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="notification",
            description="Send a desktop notification. "
                        "Requires plyer (cross-platform) or win10toast (Windows).",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Notification title.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Notification body text.",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Notification display duration in seconds (default: 5).",
                    },
                },
                "required": ["title", "message"],
            },
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        title: str = str(kwargs.get("title", ""))
        message: str = str(kwargs.get("message", ""))
        duration: int = int(kwargs.get("duration", 5))

        if not title or not message:
            return {
                "success": False,
                "output": "",
                "error": "Both title and message are required.",
            }

        if HAS_PLYER:
            return self._notify_plyer(title, message, duration)
        elif HAS_WIN10TOAST:
            return self._notify_win10toast(title, message, duration)
        else:
            return {
                "success": False,
                "output": "",
                "error": "No notification backend available. "
                         "Install plyer (pip install plyer) "
                         "or win10toast (pip install win10toast).",
            }

    def _notify_plyer(self, title: str, message: str, duration: int) -> dict[str, Any]:
        try:
            plyer_notification.notify(
                title=title,
                message=message,
                timeout=duration,
            )
        except Exception as exc:
            return {
                "success": False,
                "output": "",
                "error": f"plyer notification failed: {exc}",
            }
        return {"success": True, "output": f"Notification sent: {title}"}

    def _notify_win10toast(self, title: str, message: str, duration: int) -> dict[str, Any]:
        try:
            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=duration, threaded=True)
        except Exception as exc:
            return {
                "success": False,
                "output": "",
                "error": f"win10toast notification failed: {exc}",
            }
        return {"success": True, "output": f"Notification sent: {title}"}
