"""Date & Time tool — returns current date and time information."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec


class DateTimeTool(ITool):
    """Returns the current date, time, and timezone information."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="datetime",
            description="Get the current date, time, and timezone. Optionally specify a strftime format string.",
            parameters={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Optional strftime format (e.g. '%Y-%m-%d %H:%M:%S %Z'). Defaults to ISO 8601.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Optional timezone name (e.g. 'UTC', 'America/New_York'). Defaults to local time.",
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
        fmt = str(kwargs.get("format", "")) or None
        tz_name = str(kwargs.get("timezone", "")) or None

        now = datetime.now(timezone.utc).astimezone() if tz_name is None else datetime.now(timezone.utc)

        if tz_name:
            try:
                import zoneinfo
                desired_tz = zoneinfo.ZoneInfo(tz_name)
                now = now.astimezone(desired_tz)
            except (KeyError, TypeError, ImportError):
                return {
                    "success": False,
                    "output": "",
                    "error": f"Unknown timezone: {tz_name}",
                }

        if fmt:
            try:
                formatted = now.strftime(fmt)
            except Exception as exc:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Invalid format string: {exc}",
                }
        else:
            formatted = now.isoformat()

        return {
            "success": True,
            "output": formatted,
            "data": {
                "iso": now.isoformat(),
                "timestamp": now.timestamp(),
                "timezone": str(now.tzinfo or "local"),
                "formatted": formatted,
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "weekday": now.strftime("%A"),
            },
        }
