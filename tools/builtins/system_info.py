"""System Information tool — returns OS, CPU, memory, and Python info."""

from __future__ import annotations

import os
import platform
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec


class SystemInfoTool(ITool):
    """Returns system information including OS, CPU, memory, and Python version."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="system_info",
            description="Get system information: OS, CPU count, Python version, platform details, and environment variables (filtered list).",
            parameters={
                "type": "object",
                "properties": {
                    "include_env": {
                        "type": "boolean",
                        "description": "Whether to include environment variable names (not values) in output (default: false).",
                    },
                },
            },
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        include_env = bool(kwargs.get("include_env", False))

        info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "cpu_count": os.cpu_count() or 0,
            "cwd": os.getcwd(),
            "pid": os.getpid(),
        }

        if include_env:
            info["environment_variables"] = sorted(os.environ.keys())

        lines = [
            f"OS: {info['os']} {info['os_release']}",
            f"Architecture: {info['architecture']}",
            f"Hostname: {info['hostname']}",
            f"Python: {info['python_implementation']} {info['python_version']}",
            f"CPU Cores: {info['cpu_count']}",
            f"Processor: {info['processor']}",
            f"Working Directory: {info['cwd']}",
            f"PID: {info['pid']}",
        ]

        return {
            "success": True,
            "output": "\n".join(lines),
            "data": info,
        }
