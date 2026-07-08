"""Safe Shell tool — executes whitelisted shell commands only."""

from __future__ import annotations

import asyncio
import shlex
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec

_WHITELIST_COMMANDS: frozenset[str] = frozenset({
    "ls", "dir", "echo", "pwd", "whoami", "hostname", "date",
    "uname", "cat", "head", "tail", "wc", "sort", "find", "grep",
    "where", "which",
})


class ShellTool(ITool):
    """Executes a whitelisted shell command and returns its output.

    Only the following commands are allowed: {0}.
    """.format(", ".join(sorted(_WHITELIST_COMMANDS)))

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell",
            description="Execute a whitelisted shell command and return its output. "
                         f"Allowed commands: {', '.join(sorted(_WHITELIST_COMMANDS))}.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. The base command must be in the whitelist.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (default: 15).",
                    },
                },
                "required": ["command"],
            },
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.DANGEROUS

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        command = str(kwargs.get("command", "")).strip()
        timeout = int(kwargs.get("timeout_seconds", 15))

        if not command:
            return {"success": False, "output": "", "error": "No command provided."}

        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return {"success": False, "output": "", "error": f"Invalid command syntax: {exc}"}

        base = parts[0].lower() if parts else ""
        if base not in _WHITELIST_COMMANDS:
            return {
                "success": False,
                "output": "",
                "error": f"Command '{base}' is not in the whitelist. "
                         f"Allowed: {', '.join(sorted(_WHITELIST_COMMANDS))}",
            }

        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "output": "",
                "error": f"Command timed out after {timeout}s.",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": f"Command not found: {base}",
            }
        except Exception as exc:
            return {
                "success": False,
                "output": "",
                "error": f"Command execution failed: {exc}",
            }

        output_text = stdout.decode("utf-8", errors="replace").strip()
        error_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            return {
                "success": False,
                "output": output_text or "",
                "error": error_text or f"Command exited with code {proc.returncode}",
            }

        return {
            "success": True,
            "output": output_text or "(command produced no output)",
            "data": {
                "return_code": proc.returncode,
                "stderr": error_text,
            },
        }
