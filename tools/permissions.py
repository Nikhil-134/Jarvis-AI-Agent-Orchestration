"""Permission system for tool execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from tools.exceptions import ToolPermissionDeniedError
from tools.interfaces import PermissionLevel


class PermissionManager:
    """Manages permission checks for tool execution.

    Delegates confirmation to an optional async callback so that both
    CLI (``input()``) and automated (auto-approve / auto-deny) modes are
    supported without changing this class.
    """

    def __init__(
        self,
        confirmation_callback: Callable[[str, str], Awaitable[bool]] | None = None,
        auto_approve_dangerous: bool = False,
    ) -> None:
        self._callback = confirmation_callback
        self._auto_approve = auto_approve_dangerous

    async def confirm(
        self,
        tool_name: str,
        permission_level: PermissionLevel,
        reason: str = "",
    ) -> None:
        """Raise :class:`ToolPermissionDeniedError` if execution is not allowed.

        SAFE tools always pass.  DANGEROUS tools require confirmation
        via the configured callback (or raise if no callback is set and
        auto-approve is off).
        """
        if permission_level == PermissionLevel.SAFE:
            return

        if self._auto_approve:
            return

        if self._callback is None:
            raise ToolPermissionDeniedError(
                f"Execution of '{tool_name}' requires permission "
                f"but no confirmation callback is configured."
            )

        approved = await self._callback(tool_name, reason)
        if not approved:
            raise ToolPermissionDeniedError(
                f"Execution of '{tool_name}' was denied by the user."
            )
