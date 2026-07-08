"""Tool execution engine — permission checks, execution, logging."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tools.context import ToolContext
from tools.exceptions import ToolExecutionError, ToolNotFoundError, ToolValidationError
from tools.interfaces import IToolRegistry
from tools.permissions import PermissionManager, PermissionLevel
from tools.registry import ToolRegistry

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result of a tool execution."""

    success: bool
    output: str
    tool_name: str
    execution_time_ms: float
    error: str | None = None


class ToolExecutionEngine:
    """Orchestrates tool invocation with permission checks and logging.

    Every tool execution goes through:
    1. Lookup — find tool in registry
    2. Permission check — confirm safe/dangerous
    3. Argument validation — verify required params
    4. Execution — async tool call with timing
    5. Logging — record outcome
    """

    def __init__(
        self,
        registry: IToolRegistry | None = None,
        permission_manager: PermissionManager | None = None,
    ) -> None:
        self._registry = registry or ToolRegistry()
        self._permission_manager = permission_manager or PermissionManager()

    @property
    def registry(self) -> IToolRegistry:
        return self._registry

    @property
    def permission_manager(self) -> PermissionManager:
        return self._permission_manager

    async def execute(
        self,
        name: str,
        _context: ToolContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Look up, authorise, and execute a tool.

        Accepts an optional :class:`ToolContext` that controls timeout
        and other execution parameters.

        Returns a :class:`ToolResult` with timing and outcome.
        Never raises — all errors are captured in the result.
        """
        start = time.monotonic()
        ctx = _context or ToolContext()

        # 1. Lookup
        tool = self._registry.get(name)
        if tool is None:
            elapsed = (time.monotonic() - start) * 1000
            _logger.error("Tool '%s' not found in registry", name)
            return ToolResult(
                success=False,
                output="",
                tool_name=name,
                execution_time_ms=elapsed,
                error=f"Tool '{name}' not found.",
            )

        # 2. Permission check
        try:
            await self._permission_manager.confirm(
                name, tool.permission_level, reason=tool.spec.description
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            _logger.warning("Tool '%s' execution denied: %s", name, exc)
            return ToolResult(
                success=False,
                output="",
                tool_name=name,
                execution_time_ms=elapsed,
                error=str(exc),
            )

        # 3. Validate arguments against schema
        try:
            self._validate_args(kwargs, tool.spec.parameters)
        except ToolValidationError as exc:
            elapsed = (time.monotonic() - start) * 1000
            _logger.error("Tool '%s' argument validation failed: %s", name, exc)
            return ToolResult(
                success=False,
                output="",
                tool_name=name,
                execution_time_ms=elapsed,
                error=str(exc),
            )

        # 4. Execute with timeout
        _logger.info("Executing tool '%s' with args=%s", name, kwargs)
        timeout = ctx.timeout_seconds
        try:
            raw_result = await asyncio.wait_for(
                tool.execute(**kwargs), timeout=timeout,
            )
        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            _logger.error("Tool '%s' timed out after %.1f s", name, timeout)
            return ToolResult(
                success=False,
                output="",
                tool_name=name,
                execution_time_ms=elapsed,
                error=f"Tool '{name}' timed out after {timeout}s.",
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            _logger.exception("Tool '%s' execution raised", name)
            return ToolResult(
                success=False,
                output="",
                tool_name=name,
                execution_time_ms=elapsed,
                error=f"Tool '{name}' execution failed: {exc}",
            )

        elapsed = (time.monotonic() - start) * 1000
        success = raw_result.get("success", True)
        output = raw_result.get("output", str(raw_result))

        # 5. Log
        if success:
            _logger.info(
                "Tool '%s' completed in %.1f ms",
                name,
                elapsed,
            )
        else:
            _logger.error(
                "Tool '%s' failed in %.1f ms: %s",
                name,
                elapsed,
                raw_result.get("error", "unknown"),
            )

        return ToolResult(
            success=success,
            output=output,
            tool_name=name,
            execution_time_ms=elapsed,
            error=raw_result.get("error"),
        )

    @staticmethod
    def _validate_args(args: dict[str, Any], schema: dict[str, Any]) -> None:
        """Basic JSON Schema validation for required properties."""
        required = schema.get("required", [])
        props = schema.get("properties", {})

        missing = [r for r in required if r not in args]
        if missing:
            raise ToolValidationError(f"Missing required arguments: {', '.join(missing)}")

        for key, value in args.items():
            prop_schema = props.get(key, {})
            prop_type = prop_schema.get("type")
            if prop_type and value is not None:
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }
                expected = type_map.get(prop_type)
                if expected and not isinstance(value, expected):
                    raise ToolValidationError(
                        f"Argument '{key}' expected type '{prop_type}', "
                        f"got {type(value).__name__}"
                    )
