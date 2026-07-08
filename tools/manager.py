"""ToolManager — high-level facade for the tool system.

Combines registry, execution engine, permissions, and discovery into a
single entry point used by agents and the orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.context import ToolContext
from tools.engine import ToolExecutionEngine, ToolResult
from tools.exceptions import ToolError, ToolNotFoundError
from tools.interfaces import ITool, IToolRegistry, PermissionLevel, ToolSpec
from tools.permissions import PermissionManager
from tools.registry import ToolRegistry

_logger = logging.getLogger(__name__)


class ToolManager:
    """Unified facade over the tool system.

    Wires :class:`ToolRegistry`, :class:`ToolExecutionEngine`, and
    :class:`PermissionManager` together.  Supports per-tool enable/disable
    and configurable execution context.

    Usage::

        mgr = ToolManager()
        mgr.register_tool(CalculatorTool())
        result = await mgr.execute("calculator", expression="2+2")
    """

    def __init__(
        self,
        registry: IToolRegistry | None = None,
        permission_manager: PermissionManager | None = None,
        enabled_tools: set[str] | None = None,
        disabled_tools: set[str] | None = None,
        default_timeout: float = 30.0,
    ) -> None:
        self._registry = registry or ToolRegistry()
        self._permission_manager = permission_manager or PermissionManager()
        self._engine = ToolExecutionEngine(
            registry=self._registry,
            permission_manager=self._permission_manager,
        )
        self._enabled_tools: set[str] | None = enabled_tools
        self._disabled_tools: set[str] = disabled_tools or set()
        self._default_timeout = default_timeout

    # ------------------------------------------------------------------
    # Registry delegation
    # ------------------------------------------------------------------

    @property
    def registry(self) -> IToolRegistry:
        return self._registry

    @property
    def engine(self) -> ToolExecutionEngine:
        return self._engine

    @property
    def permission_manager(self) -> PermissionManager:
        return self._permission_manager

    def register_tool(self, tool: ITool) -> None:
        self._registry.register(tool)
        _logger.info("ToolManager registered tool '%s'", tool.spec.name)

    def register_tools(self, tools: list[ITool]) -> None:
        for tool in tools:
            self.register_tool(tool)

    def unregister_tool(self, name: str) -> None:
        self._registry.unregister(name)

    def get_tool(self, name: str) -> ITool | None:
        return self._registry.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return self._registry.list_specs()

    def list_categories(self) -> set[str]:
        return self._registry.get_categories()

    def get_tools_by_category(self, category: str) -> list[ITool]:
        return self._registry.get_by_category(category)

    @property
    def tool_count(self) -> int:
        return self._registry.count

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable_tool(self, name: str) -> None:
        self._disabled_tools.discard(name)

    def disable_tool(self, name: str) -> None:
        self._disabled_tools.add(name)

    def is_tool_enabled(self, name: str) -> bool:
        if self._enabled_tools is not None and name not in self._enabled_tools:
            return False
        if name in self._disabled_tools:
            return False
        return True

    def set_enabled_tools(self, names: set[str]) -> None:
        self._enabled_tools = names

    def set_disabled_tools(self, names: set[str]) -> None:
        self._disabled_tools = names

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        name: str,
        context: ToolContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a tool by name with optional :class:`ToolContext`.

        Checks enable/disable status first.  Passes timeout from context
        to the engine.
        """
        if not self.is_tool_enabled(name):
            _logger.warning("Tool '%s' is disabled", name)
            return ToolResult(
                success=False,
                output="",
                tool_name=name,
                execution_time_ms=0.0,
                error=f"Tool '{name}' is disabled.",
            )

        ctx = context or ToolContext(timeout_seconds=self._default_timeout)
        return await self._engine.execute(name, _context=ctx, **kwargs)

    async def execute_many(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        context: ToolContext | None = None,
    ) -> list[ToolResult]:
        """Execute multiple tools sequentially."""
        results: list[ToolResult] = []
        for name, kwargs in calls:
            result = await self.execute(name, context=context, **kwargs)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_specs_for_llm(self) -> list[dict[str, Any]]:
        """Return tool specs in LLM-compatible format (OpenAI tool format).

        Only returns enabled tools.
        """
        specs = self._registry.list_specs()
        result: list[dict[str, Any]] = []
        for s in specs:
            if self.is_tool_enabled(s.name):
                result.append({
                    "type": "function",
                    "function": {
                        "name": s.name,
                        "description": s.description,
                        "parameters": s.parameters,
                    },
                })
        return result

    def health(self) -> dict[str, Any]:
        return {
            "tool_count": self._registry.count,
            "enabled_count": sum(1 for s in self._registry.list_specs() if self.is_tool_enabled(s.name)),
            "categories": sorted(self._registry.get_categories()),
            "tools": [s.name for s in self._registry.list_specs()],
        }
