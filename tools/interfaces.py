"""Tool system interface definitions for Jarvis.

These interfaces define contracts for tool registration and execution.
Implementations (built-in tools, MCP tools, plugin tools) will be
added in future phases.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Specification of a tool's metadata and JSON Schema parameters."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


class ITool(ABC):
    """Interface for an executable tool.

    Implementations: file-system tools, web-search tools, code executors.
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool's specification (name, description, schema)."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with the supplied keyword arguments.

        Returns a dict with at least ``"success": bool`` and
        ``"output": str | dict``.
        """


class IToolRegistry(ABC):
    """Interface for registering and discovering tools.

    Implementations: ToolRegistry (in-memory dict).
    """

    @abstractmethod
    def register(self, tool: ITool) -> None:
        """Register a tool by its spec.name."""

    @abstractmethod
    def unregister(self, name: str) -> None:
        """Remove a tool by name."""

    @abstractmethod
    def get(self, name: str) -> ITool | None:
        """Return a tool by name, or None."""

    @abstractmethod
    def list_specs(self) -> list[ToolSpec]:
        """Return specs for all registered tools."""
