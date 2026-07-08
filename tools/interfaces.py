"""Tool system interface definitions for Jarvis.

Defines contracts for tool registration, execution, permissions,
and capability discovery.  All tool implementations follow these
interfaces so that built-in, plugin, and MCP tools are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class PermissionLevel(IntEnum):
    """Permission level for a tool.

    SAFE tools execute without user confirmation.
    DANGEROUS tools require explicit user confirmation.
    """

    SAFE = 0
    DANGEROUS = 1


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Specification of a tool's metadata and JSON Schema parameters."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


class ITool(ABC):
    """Interface for an executable tool."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool's specification (name, description, schema)."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Return the tool's category for grouping/discovery."""

    @property
    @abstractmethod
    def permission_level(self) -> PermissionLevel:
        """Return the tool's permission level (SAFE or DANGEROUS)."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with the supplied keyword arguments.

        Returns a dict with at least ``"success": bool`` and
        ``"output": str | dict``.
        """


class IToolRegistry(ABC):
    """Interface for registering and discovering tools."""

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


class IToolExecutionEngine(ABC):
    """Interface for the tool execution engine."""

    @property
    @abstractmethod
    def registry(self) -> IToolRegistry:
        """Return the underlying tool registry."""

    @abstractmethod
    async def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Look up, authorise, and execute a tool.

        Returns a dict with keys: ``success``, ``output``, ``execution_time_ms``.
        Never raises — all errors are captured in the result.
        """
