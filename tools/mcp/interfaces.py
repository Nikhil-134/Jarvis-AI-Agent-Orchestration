"""MCP (Model Context Protocol) abstract interfaces.

These interfaces define how MCP servers and clients integrate with
Jarvis's tool system.  Implementations are added in a future phase;
the architecture ensures that any MCP-compatible server can expose its
tools through the standard :class:`tools.interfaces.IToolRegistry`
without modifying the orchestrator or existing agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tools.interfaces import ITool


class MCPError(Exception):
    """Base exception for MCP-related errors."""


class IMCPServer(ABC):
    """Interface for an MCP server that exposes tools.

    An MCP server is a process or service that provides tool definitions
    and executes tool calls.  In a future phase, implementations will
    connect to local or remote MCP servers via stdio or TCP transports.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the server name."""

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """Return a list of tool definitions from this server.

        Each dict must have ``name``, ``description``, and
        ``inputSchema`` keys (MCP specification).
        """

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server and return its result.

        The result dict should contain at least ``content`` (list of
        content items per the MCP spec) and ``isError`` (bool).
        """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialise the connection to the MCP server."""

    @abstractmethod
    async def close(self) -> None:
        """Close the connection to the MCP server."""


class IMCPClient(ABC):
    """Interface for an MCP client that manages server connections.

    Implementations handle server discovery, capability negotiation,
    and lifecycle management across multiple MCP servers.
    """

    @abstractmethod
    async def connect_server(self, server: IMCPServer) -> None:
        """Connect and initialise an MCP server."""

    @abstractmethod
    async def disconnect_server(self, name: str) -> None:
        """Disconnect and shut down an MCP server by name."""

    @abstractmethod
    def get_server(self, name: str) -> IMCPServer | None:
        """Return a connected server by name, or None."""

    @abstractmethod
    async def sync_tools(self, registry: Any) -> None:
        """Synchronise tools from all connected MCP servers into a tool registry.

        The ``registry`` parameter follows :class:`tools.interfaces.IToolRegistry`.
        """


class IMCPToolProvider(ABC):
    """Interface for a wrapper that adapts MCP tools to ITool.

    In a future phase, implementations will wrap each tool from an MCP
    server as an :class:`ITool` instance that delegates ``execute()`` to
    ``IMCPServer.call_tool()``, enabling seamless integration with the
    existing tool registry and execution engine.
    """
