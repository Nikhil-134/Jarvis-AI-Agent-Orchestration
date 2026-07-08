"""MCP (Model Context Protocol) preparation package."""

from tools.mcp.interfaces import IMCPClient, IMCPServer, IMCPToolProvider, MCPError

__all__ = [
    "IMCPServer",
    "IMCPClient",
    "IMCPToolProvider",
    "MCPError",
]

MCP_TOOL_PROVIDER_CATEGORY = "mcp"
