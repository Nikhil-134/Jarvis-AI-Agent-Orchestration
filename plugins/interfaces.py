"""Plugin system interface definitions for Jarvis.

These interfaces define the contract for third-party plugins.
Implementations (plugin manager, entry-point discovery) will be
added in a future phase.
"""

from abc import ABC, abstractmethod
from typing import Any


class IPlugin(ABC):
    """Interface for a Jarvis plugin.

    Plugins can register agents, tools, memory backends, LLM providers,
    or middleware hooks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the plugin version string."""

    @abstractmethod
    async def initialize(self, registry: Any) -> None:
        """Initialise the plugin with access to the Jarvis component registry.

        The *registry* object provides ``register_agent``, ``register_tool``,
        ``register_llm_provider``, etc.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up plugin resources."""
