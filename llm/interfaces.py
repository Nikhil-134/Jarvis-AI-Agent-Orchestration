"""LLM provider interface definitions for Jarvis."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Schema for a tool that an LLM provider can call.

    ``parameters`` must follow JSON Schema (draft-2020-12 or equivalent)
    so that the provider can serialise it into the native tool format.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


class ILLMProvider(ABC):
    """Interface all LLM providers must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g. 'ollama', 'openai')."""

    @property
    @abstractmethod
    def capabilities(self) -> set[str]:
        """Return the set of capabilities this provider supports.

        Standard capability tags:

        - ``"streaming"``  – provider supports ``stream()``
        - ``"tool_calling"`` – provider supports tool/function definitions
        - ``"vision"`` – provider accepts image inputs
        - ``"json_mode"`` – provider can constrain output to JSON
        """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> str:
        """Generate a complete response for a prompt.

        If *tools* are supplied the provider MAY return tool-call
        content serialised as text.  Structured tool-call handling
        will be added in a later phase.
        """

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterable[str]:
        """Stream response chunks for a prompt."""


class IProviderRegistry(ABC):
    """Interface for a self-registering provider registry."""

    @abstractmethod
    def register(self, name: str, provider_cls: type[ILLMProvider]) -> None:
        """Register a provider class under a name."""

    @abstractmethod
    def get(self, name: str) -> type[ILLMProvider]:
        """Return the provider class registered under *name*."""

    @abstractmethod
    def all(self) -> dict[str, type[ILLMProvider]]:
        """Return all registered providers mapped by name."""
