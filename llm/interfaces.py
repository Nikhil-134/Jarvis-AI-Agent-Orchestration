"""LLM provider interface definitions for Jarvis."""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call returned by an LLM provider."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured response from an LLM provider.

    When the provider returns a conversational response, ``content``
    contains the text.  When the provider selects a tool, ``tool_calls``
    contains the selected tool invocations (and ``content`` is typically
    empty).

    Contract: ``content`` is *always* a ``str`` and ``tool_calls`` is *always*
    a ``tuple`` — even when a provider hands us ``None``. This is the single
    enforcement point that keeps ``None`` out of every downstream ``.strip()``,
    ``.split()`` and string-format call in the runtime.
    """

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise a provider that returned None / a non-string content.
        if self.content is None:
            object.__setattr__(self, "content", "")
        elif not isinstance(self.content, str):
            object.__setattr__(self, "content", str(self.content))
        # Normalise tool_calls to an immutable tuple (never None).
        if self.tool_calls is None:
            object.__setattr__(self, "tool_calls", ())
        elif not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


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
    ) -> LLMResponse:
        """Generate a complete response for a prompt.

        If *tools* are supplied the provider MAY return tool-call
        content as :class:`ToolCall` instances inside the returned
        :class:`LLMResponse`.
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
