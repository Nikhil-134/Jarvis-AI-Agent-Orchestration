"""Chat session management with conversation history."""

import logging
from dataclasses import dataclass

from llm.base import BaseLLMProvider
from llm.interfaces import LLMResponse, ToolDefinition

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat conversation."""

    role: str
    content: str


class ChatSession:
    """Maintains conversation history around an LLM provider.

    Usage::

        session = ChatSession(provider, system_prompt="You are a helpful assistant.")
        reply = await session.send("Hello!")
        async for chunk in session.stream("Tell me more."):
            print(chunk, end="")
    """

    def __init__(self, provider: BaseLLMProvider, system_prompt: str | None = None) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self._history: list[ChatMessage] = []
        self._prompt_cache: list[str] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """Return immutable conversation history."""
        return tuple(self._history)

    async def send(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Send a prompt, store user and assistant messages, return the LLM response.

        When *tools* are provided the LLM may return tool calls instead
        of (or in addition to) text content in the ``LLMResponse``.
        """
        self._history.append(ChatMessage(role="user", content=prompt))
        self._prompt_cache.append(f"user: {prompt}")
        response: LLMResponse = await self.provider.generate(
            self._conversation_prompt(), self.system_prompt, tools=tools,
        )
        self._history.append(ChatMessage(role="assistant", content=response.content))
        self._prompt_cache.append(f"assistant: {response.content}")
        _logger.debug("Chat session send: %d messages in history", len(self._history))
        return response

    async def stream(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
    ) -> list[str]:
        """Stream a prompt response and store the final assistant message."""
        self._history.append(ChatMessage(role="user", content=prompt))
        self._prompt_cache.append(f"user: {prompt}")
        chunks: list[str] = []
        async for chunk in self.provider.stream(self._conversation_prompt(), self.system_prompt, tools=tools):
            chunks.append(chunk)
        response = "".join(chunks)
        self._history.append(ChatMessage(role="assistant", content=response))
        self._prompt_cache.append(f"assistant: {response}")
        _logger.debug("Chat session stream: %d chunks, %d messages", len(chunks), len(self._history))
        return chunks

    def clear(self) -> None:
        """Clear all conversation history."""
        self._history.clear()
        self._prompt_cache.clear()
        _logger.debug("Chat session history cleared")

    def append_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history and cache."""
        self._history.append(ChatMessage(role=role, content=content))
        self._prompt_cache.append(f"{role}: {content[:200]}")

    def replace_last_assistant(self, content: str) -> None:
        """Replace the last assistant message in history and cache."""
        for i in range(len(self._history) - 1, -1, -1):
            if self._history[i].role == "assistant":
                self._history[i] = ChatMessage(role="assistant", content=content)
                break
        for i in range(len(self._prompt_cache) - 1, -1, -1):
            if self._prompt_cache[i].startswith("assistant:"):
                self._prompt_cache[i] = f"assistant: {content[:200]}"
                break

    def _conversation_prompt(self) -> str:
        """Render history into a provider-neutral prompt string."""
        if self._prompt_cache:
            return "\n".join(self._prompt_cache)
        return "\n".join(f"{message.role}: {message.content}" for message in self._history)
