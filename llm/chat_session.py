"""Chat session management with conversation history."""

from dataclasses import dataclass

from llm.base import BaseLLMProvider


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat conversation."""

    role: str
    content: str


class ChatSession:
    """Maintains conversation history around an LLM provider."""

    def __init__(self, provider: BaseLLMProvider, system_prompt: str | None = None) -> None:
        """Initialize a chat session for a provider."""
        self.provider = provider
        self.system_prompt = system_prompt
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """Return immutable conversation history."""
        return tuple(self._history)

    def send(self, prompt: str) -> str:
        """Send a prompt, store user and assistant messages, and return the response."""
        self._history.append(ChatMessage(role="user", content=prompt))
        response = self.provider.generate(self._conversation_prompt(), self.system_prompt)
        self._history.append(ChatMessage(role="assistant", content=response))
        return response

    def stream(self, prompt: str) -> list[str]:
        """Stream a prompt response and store the final assistant message."""
        self._history.append(ChatMessage(role="user", content=prompt))
        chunks = list(self.provider.stream(self._conversation_prompt(), self.system_prompt))
        self._history.append(ChatMessage(role="assistant", content="".join(chunks)))
        return chunks

    def clear(self) -> None:
        """Clear all conversation history."""
        self._history.clear()

    def _conversation_prompt(self) -> str:
        """Render history into a provider-neutral prompt."""
        return "\n".join(f"{message.role}: {message.content}" for message in self._history)


if __name__ == "__main__":
    from llm.base import LLMConfig

    class EchoProvider(BaseLLMProvider):
        @property
        def name(self) -> str:
            return "echo"

        def generate(self, prompt: str, system_prompt: str | None = None) -> str:
            return prompt

        def stream(self, prompt: str, system_prompt: str | None = None) -> list[str]:
            return [prompt]

    session = ChatSession(EchoProvider(LLMConfig(provider="echo", model="echo")))
    print(session.send("demo"))
