"""LLM provider package exports."""

from typing import Any

__all__ = [
    "BaseLLMProvider",
    "ChatMessage",
    "ChatSession",
    "ILLMProvider",
    "IProviderRegistry",
    "LLMConfig",
    "LLMError",
    "LLMProviderError",
    "LLMTimeoutError",
    "OllamaProvider",
    "OpenAIProvider",
    "PromptManager",
    "ToolDefinition",
    "build_llm_provider",
]


def __getattr__(name: str) -> Any:
    """Load LLM exports lazily so provider modules remain runnable."""
    if name in {"BaseLLMProvider", "LLMConfig"}:
        from llm.base import BaseLLMProvider, LLMConfig

        return {"BaseLLMProvider": BaseLLMProvider, "LLMConfig": LLMConfig}[name]
    if name == "ChatMessage":
        from llm.chat_session import ChatMessage
        return ChatMessage
    if name in {"ChatSession"}:
        from llm.chat_session import ChatSession
        return ChatSession
    if name in {"ILLMProvider", "IProviderRegistry", "ToolDefinition"}:
        from llm.interfaces import ILLMProvider, IProviderRegistry, ToolDefinition
        return {
            "ILLMProvider": ILLMProvider,
            "IProviderRegistry": IProviderRegistry,
            "ToolDefinition": ToolDefinition,
        }[name]
    if name in {"LLMError", "LLMProviderError", "LLMTimeoutError"}:
        from llm.errors import LLMError, LLMProviderError, LLMTimeoutError
        return {
            "LLMError": LLMError,
            "LLMProviderError": LLMProviderError,
            "LLMTimeoutError": LLMTimeoutError,
        }[name]
    if name == "OllamaProvider":
        from llm.ollama_provider import OllamaProvider
        return OllamaProvider
    if name == "OpenAIProvider":
        from llm.openai_provider import OpenAIProvider
        return OpenAIProvider
    if name == "PromptManager":
        from llm.prompt_manager import PromptManager
        return PromptManager
    if name == "build_llm_provider":
        from llm.factory import build_llm_provider
        return build_llm_provider
    raise AttributeError(f"module 'llm' has no attribute {name!r}")
