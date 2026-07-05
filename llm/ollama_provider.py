"""Ollama LLM provider implementation (async, httpx)."""

import json
from collections.abc import AsyncIterable

from llm.base import BaseLLMProvider, LLMConfig
from llm.errors import LLMProviderError
from llm.interfaces import ToolDefinition
from llm.registry import register_provider


@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):
    """LLM provider for a local Ollama chat endpoint."""

    CAPABILITIES = {"streaming", "tool_calling", "json_mode", "vision"}

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> set[str]:
        return self.CAPABILITIES

    async def _generate_once(
        self,
        prompt: str,
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> str:
        payload = self._build_payload(prompt, system_prompt, tools, stream=False)
        response = await self._http_post_json(
            self._url(), payload, {"Content-Type": "application/json"}
        )
        try:
            return str(response["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise LLMProviderError("Ollama response did not include message content.") from exc

    async def _stream_once(
        self,
        prompt: str,
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterable[str]:
        payload = self._build_payload(prompt, system_prompt, tools, stream=True)
        async for event in self._stream_jsonl(payload):
            try:
                chunk = event.get("message", {}).get("content")
            except AttributeError as exc:
                raise LLMProviderError("Ollama stream event was malformed.") from exc
            if chunk:
                yield str(chunk)
            if event.get("done") is True:
                break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
        stream: bool,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": self._build_messages(prompt, system_prompt),
            "stream": stream,
        }
        if tools:
            payload["tools"] = self._build_tools_payload(tools)
        return payload

    async def _stream_jsonl(
        self, payload: dict[str, object]
    ) -> AsyncIterable[dict[str, object]]:
        async for line in self._http_stream_lines(
            self._url(), payload, {"Content-Type": "application/json"}
        ):
            if line:
                yield json.loads(line)

    def _url(self) -> str:
        base = (self.config.base_url or "http://localhost:11434").rstrip("/")
        return f"{base}/api/chat"
