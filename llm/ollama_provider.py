"""Ollama LLM provider implementation (async, httpx)."""

import json
from collections.abc import AsyncIterable
from typing import Any

from llm.base import BaseLLMProvider, LLMConfig
from llm.errors import LLMProviderError
from llm.interfaces import LLMResponse, ToolCall, ToolDefinition
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
    ) -> LLMResponse:
        payload = self._build_payload(prompt, system_prompt, tools, stream=False)
        response = await self._http_post_json(
            self._url(), payload, {"Content-Type": "application/json"}
        )
        try:
            message = response["message"]
            content = str(message.get("content", ""))
            tool_calls = self._parse_tool_calls(message.get("tool_calls", []))
            return LLMResponse(content=content, tool_calls=tuple(tool_calls))
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
    # Tool call parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse Ollama's ``tool_calls`` response format into ToolCall instances."""
        result: list[ToolCall] = []
        for call in raw:
            func = call.get("function", {})
            name = str(func.get("name", ""))
            raw_args = func.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    raw_args = {}
            if name:
                result.append(ToolCall(name=name, arguments=dict(raw_args)))
        return result

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
        # Prevent runaway responses — set a generous max_tokens bound
        if self.config.max_tokens is not None:
            payload["options"] = {"num_predict": self.config.max_tokens}
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
