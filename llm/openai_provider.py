"""OpenAI-compatible LLM provider implementation (async, httpx)."""

import json
from collections.abc import AsyncIterable
from typing import Any

from llm.base import BaseLLMProvider, LLMConfig
from llm.errors import LLMProviderError
from llm.interfaces import LLMResponse, ToolCall, ToolDefinition
from llm.registry import register_provider


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """LLM provider for OpenAI-compatible chat completions."""

    CAPABILITIES = {"streaming", "tool_calling", "json_mode"}

    @property
    def name(self) -> str:
        return "openai"

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
        response = await self._http_post_json(self._url(), payload, self._headers())
        try:
            message = response["choices"][0]["message"]
            content = str(message.get("content") or "")
            tool_calls = self._parse_tool_calls(message.get("tool_calls", []))
            return LLMResponse(content=content, tool_calls=tuple(tool_calls))
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("OpenAI response did not include message content.") from exc

    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse OpenAI's ``tool_calls`` response format into ToolCall instances.

        OpenAI returns tool_calls as::
            [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
        """
        result: list[ToolCall] = []
        for call in raw:
            func = call.get("function", {})
            name = str(func.get("name", ""))
            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    raw_args = {}
            if name:
                result.append(ToolCall(name=name, arguments=dict(raw_args)))
        return result

    async def _stream_once(
        self,
        prompt: str,
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterable[str]:
        payload = self._build_payload(prompt, system_prompt, tools, stream=True)
        async for event in self._stream_sse(payload):
            if event == "[DONE]":
                break
            try:
                chunk = event["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMProviderError("OpenAI stream event was malformed.") from exc
            if chunk:
                yield str(chunk)

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
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        return payload

    async def _stream_sse(
        self, payload: dict[str, object]
    ) -> AsyncIterable[dict[str, object] | str]:
        async for line in self._http_stream_lines(self._url(), payload, self._headers()):
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                yield data
            else:
                yield json.loads(data)

    def _url(self) -> str:
        base = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        if not self.config.api_key:
            raise LLMProviderError("OPENAI_API_KEY is required for OpenAIProvider.")
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
