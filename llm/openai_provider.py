"""OpenAI-compatible LLM provider implementation (async, httpx)."""

import json
from collections.abc import AsyncIterable

from llm.base import BaseLLMProvider, LLMConfig
from llm.errors import LLMProviderError
from llm.interfaces import ToolDefinition
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
    ) -> str:
        payload = self._build_payload(prompt, system_prompt, tools, stream=False)
        response = await self._http_post_json(self._url(), payload, self._headers())
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("OpenAI response did not include message content.") from exc

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
