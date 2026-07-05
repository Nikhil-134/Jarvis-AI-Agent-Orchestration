"""Base abstractions for LLM providers."""

import asyncio
import json
from abc import abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from llm.errors import LLMError, LLMProviderError, LLMTimeoutError
from llm.interfaces import ILLMProvider, ToolDefinition

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Runtime settings shared by LLM providers."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25


class BaseLLMProvider(ILLMProvider):
    """Abstract base for LLM providers with retry and HTTP support.

    Subclasses must implement :meth:`_generate_once` and
    :meth:`_stream_once`.  Retry, HTTP client management, and
    common error translation are handled here.
    """

    CAPABILITY_STREAMING = "streaming"
    CAPABILITY_TOOL_CALLING = "tool_calling"
    CAPABILITY_VISION = "vision"
    CAPABILITY_JSON_MODE = "json_mode"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._http_client: httpx.AsyncClient | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def capabilities(self) -> set[str]:
        return set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> str:
        return await self._with_retries(lambda: self._generate_once(prompt, system_prompt, tools))

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterable[str]:
        chunks: list[str] = await self._with_retries(
            lambda: self._collect_stream(prompt, system_prompt, tools)
        )
        for chunk in chunks:
            yield chunk

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # HTTP client (lazy, per-instance)
    # ------------------------------------------------------------------

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
            )
        return self._http_client

    # ------------------------------------------------------------------
    # Shared HTTP helpers with LLM error translation
    # ------------------------------------------------------------------

    async def _http_post_json(
        self, url: str, payload: dict[str, object], headers: dict[str, str]
    ) -> dict[str, object]:
        """POST JSON to *url* and return decoded response.

        Translates httpx exceptions into :class:`LLMError` subclasses
        with the provider name automatically included in messages.
        """
        try:
            response = await self._http.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"{self.name} request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"{self.name} request failed with HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"{self.name} request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMProviderError(f"{self.name} response was not valid JSON.") from exc

    async def _http_stream_lines(
        self, url: str, payload: dict[str, object], headers: dict[str, str]
    ) -> AsyncIterator[str]:
        """Stream raw text lines from a POST request.

        Each yielded line has leading/trailing whitespace stripped.
        Translates httpx exceptions into :class:`LLMError` subclasses.
        """
        try:
            async with self._http.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    yield raw_line.strip()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"{self.name} streaming request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"{self.name} stream failed with HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"{self.name} stream failed: {exc}") from exc

    @staticmethod
    def _build_tools_payload(tools: list[ToolDefinition]) -> list[dict[str, object]]:
        """Convert ToolDefinitions into the OpenAI-compatible tool format.

        Shared by both OpenAI and Ollama providers since Ollama follows
        the same schema.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    @staticmethod
    def _build_messages(
        prompt: str, system_prompt: str | None
    ) -> list[dict[str, str]]:
        """Build the messages array from a prompt and optional system prompt."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    async def _with_retries(self, operation: Callable[[], Any]) -> T:
        """Execute *operation* with exponential backoff retry."""
        last_error: LLMError | None = None
        attempts = self.config.max_retries + 1

        for attempt in range(attempts):
            try:
                result = await operation()
                return result
            except LLMError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(self.config.retry_backoff_seconds * (2**attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM retry loop exited without a result or error.")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def _generate_once(
        self,
        prompt: str,
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> str:
        """Execute one (non-retried) generate call."""

    @abstractmethod
    async def _stream_once(
        self,
        prompt: str,
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterable[str]:
        """Execute one (non-retried) streaming call — yields chunks."""

    async def _collect_stream(
        self,
        prompt: str,
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> list[str]:
        """Eagerly collect a streaming response into a list (for retry)."""
        chunks: list[str] = []
        async for chunk in self._stream_once(prompt, system_prompt, tools):
            chunks.append(chunk)
        return chunks
