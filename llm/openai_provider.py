"""OpenAI LLM provider implementation."""

import json
from collections.abc import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm.base import BaseLLMProvider, LLMConfig
from llm.errors import LLMProviderError, LLMTimeoutError


class OpenAIProvider(BaseLLMProvider):
    """LLM provider for OpenAI-compatible chat completions."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "openai"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a complete chat response."""
        return self._with_retries(lambda: self._generate_once(prompt, system_prompt))

    def stream(self, prompt: str, system_prompt: str | None = None) -> Iterable[str]:
        """Stream chat response chunks."""
        return self._with_retries(lambda: tuple(self._stream_once(prompt, system_prompt)))

    def _generate_once(self, prompt: str, system_prompt: str | None) -> str:
        """Send one non-streaming OpenAI request."""
        payload = self._payload(prompt, system_prompt, stream=False)
        response = self._post_json(payload)
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("OpenAI response did not include message content.") from exc

    def _stream_once(self, prompt: str, system_prompt: str | None) -> Iterable[str]:
        """Send one streaming OpenAI request."""
        payload = self._payload(prompt, system_prompt, stream=True)
        for event in self._stream_json_events(payload):
            if event == "[DONE]":
                break
            try:
                chunk = event["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMProviderError("OpenAI stream event was malformed.") from exc
            if chunk:
                yield str(chunk)

    def _payload(self, prompt: str, system_prompt: str | None, stream: bool) -> dict[str, object]:
        """Build an OpenAI chat completions payload."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {"model": self.config.model, "messages": messages, "stream": stream}

    def _post_json(self, payload: dict[str, object]) -> dict[str, object]:
        """POST JSON to OpenAI and return decoded JSON."""
        request = self._request(payload)
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise LLMTimeoutError("OpenAI request timed out.") from exc
        except HTTPError as exc:
            raise LLMProviderError(f"OpenAI request failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError("OpenAI response was not valid JSON.") from exc

    def _stream_json_events(self, payload: dict[str, object]) -> Iterable[dict[str, object] | str]:
        """Yield decoded Server-Sent Events from OpenAI."""
        request = self._request(payload)
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        yield data
                    else:
                        yield json.loads(data)
        except TimeoutError as exc:
            raise LLMTimeoutError("OpenAI streaming request timed out.") from exc
        except HTTPError as exc:
            raise LLMProviderError(f"OpenAI stream failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            raise LLMProviderError(f"OpenAI stream failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError("OpenAI stream event was not valid JSON.") from exc

    def _request(self, payload: dict[str, object]) -> Request:
        """Build an authenticated OpenAI request."""
        if not self.config.api_key:
            raise LLMProviderError("OPENAI_API_KEY is required for OpenAIProvider.")

        base_url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        return Request(
            url=f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )


if __name__ == "__main__":
    provider = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4.1-mini"))
    print(provider.name)
