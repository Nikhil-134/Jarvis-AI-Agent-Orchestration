"""Ollama LLM provider implementation."""

import json
from collections.abc import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm.base import BaseLLMProvider, LLMConfig
from llm.errors import LLMProviderError, LLMTimeoutError


class OllamaProvider(BaseLLMProvider):
    """LLM provider for a local Ollama chat endpoint."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "ollama"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a complete chat response."""
        return self._with_retries(lambda: self._generate_once(prompt, system_prompt))

    def stream(self, prompt: str, system_prompt: str | None = None) -> Iterable[str]:
        """Stream chat response chunks."""
        return self._with_retries(lambda: tuple(self._stream_once(prompt, system_prompt)))

    def _generate_once(self, prompt: str, system_prompt: str | None) -> str:
        """Send one non-streaming Ollama request."""
        payload = self._payload(prompt, system_prompt, stream=False)
        response = self._post_json(payload)
        try:
            return str(response["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise LLMProviderError("Ollama response did not include message content.") from exc

    def _stream_once(self, prompt: str, system_prompt: str | None) -> Iterable[str]:
        """Send one streaming Ollama request."""
        payload = self._payload(prompt, system_prompt, stream=True)
        for event in self._stream_json_lines(payload):
            try:
                chunk = event.get("message", {}).get("content")
            except AttributeError as exc:
                raise LLMProviderError("Ollama stream event was malformed.") from exc
            if chunk:
                yield str(chunk)
            if event.get("done") is True:
                break

    def _payload(self, prompt: str, system_prompt: str | None, stream: bool) -> dict[str, object]:
        """Build an Ollama chat payload."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {"model": self.config.model, "messages": messages, "stream": stream}

    def _post_json(self, payload: dict[str, object]) -> dict[str, object]:
        """POST JSON to Ollama and return decoded JSON."""
        request = self._request(payload)
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise LLMTimeoutError("Ollama request timed out.") from exc
        except HTTPError as exc:
            raise LLMProviderError(f"Ollama request failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            raise LLMProviderError(f"Ollama request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError("Ollama response was not valid JSON.") from exc

    def _stream_json_lines(self, payload: dict[str, object]) -> Iterable[dict[str, object]]:
        """Yield decoded JSON lines from Ollama."""
        request = self._request(payload)
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
        except TimeoutError as exc:
            raise LLMTimeoutError("Ollama streaming request timed out.") from exc
        except HTTPError as exc:
            raise LLMProviderError(f"Ollama stream failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            raise LLMProviderError(f"Ollama stream failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError("Ollama stream event was not valid JSON.") from exc

    def _request(self, payload: dict[str, object]) -> Request:
        """Build an Ollama request."""
        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        return Request(
            url=f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )


if __name__ == "__main__":
    provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3.1"))
    print(provider.name)
