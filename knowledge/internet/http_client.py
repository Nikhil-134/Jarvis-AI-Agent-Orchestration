"""SafeHttpClient — the single, hardened gateway for all outbound requests.

Every byte JARVIS fetches from the public internet goes through here. The client
is deliberately paranoid because it is the project's only egress point:

* **Domain whitelist** — requests to any host not explicitly allowed are
  rejected *before* a socket is opened. There is no way to fetch an arbitrary
  URL, which closes SSRF and data-exfiltration vectors.
* **HTTPS only** — plaintext and non-web schemes (``file:``, ``ftp:`` ...) are
  rejected.
* **No redirect following** — a whitelisted host cannot bounce us to an
  internal address (a classic SSRF pivot). Redirects are treated as failures.
* **Bounded everything** — connect/read timeout, a hard cap on retries with
  backoff, and a maximum response size enforced while streaming so a hostile or
  broken endpoint cannot exhaust memory.
* **JSON only** — the client returns parsed JSON; we never parse HTML, so there
  is no HTML-parser attack surface.

The client owns no global state and is fully injectable (the underlying
transport can be replaced in tests), so nothing here touches the network under
unit test.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx

_logger = logging.getLogger(__name__)

# Only these exact hosts may ever be contacted. Wildcards are matched as
# suffixes on a dot boundary (see _host_allowed), so "wikipedia.org" also
# permits "en.wikipedia.org" but NOT "evilwikipedia.org".
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = (
    "api.duckduckgo.com",
    "wikipedia.org",
    "en.wikipedia.org",
)

_MAX_RESPONSE_BYTES = 512 * 1024          # 512 KiB hard cap
_DEFAULT_TIMEOUT = 6.0                     # seconds, whole request
_DEFAULT_MAX_RETRIES = 2                   # total attempts = 1 + retries
_RETRY_BACKOFF = 0.4                       # seconds, multiplied by attempt


class HttpClientError(Exception):
    """Raised for a disallowed request or a hard failure. Callers fail safe."""


def _host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """True if *host* is exactly an allowed host or a subdomain of one."""
    host = (host or "").lower().strip().rstrip(".")
    if not host:
        return False
    for allow in allowed:
        allow = allow.lower()
        if host == allow or host.endswith("." + allow):
            return True
    return False


class SafeHttpClient:
    """A minimal, whitelist-only async JSON HTTP client.

    Parameters
    ----------
    allowed_hosts:
        Exact hostnames (subdomains permitted) that may be contacted.
    timeout / max_retries / max_response_bytes:
        Conservative bounds; see module docstring.
    transport:
        Optional ``httpx.AsyncBaseTransport`` for tests (dependency injection);
        when provided, no real sockets are opened.
    """

    def __init__(
        self,
        *,
        allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
        user_agent: str = "JARVIS-Local/1.0 (+local-first; contact: local user)",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._allowed_hosts = tuple(allowed_hosts)
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._max_bytes = max_response_bytes
        self._headers = {"User-Agent": user_agent, "Accept": "application/json"}
        self._transport = transport

    def is_allowed(self, url: str) -> bool:
        """Public check: would this URL be permitted? (scheme + host)."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme != "https":
            return False
        return _host_allowed(parsed.hostname or "", self._allowed_hosts)

    async def get_json(
        self, url: str, *, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """GET *url* and return parsed JSON.

        Enforces the whitelist, HTTPS, no-redirects, timeout, retries and the
        response-size cap. Raises :class:`HttpClientError` on any disallowed or
        failed request — providers catch this and return no results (fail safe).
        """
        if not self.is_allowed(url):
            raise HttpClientError(f"Blocked non-whitelisted or non-HTTPS URL: {url!r}")

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._get_once(url, params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Transient/network errors are retryable.
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(_RETRY_BACKOFF * (attempt + 1))
                    continue
            except HttpClientError:
                raise  # policy violations are never retried
        raise HttpClientError(f"Request failed after retries: {url!r} ({last_error})")

    async def _get_once(self, url: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Single attempt: open a short-lived client, stream with a size cap."""
        # follow_redirects=False is critical: a whitelisted host must not be
        # able to redirect us to an internal/again-unchecked address.
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,
            headers=self._headers,
            transport=self._transport,
        ) as client:
            async with client.stream("GET", url, params=params) as response:
                # Any redirect or non-2xx is a hard failure (fail safe).
                if response.is_redirect:
                    raise HttpClientError(f"Redirect refused for {url!r}")
                if response.status_code >= 400:
                    raise HttpClientError(
                        f"HTTP {response.status_code} for {url!r}"
                    )

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise HttpClientError(
                            f"Response exceeded {self._max_bytes} bytes for {url!r}"
                        )
                    chunks.append(chunk)

        body = b"".join(chunks)
        try:
            import json

            return json.loads(body.decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001 - malformed JSON is a failure
            raise HttpClientError(f"Invalid JSON from {url!r}: {exc}") from exc
