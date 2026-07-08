"""Edge-TTS text-to-speech provider for Jarvis.

Uses the edge-tts library (Microsoft Edge TTS) for voice synthesis.
Falls back gracefully when the library is not installed.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable

from voice.interfaces import ITTSProvider

try:
    import edge_tts

    _HAS_EDGE_TTS = True
except ImportError:
    edge_tts = None  # type: ignore[assignment]
    _HAS_EDGE_TTS = False

_logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "en-US-JennyNeural"


class EdgeTTSProvider(ITTSProvider):
    """Text-to-speech provider using edge-tts."""

    def __init__(self, voice: str = _DEFAULT_VOICE) -> None:
        self._voice = voice

    @property
    def available(self) -> bool:
        """Whether the edge-tts library is installed."""
        return _HAS_EDGE_TTS

    async def speak(self, text: str) -> bytes:
        """Synthesise text to MP3 audio bytes using edge-tts."""
        if not self.available:
            return b""
        try:
            communicate = edge_tts.Communicate(text, self._voice)
            audio = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
            return audio
        except Exception:
            _logger.exception("TTS speak failed")
            return b""

    async def speak_stream(self, text: str) -> AsyncIterable[bytes]:
        """Synthesise text to a stream of audio chunks using edge-tts."""
        if not self.available:
            return
        try:
            communicate = edge_tts.Communicate(text, self._voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception:
            _logger.exception("TTS speak_stream failed")
