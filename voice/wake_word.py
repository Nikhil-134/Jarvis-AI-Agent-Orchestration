"""OpenWakeWord wake-word detection provider for Jarvis.

Uses the openwakeword library for local wake-word/hotword detection.
Falls back gracefully when the library is not installed.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable

from voice.interfaces import IWakeWordDetector

try:
    import openwakeword

    _HAS_OPENWAKEWORD = True
except ImportError:
    openwakeword = None  # type: ignore[assignment]
    _HAS_OPENWAKEWORD = False

_logger = logging.getLogger(__name__)

_DEFAULT_WAKE_WORDS = ["jarvis", "computer"]


class OpenWakeWordDetector(IWakeWordDetector):
    """Wake-word detector using OpenWakeWord."""

    def __init__(self, wake_words: list[str] | None = None) -> None:
        self._wake_words = wake_words or _DEFAULT_WAKE_WORDS
        self._model: openwakeword.Model | None = None  # type: ignore[name-defined]
        if _HAS_OPENWAKEWORD:
            try:
                self._model = openwakeword.Model(wakeword_models=self._wake_words)
            except Exception:
                _logger.exception("Failed to load OpenWakeWord model")

    @property
    def available(self) -> bool:
        """Whether the OpenWakeWord library is installed and loaded."""
        return _HAS_OPENWAKEWORD and self._model is not None

    async def detect(self, audio_stream: AsyncIterable[bytes]) -> AsyncIterable[str]:
        """Yield detected wake-word labels from an audio stream."""
        if not self.available:
            return
        try:
            async for chunk in audio_stream:
                prediction = self._model.predict(chunk)  # type: ignore[union-attr]
                for wake_word in self._wake_words:
                    score = prediction.get(wake_word, 0.0)
                    if score > 0.5:
                        yield wake_word
        except Exception:
            _logger.exception("Wake-word detection failed")
