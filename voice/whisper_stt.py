"""Faster-Whisper speech-to-text provider for Jarvis.

Uses the faster-whisper library for local, efficient transcription.
Falls back gracefully when the library is not installed.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections.abc import AsyncIterable

from voice.interfaces import ISTTProvider

try:
    from faster_whisper import WhisperModel

    _HAS_WHISPER = True
except ImportError:
    WhisperModel = None  # type: ignore[assignment]
    _HAS_WHISPER = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

_logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "base"


class WhisperSTTProvider(ISTTProvider):
    """Speech-to-text provider using faster-whisper."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: WhisperModel | None = None
        if _HAS_WHISPER:
            try:
                self._model = WhisperModel(model_name)
            except Exception:
                _logger.exception("Failed to load faster-whisper model '%s'", model_name)

    @property
    def available(self) -> bool:
        """Whether the underlying library is installed and loaded."""
        return _HAS_WHISPER and self._model is not None and _HAS_NUMPY

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe WAV audio bytes to text using faster-whisper.

        *audio_bytes* must be a complete WAV container (16-bit PCM). The bytes
        are decoded into the float32 mono array that faster-whisper expects —
        the previous implementation passed raw bytes straight to
        ``model.transcribe``, which never worked.

        The blocking model call runs in a worker thread so the event loop is
        not stalled during transcription.
        """
        if not self.available or not audio_bytes:
            return ""
        try:
            samples = self._wav_to_float32(audio_bytes)
            if samples is None or samples.size == 0:
                return ""
            return await asyncio.to_thread(self._transcribe_sync, samples)
        except Exception:
            _logger.exception("Transcription failed")
            return ""

    def _transcribe_sync(self, samples: "np.ndarray") -> str:
        segments, _ = self._model.transcribe(samples)  # type: ignore[union-attr]
        return "".join(segment.text for segment in segments).strip()

    @staticmethod
    def _wav_to_float32(wav_bytes: bytes) -> "np.ndarray | None":
        """Decode a 16-bit PCM WAV into a float32 mono array in [-1, 1]."""
        if not _HAS_NUMPY:
            return None
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                frames = wav_file.readframes(wav_file.getnframes())
        except (wave.Error, EOFError):
            return None
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:  # down-mix to mono
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio

    async def transcribe_stream(self, audio_stream: AsyncIterable[bytes]) -> str:
        """Transcribe a stream of WAV audio chunks to text."""
        if not self.available:
            return ""
        try:
            audio_bytes = b"".join([chunk async for chunk in audio_stream])
            return await self.transcribe(audio_bytes)
        except Exception:
            _logger.exception("Stream transcription failed")
            return ""
