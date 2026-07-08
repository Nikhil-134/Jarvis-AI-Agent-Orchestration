"""Local audio I/O — microphone capture and speaker playback.

Thin, dependency-guarded wrapper around ``sounddevice``. When the library or
audio hardware is unavailable, every method degrades to a no-op / empty result
instead of raising, so headless environments and CI never crash.

All audio is 16-bit mono PCM, which matches both Whisper (expects 16 kHz) and
Piper (emits 22.05 kHz WAV). Resampling is intentionally out of scope here;
capture and playback each use their own native rate.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections.abc import AsyncIterator

try:
    import sounddevice as sd
    import numpy as np

    _HAS_AUDIO = True
except Exception:  # ImportError, or OSError when PortAudio is missing
    sd = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _HAS_AUDIO = False

_logger = logging.getLogger(__name__)


class AudioIO:
    """Microphone capture and WAV playback via sounddevice.

    Usage::

        audio = AudioIO()
        pcm = audio.record(seconds=4)      # capture 16 kHz mono PCM bytes
        audio.play_wav(wav_bytes)          # play a WAV byte string
    """

    def __init__(self, capture_rate: int = 16_000) -> None:
        self._capture_rate = capture_rate

    @property
    def available(self) -> bool:
        """Whether audio hardware and the sounddevice library are usable."""
        if not _HAS_AUDIO:
            return False
        try:
            sd.check_input_settings(samplerate=self._capture_rate, channels=1)
            return True
        except Exception:
            return False

    @property
    def capture_rate(self) -> int:
        return self._capture_rate

    def record(self, seconds: float) -> bytes:
        """Record *seconds* of 16-bit mono PCM from the default microphone.

        Returns raw PCM bytes (no WAV header), or b'' if audio is unavailable.
        """
        if not _HAS_AUDIO:
            return b""
        try:
            frames = int(seconds * self._capture_rate)
            recording = sd.rec(
                frames, samplerate=self._capture_rate, channels=1, dtype="int16",
            )
            sd.wait()
            return recording.tobytes()
        except Exception:
            _logger.exception("Microphone capture failed")
            return b""

    async def stream_frames(
        self, frame_ms: int = 30, stop: asyncio.Event | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield fixed-size 16-bit mono PCM frames from the microphone.

        The stream stays open for the lifetime of the iteration (mic is not
        reopened per utterance → low latency between turns). Audio arrives on a
        PortAudio callback thread and is handed to the event loop via a queue.

        Stops when *stop* is set (if provided) or the consumer stops iterating.
        Yields nothing if audio hardware is unavailable.
        """
        if not _HAS_AUDIO:
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        blocksize = int(self._capture_rate * frame_ms / 1000)

        def _callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
            if status:
                _logger.debug("Audio input status: %s", status)
            # bytes(indata) copies the buffer; safe to hand across threads.
            try:
                loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))
            except asyncio.QueueFull:  # pragma: no cover - drop on overflow
                pass

        try:
            stream = sd.RawInputStream(
                samplerate=self._capture_rate,
                blocksize=blocksize,
                dtype="int16",
                channels=1,
                callback=_callback,
            )
        except Exception:
            _logger.exception("Failed to open microphone stream")
            return

        with stream:
            while stop is None or not stop.is_set():
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue  # re-check stop flag
                yield frame

    def pcm_to_wav(self, pcm: bytes, sample_rate: int | None = None) -> bytes:
        """Wrap raw 16-bit mono PCM in a WAV container."""
        rate = sample_rate or self._capture_rate
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()

    def play_wav(self, wav_bytes: bytes) -> None:
        """Play *wav_bytes* (a complete WAV) through the default speaker."""
        if not _HAS_AUDIO or not wav_bytes:
            return
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)
            sd.play(data, samplerate=rate)
            sd.wait()
        except Exception:
            _logger.exception("Audio playback failed")
