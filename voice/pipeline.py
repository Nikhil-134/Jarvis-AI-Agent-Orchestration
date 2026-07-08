"""Voice pipeline — ties local audio, STT, the JARVIS brain, and TTS together.

Flow per interaction::

    [wake word] -> record -> STT -> Runtime.run(text) -> TTS -> playback

The pipeline is deliberately provider-agnostic: it depends only on the
``ISTTProvider`` / ``ITTSProvider`` interfaces and an object exposing an async
``run(str) -> str`` method (the existing :class:`~runtime.runtime.Runtime`).
That keeps the voice layer decoupled from the brain and trivially testable
with fakes.

Everything degrades gracefully. If the microphone, STT, or TTS is unavailable,
the affected step is skipped and the pipeline reports why instead of crashing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from voice.audio_io import AudioIO
from voice.interfaces import ISTTProvider, ITTSProvider

_logger = logging.getLogger(__name__)


class _Brain(Protocol):
    """Anything that turns user text into a response string."""

    async def run(self, user_input: str) -> str: ...


@dataclass(frozen=True, slots=True)
class VoiceTurn:
    """The outcome of one voice interaction, for logging/inspection."""

    transcript: str
    response: str
    spoke: bool


class VoicePipeline:
    """Local voice loop around a text brain.

    Usage::

        pipeline = VoicePipeline(brain=runtime, stt=whisper, tts=piper)
        turn = await pipeline.listen_and_respond(record_seconds=4)
        print(turn.transcript, "->", turn.response)
    """

    def __init__(
        self,
        brain: _Brain,
        stt: ISTTProvider,
        tts: ITTSProvider,
        audio: AudioIO | None = None,
        *,
        record_seconds: float = 5.0,
    ) -> None:
        self._brain = brain
        self._stt = stt
        self._tts = tts
        self._audio = audio or AudioIO()
        self._record_seconds = record_seconds

    @property
    def can_listen(self) -> bool:
        """Whether microphone capture and STT are both usable."""
        return self._audio.available and getattr(self._stt, "available", False)

    @property
    def can_speak(self) -> bool:
        """Whether TTS and audio playback are both usable."""
        return self._audio.available and getattr(self._tts, "available", False)

    async def transcribe_once(self, seconds: float | None = None) -> str:
        """Record from the mic and return the transcribed text ('' on failure)."""
        if not self.can_listen:
            return ""
        pcm = self._audio.record(seconds or self._record_seconds)
        if not pcm:
            return ""
        wav = self._audio.pcm_to_wav(pcm)
        return await self._stt.transcribe(wav)

    async def speak(self, text: str) -> bool:
        """Synthesise *text* and play it. Returns True if audio was played."""
        if not text or not self.can_speak:
            return False
        wav = await self._tts.speak(text)
        if not wav:
            return False
        self._audio.play_wav(wav)
        return True

    async def respond_to_text(self, text: str) -> VoiceTurn:
        """Run *text* through the brain and speak the reply.

        Useful when the caller already has a transcript (e.g. from a test or a
        push-to-talk UI) and only wants brain + TTS.
        """
        clean = (text or "").strip()
        if not clean:
            return VoiceTurn(transcript="", response="", spoke=False)
        response = await self._brain.run(clean)
        spoke = await self.speak(response)
        return VoiceTurn(transcript=clean, response=response, spoke=spoke)

    async def listen_and_respond(self, record_seconds: float | None = None) -> VoiceTurn:
        """One full turn: record -> STT -> brain -> TTS -> playback."""
        transcript = await self.transcribe_once(record_seconds)
        if not transcript:
            return VoiceTurn(transcript="", response="", spoke=False)
        return await self.respond_to_text(transcript)
