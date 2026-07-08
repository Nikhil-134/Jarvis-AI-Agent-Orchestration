"""Continuous conversation loop — the production hands-free voice runtime.

State machine::

        ┌────────────────────────── stop() / exit phrase ─────────────────────┐
        │                                                                     ▼
   ┌─────────┐  wake word / AlwaysAwake   ┌────────┐  exit phrase / shutdown  (end)
   │ ASLEEP  │ ─────────────────────────► │ ACTIVE │
   └─────────┘                            └────────┘
        ▲   inactivity timeout / "go to sleep"  │
        └───────────────────────────────────────┘

Per active turn::

    stream mic ─► VAD endpointer ─► STT ─► [wake gate] ─► brain.run(text)
        ─► TTS ─► playback ─► back to listening (no restart, mic stays open)

The loop:
    * keeps the microphone open for the whole session (low turn-to-turn latency);
    * ends an utterance on trailing silence, not a fixed timer;
    * recognises exit phrases ("goodbye", "shutdown", ...) and sleep phrases
      ("go to sleep", "stop listening");
    * returns to wake-gating after an inactivity timeout;
    * degrades gracefully and never crashes the process on an audio error;
    * is fully injectable (audio source, STT, TTS, brain, wake strategy) so it
      is unit-testable without a microphone.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from voice.interfaces import ISTTProvider, ITTSProvider
from voice.vad import STREAM_ENDED, EndpointConfig, Endpointer, EnergyVAD
from voice.wake import WakeStrategy, build_wake_strategy

_logger = logging.getLogger(__name__)


# Phrases that end the session entirely.
_EXIT_PHRASES: tuple[str, ...] = (
    "goodbye", "good bye", "shutdown", "shut down", "exit",
    "quit", "power off", "turn off", "that is all", "that's all",
)
# Phrases that send JARVIS back to sleep (re-arm wake word) without exiting.
_SLEEP_PHRASES: tuple[str, ...] = (
    "go to sleep", "stop listening", "never mind", "nevermind", "stand down",
)


class LoopState(Enum):
    ASLEEP = "asleep"
    ACTIVE = "active"
    STOPPED = "stopped"


class _Brain(Protocol):
    async def run(self, user_input: str) -> str: ...


class _AudioSink(Protocol):
    """Playback + PCM→WAV helper (satisfied by :class:`voice.audio_io.AudioIO`)."""

    def play_wav(self, wav_bytes: bytes) -> None: ...
    def pcm_to_wav(self, pcm: bytes, sample_rate: int | None = ...) -> bytes: ...


# A frame source is any callable returning an async iterator of PCM frames.
FrameSource = Callable[[], AsyncIterator[bytes]]


@dataclass
class LoopConfig:
    """Runtime tuning for the continuous loop."""

    wake_mode: str = "transcript"           # "transcript" | "none"
    wake_words: tuple[str, ...] = ("jarvis", "computer")
    frame_ms: int = 30
    inactivity_timeout: float = 30.0        # active→asleep after this much quiet
    greeting: str = "Yes?"                  # spoken on wake (transcript mode)
    farewell: str = "Goodbye."              # spoken on exit
    sleep_ack: str = "Going to sleep."      # spoken when re-arming wake word
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)


@dataclass
class LoopStats:
    """Observable counters for tests and diagnostics."""

    turns: int = 0
    wakes: int = 0
    transcripts: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    stopped_reason: str = ""


class ContinuousVoiceLoop:
    """Hands-free, always-on conversational loop around a text brain."""

    def __init__(
        self,
        brain: _Brain,
        stt: ISTTProvider,
        tts: ITTSProvider,
        audio: _AudioSink,
        frame_source: FrameSource,
        *,
        config: LoopConfig | None = None,
        capture_rate: int = 16_000,
        on_event: Callable[[str, str], None] | None = None,
    ) -> None:
        self._brain = brain
        self._stt = stt
        self._tts = tts
        self._audio = audio
        self._frame_source = frame_source
        self._cfg = config or LoopConfig()
        self._capture_rate = capture_rate
        self._on_event = on_event or (lambda kind, text: None)

        self._vad = EnergyVAD(sample_rate=capture_rate, frame_ms=self._cfg.frame_ms)
        self._endpointer = Endpointer(self._vad, self._cfg.endpoint)
        self._wake: WakeStrategy = build_wake_strategy(self._cfg.wake_mode, self._cfg.wake_words)

        self._state = LoopState.ASLEEP if self._cfg.wake_mode != "none" else LoopState.ACTIVE
        self._stop = asyncio.Event()
        self.stats = LoopStats()

    @property
    def state(self) -> LoopState:
        return self._state

    def stop(self) -> None:
        """Request a graceful shutdown of the loop."""
        self._stop.set()

    async def run(self) -> LoopStats:
        """Run the conversation loop until stopped, an exit phrase, or Ctrl+C."""
        frames = self._frame_source()
        if self._state == LoopState.ACTIVE:
            self._emit("state", "active")
        else:
            self._emit("state", "asleep")

        try:
            while not self._stop.is_set():
                utterance = await self._endpointer.collect_utterance(
                    frames, self._endpoint_for_state(),
                )

                if utterance is STREAM_ENDED:
                    self.stats.stopped_reason = self.stats.stopped_reason or "stream_ended"
                    break

                if utterance is None:
                    # No speech within the window.
                    if self._state == LoopState.ACTIVE and self._cfg.wake_mode != "none":
                        await self._go_to_sleep("inactivity")
                    continue

                transcript = (await self._safe_transcribe(utterance)).strip()
                if not transcript:
                    continue
                self._emit("heard", transcript)

                if not await self._handle_transcript(transcript):
                    break  # exit phrase → end session
        except (asyncio.CancelledError, KeyboardInterrupt):  # pragma: no cover
            self.stats.stopped_reason = self.stats.stopped_reason or "interrupted"
            _logger.info("Voice loop interrupted")
        finally:
            await self._aclose(frames)
            self._state = LoopState.STOPPED

        return self.stats

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    async def _handle_transcript(self, transcript: str) -> bool:
        """Process one transcript. Returns False if the session should end."""
        if self._state == LoopState.ASLEEP:
            result = self._wake.check(transcript)
            if not result.activated:
                return True  # ignore chatter until woken
            self.stats.wakes += 1
            self._state = LoopState.ACTIVE
            self._emit("state", "active")
            if result.command:
                # "Jarvis, what's the time?" → wake + command in one utterance.
                return await self._respond(result.command)
            await self._say(self._cfg.greeting)
            return True

        # ACTIVE
        if self._is_exit_phrase(transcript):
            await self._say(self._cfg.farewell)
            self.stats.stopped_reason = "exit_phrase"
            return False
        if self._is_sleep_phrase(transcript):
            await self._go_to_sleep("sleep_phrase")
            return True

        # In "none" wake mode a bare wake word may still prefix a command.
        command = transcript
        if self._cfg.wake_mode != "none":
            woke = self._wake.check(transcript)
            if woke.activated and woke.command:
                command = woke.command
        return await self._respond(command)

    async def _respond(self, command: str) -> bool:
        """Run *command* through the brain and speak the reply."""
        command = command.strip()
        if not command:
            return True
        self.stats.turns += 1
        self.stats.transcripts.append(command)
        try:
            response = await self._brain.run(command)
        except Exception:
            _logger.exception("Brain failed on voice command")
            response = "Sorry, something went wrong handling that."
        response = (response or "").strip()
        self.stats.responses.append(response)
        self._emit("response", response)
        await self._say(response)
        return True

    async def _go_to_sleep(self, reason: str) -> None:
        self._state = LoopState.ASLEEP
        self._emit("state", "asleep")
        _logger.info("Voice loop returning to sleep (%s)", reason)
        await self._say(self._cfg.sleep_ack)

    # ------------------------------------------------------------------
    # Speech in/out
    # ------------------------------------------------------------------

    async def _safe_transcribe(self, pcm: bytes) -> str:
        try:
            wav = self._audio.pcm_to_wav(pcm, self._capture_rate)
            return await self._stt.transcribe(wav)
        except Exception:
            _logger.exception("Transcription failed in voice loop")
            return ""

    async def _say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        try:
            wav = await self._tts.speak(text)
            if wav:
                # Playback is blocking (sounddevice); run off the event loop.
                await asyncio.to_thread(self._audio.play_wav, wav)
        except Exception:
            _logger.exception("Speech playback failed in voice loop")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _endpoint_for_state(self) -> EndpointConfig:
        """Use the inactivity timeout as the initial-wait while ACTIVE."""
        base = self._cfg.endpoint
        if self._state == LoopState.ACTIVE and self._cfg.wake_mode != "none":
            return EndpointConfig(
                max_initial_wait=self._cfg.inactivity_timeout,
                max_utterance=base.max_utterance,
                trailing_silence=base.trailing_silence,
                onset_frames=base.onset_frames,
                preroll_frames=base.preroll_frames,
            )
        return base

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.lower().strip(" .,!?").split())

    def _is_exit_phrase(self, text: str) -> bool:
        norm = self._normalise(text)
        return any(norm == p or norm.endswith(" " + p) for p in _EXIT_PHRASES)

    def _is_sleep_phrase(self, text: str) -> bool:
        norm = self._normalise(text)
        return any(p in norm for p in _SLEEP_PHRASES)

    def _emit(self, kind: str, text: str) -> None:
        try:
            self._on_event(kind, text)
        except Exception:  # pragma: no cover - never let a hook break the loop
            _logger.debug("on_event hook raised", exc_info=True)

    @staticmethod
    async def _aclose(frames: AsyncIterator[bytes]) -> None:
        aclose = getattr(frames, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # pragma: no cover
                _logger.debug("Frame source aclose raised", exc_info=True)
