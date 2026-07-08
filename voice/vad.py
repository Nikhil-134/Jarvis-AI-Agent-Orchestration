"""Voice activity detection and utterance endpointing.

Energy-based VAD with an adaptive noise floor, plus an async *endpointer* that
turns a continuous stream of PCM frames into discrete spoken utterances.

Design goals:
    * Zero extra dependencies — only numpy (already required). No webrtcvad /
      silero / torch. Keeps the stack light and ₹0.
    * Deterministic and unit-testable — the endpointer consumes any async
      iterator of frames, so tests feed synthetic audio and real code feeds the
      microphone stream. No hidden global state.
    * Robust silence handling — a short pre-roll buffer avoids clipping the
      first phoneme, and trailing-silence detection ends the utterance
      naturally instead of on a fixed timer.

Audio contract: 16-bit signed mono PCM (little-endian), default 16 kHz, in
fixed-size frames (default 30 ms → 480 samples → 960 bytes).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a hard dep in practice
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

_logger = logging.getLogger(__name__)


class _StreamEnded:
    """Sentinel returned by the endpointer when the frame source is exhausted.

    Distinct from ``None`` (which means 'no speech within the wait window' —
    used for inactivity). A real microphone stream never ends, so this only
    fires for finite sources (tests, files) and lets the loop terminate cleanly.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "STREAM_ENDED"


STREAM_ENDED = _StreamEnded()

# Absolute floor for the speech threshold so a dead-silent mic (noise floor ~0)
# still needs a real signal to trigger. Tuned for int16 RMS.
_MIN_SPEECH_RMS = 320.0
# Speech must exceed noise_floor * this multiplier to count as voiced.
_SPEECH_MULTIPLIER = 3.0
# How quickly the noise floor tracks the ambient level (higher = slower).
_NOISE_ADAPT = 0.96


def frame_rms(frame: bytes) -> float:
    """Return the RMS amplitude of a 16-bit PCM *frame* (0 for empty/odd input)."""
    if not frame or len(frame) < 2:
        return 0.0
    if _HAS_NUMPY:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples * samples)))
    # Pure-python fallback (used only if numpy is somehow absent).
    import array

    samples = array.array("h")
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class EnergyVAD:
    """Frame-level speech/silence classifier with an adaptive noise floor.

    Usage::

        vad = EnergyVAD(sample_rate=16_000, frame_ms=30)
        vad.calibrate(ambient_frames)   # optional
        voiced = vad.is_speech(frame)
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        *,
        min_speech_rms: float = _MIN_SPEECH_RMS,
        speech_multiplier: float = _SPEECH_MULTIPLIER,
        noise_adapt: float = _NOISE_ADAPT,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit
        self._min_speech_rms = min_speech_rms
        self._speech_multiplier = speech_multiplier
        self._noise_adapt = noise_adapt
        self._noise_floor = min_speech_rms / speech_multiplier

    @property
    def threshold(self) -> float:
        """Current speech decision threshold (RMS)."""
        return max(self._min_speech_rms, self._noise_floor * self._speech_multiplier)

    def calibrate(self, frames: list[bytes]) -> None:
        """Seed the noise floor from a batch of presumed-silent ambient frames."""
        values = [frame_rms(f) for f in frames if f]
        if values:
            self._noise_floor = sum(values) / len(values)
            _logger.debug("VAD calibrated: noise_floor=%.1f threshold=%.1f",
                          self._noise_floor, self.threshold)

    def is_speech(self, frame: bytes) -> bool:
        """Classify *frame*; adapt the noise floor slowly on silence."""
        rms = frame_rms(frame)
        voiced = rms >= self.threshold
        if not voiced:
            # Track ambient noise only while silent so it never chases speech up.
            self._noise_floor = (
                self._noise_adapt * self._noise_floor
                + (1.0 - self._noise_adapt) * rms
            )
        return voiced


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """Tuning for utterance endpointing (all times in seconds)."""

    max_initial_wait: float = 8.0      # give up waiting for speech to start
    max_utterance: float = 15.0        # hard cap on a single utterance
    trailing_silence: float = 0.8      # silence that ends an utterance
    onset_frames: int = 2              # consecutive voiced frames to start
    preroll_frames: int = 4            # frames of audio kept before onset


class Endpointer:
    """Collects one complete spoken utterance from a frame stream.

    Reuses a single shared frame iterator across calls so the microphone stays
    open between turns (low latency). Pure async logic — no audio hardware here.
    """

    def __init__(self, vad: EnergyVAD, config: EndpointConfig | None = None) -> None:
        self._vad = vad
        self._cfg = config or EndpointConfig()

    async def collect_utterance(
        self, frames: AsyncIterator[bytes], config: EndpointConfig | None = None,
    ) -> bytes | None | _StreamEnded:
        """Return the next utterance's PCM bytes, ``None``, or ``STREAM_ENDED``.

        * ``bytes``        — a captured utterance.
        * ``None``         — no speech began within ``max_initial_wait``
                             (inactivity; a live mic keeps flowing).
        * ``STREAM_ENDED`` — the frame source was exhausted (finite source only).
        """
        cfg = config or self._cfg
        frame_ms = self._vad.frame_ms
        max_wait_frames = int(self._cfg_seconds_to_frames(cfg.max_initial_wait, frame_ms))
        max_utt_frames = int(self._cfg_seconds_to_frames(cfg.max_utterance, frame_ms))
        trailing_frames = max(1, int(self._cfg_seconds_to_frames(cfg.trailing_silence, frame_ms)))

        preroll: deque[bytes] = deque(maxlen=cfg.preroll_frames)
        collected: list[bytes] = []
        onset_run = 0
        waited = 0
        trailing_silent = 0
        started = False
        exhausted = True  # set False as soon as we break out for a real reason

        try:
            async for frame in frames:
                if not frame:
                    continue

                if not started:
                    preroll.append(frame)
                    if self._vad.is_speech(frame):
                        onset_run += 1
                        if onset_run >= cfg.onset_frames:
                            started = True
                            collected.extend(preroll)  # include pre-roll
                            trailing_silent = 0
                    else:
                        onset_run = 0
                        waited += 1
                        if waited >= max_wait_frames:
                            return None  # no speech began → inactivity
                    continue

                # started: keep collecting until trailing silence or hard cap
                collected.append(frame)
                if self._vad.is_speech(frame):
                    trailing_silent = 0
                else:
                    trailing_silent += 1
                    if trailing_silent >= trailing_frames:
                        exhausted = False
                        break
                if len(collected) >= max_utt_frames:
                    _logger.debug("Utterance hit max duration cap")
                    exhausted = False
                    break
        except StopAsyncIteration:
            pass

        if started and collected:
            return b"".join(collected)
        # Nothing captured. Distinguish stream-exhaustion from a live mid-stream.
        return STREAM_ENDED if exhausted else None

    @staticmethod
    def _cfg_seconds_to_frames(seconds: float, frame_ms: int) -> float:
        return seconds * 1000.0 / frame_ms
