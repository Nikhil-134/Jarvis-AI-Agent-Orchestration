"""Tests for the continuous voice loop: VAD, endpointing, wake, and the loop
state machine — all with in-memory fakes (no microphone, no models required).

The last test is an OPTIONAL real-audio integration check: if Piper + Whisper
are installed it synthesises speech, feeds it through the VAD endpointer, and
transcribes it back — proving the endpointer works on real speech without a mic.
"""

from __future__ import annotations

import wave
import io

import numpy as np
import pytest

from voice.vad import EnergyVAD, Endpointer, EndpointConfig, frame_rms
from voice.wake import TranscriptWakeWord, AlwaysAwake, build_wake_strategy
from voice.continuous_loop import ContinuousVoiceLoop, LoopConfig, LoopState, LoopStats


SR, FMS = 16_000, 30
FRAME = int(SR * FMS / 1000)


def _silence() -> bytes:
    return np.zeros(FRAME, dtype=np.int16).tobytes()


def _tone(amp: int = 8000) -> bytes:
    return (np.sin(np.arange(FRAME) / 3.0) * amp).astype(np.int16).tobytes()


async def _frames(seq: list[bytes]):
    for f in seq:
        yield f


# --------------------------------------------------------------------------
# VAD
# --------------------------------------------------------------------------


def test_frame_rms_silence_vs_tone():
    assert frame_rms(_silence()) == 0.0
    assert frame_rms(_tone()) > 1000


def test_vad_classifies_speech_and_silence():
    vad = EnergyVAD(SR, FMS)
    assert vad.is_speech(_silence()) is False
    assert vad.is_speech(_tone()) is True


def test_vad_calibration_raises_threshold_on_noisy_room():
    quiet = EnergyVAD(SR, FMS)
    noisy = EnergyVAD(SR, FMS)
    noisy.calibrate([_tone(amp=400) for _ in range(10)])  # loud ambient
    assert noisy.threshold >= quiet.threshold


# --------------------------------------------------------------------------
# Endpointer
# --------------------------------------------------------------------------


async def test_endpointer_captures_speech_between_silence():
    ep = Endpointer(EnergyVAD(SR, FMS), EndpointConfig(trailing_silence=0.3, max_initial_wait=2))
    seq = [_silence()] * 5 + [_tone()] * 20 + [_silence()] * 40
    utt = await ep.collect_utterance(_frames(seq))
    assert utt is not None
    # ~20 tone frames + preroll, well under the full sequence
    assert 0.4 < len(utt) / 2 / SR < 1.2


async def test_endpointer_returns_none_on_silence_only():
    ep = Endpointer(EnergyVAD(SR, FMS))
    seq = [_silence()] * 30
    utt = await ep.collect_utterance(_frames(seq), EndpointConfig(max_initial_wait=0.2))
    assert utt is None


async def test_endpointer_respects_max_utterance_cap():
    ep = Endpointer(EnergyVAD(SR, FMS))
    seq = [_tone()] * 200  # continuous speech, never stops
    utt = await ep.collect_utterance(_frames(seq), EndpointConfig(max_utterance=0.5, max_initial_wait=1))
    assert utt is not None
    assert len(utt) / 2 / SR <= 0.6  # capped near 0.5s


# --------------------------------------------------------------------------
# Wake strategies
# --------------------------------------------------------------------------


def test_transcript_wake_extracts_command():
    w = TranscriptWakeWord(["jarvis"])
    r = w.check("jarvis what is the time")
    assert r.activated and r.command == "what is the time"


def test_transcript_wake_ignores_non_wake():
    assert TranscriptWakeWord(["jarvis"]).check("hello there").activated is False


def test_always_awake_passes_everything():
    r = AlwaysAwake().check("do the thing")
    assert r.activated and r.command == "do the thing"


def test_wake_factory_none_is_always_awake():
    assert isinstance(build_wake_strategy("none"), AlwaysAwake)


# --------------------------------------------------------------------------
# ContinuousVoiceLoop — fakes (drives whole state machine without audio)
# --------------------------------------------------------------------------


class _FakeBrain:
    def __init__(self):
        self.seen = []

    async def run(self, text: str) -> str:
        self.seen.append(text)
        return f"echo:{text}"


class _ScriptedSTT:
    """Returns a queued transcript for each utterance handed to it."""

    def __init__(self, transcripts: list[str]):
        self._q = list(transcripts)
        self.available = True

    async def transcribe(self, wav: bytes) -> str:
        return self._q.pop(0) if self._q else ""

    async def transcribe_stream(self, s):  # pragma: no cover
        return ""


class _FakeTTS:
    def __init__(self):
        self.available = True
        self.spoken = []

    async def speak(self, text: str) -> bytes:
        self.spoken.append(text)
        return b"RIFFfake"

    async def speak_stream(self, text):  # pragma: no cover
        yield b"RIFFfake"


class _FakeAudio:
    def __init__(self):
        self.played = []

    def play_wav(self, wav: bytes) -> None:
        self.played.append(wav)

    def pcm_to_wav(self, pcm: bytes, sample_rate=None) -> bytes:
        return b"RIFF" + pcm


def _one_utterance_source():
    """A frame source that emits exactly one utterance then stops.

    The loop calls the source once and iterates; each collect_utterance pulls
    from the same iterator. We emit N utterances of tone separated by silence,
    then end the stream so the loop terminates.
    """
    def factory():
        async def gen():
            # utterance 1
            for _ in range(3):
                yield _silence()
            for _ in range(15):
                yield _tone()
            for _ in range(20):
                yield _silence()
            # utterance 2
            for _ in range(15):
                yield _tone()
            for _ in range(20):
                yield _silence()
        return gen()
    return factory


async def _run_loop(stt_transcripts, wake_mode="none", **cfg_kw):
    brain, tts, audio = _FakeBrain(), _FakeTTS(), _FakeAudio()
    cfg = LoopConfig(wake_mode=wake_mode, frame_ms=FMS,
                     endpoint=EndpointConfig(trailing_silence=0.2, max_initial_wait=1.0),
                     **cfg_kw)
    loop = ContinuousVoiceLoop(
        brain=brain, stt=_ScriptedSTT(stt_transcripts), tts=tts, audio=audio,
        frame_source=_one_utterance_source(), config=cfg, capture_rate=SR,
    )
    stats = await loop.run()
    return loop, brain, tts, stats


async def test_loop_always_awake_processes_multiple_turns():
    loop, brain, tts, stats = await _run_loop(
        ["what is two plus two", "what is the capital of france"], wake_mode="none",
    )
    assert brain.seen == ["what is two plus two", "what is the capital of france"]
    assert stats.turns == 2
    assert "echo:what is two plus two" in tts.spoken


async def test_loop_exit_phrase_ends_session():
    loop, brain, tts, stats = await _run_loop(
        ["tell me a joke", "goodbye"], wake_mode="none",
    )
    assert brain.seen == ["tell me a joke"]  # goodbye is not sent to brain
    assert stats.stopped_reason == "exit_phrase"
    assert loop.state == LoopState.STOPPED


async def test_loop_wake_gating_ignores_until_woken():
    # First utterance lacks the wake word → ignored; second wakes + commands.
    loop, brain, tts, stats = await _run_loop(
        ["random chatter", "jarvis what time is it"],
        wake_mode="transcript",
    )
    loop_cfg_wake = TranscriptWakeWord(["jarvis"])
    assert brain.seen == ["what time is it"]
    assert stats.wakes == 1


# --------------------------------------------------------------------------
# OPTIONAL real-audio integration: Piper speech → endpointer → Whisper
# --------------------------------------------------------------------------


def _wav_to_frames(wav_bytes: bytes, target_sr: int = SR, frame_ms: int = FMS) -> list[bytes]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if sr != target_sr:  # simple linear resample to 16k
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * target_sr / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio)
    a16 = audio.astype(np.int16).tobytes()
    n = FRAME * 2
    return [a16[i:i + n] for i in range(0, len(a16), n) if len(a16[i:i + n]) == n]


@pytest.mark.integration
async def test_real_speech_endpoint_and_transcribe():
    """Piper-synthesised speech survives VAD endpointing and Whisper STT."""
    from pathlib import Path
    from voice.piper_tts import PiperTTSProvider
    from voice.whisper_stt import WhisperSTTProvider

    model = Path("voice_models/en_US-lessac-medium.onnx")
    if not model.is_file():
        pytest.skip("Piper voice model not present")

    tts = PiperTTSProvider(model)
    stt = WhisperSTTProvider("base")
    if not (tts.available and stt.available):
        pytest.skip("Piper/Whisper not available")

    phrase = "what is the capital of france"
    wav = await tts.speak(phrase)
    assert wav[:4] == b"RIFF"

    frames = [_silence()] * 5 + _wav_to_frames(wav) + [_silence()] * 30
    ep = Endpointer(EnergyVAD(SR, FMS), EndpointConfig(trailing_silence=0.5, max_initial_wait=3))
    utt = await ep.collect_utterance(_frames(frames))
    assert utt is not None

    audio_wav = _pcm_wav(utt)
    heard = (await stt.transcribe(audio_wav)).lower()
    assert "capital" in heard and "france" in heard


def _pcm_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()
