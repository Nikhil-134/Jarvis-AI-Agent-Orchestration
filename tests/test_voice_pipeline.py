"""Tests for the local voice pipeline, VoiceAgent, and Piper provider wiring.

These use in-memory fakes so they run with no audio hardware, no models, and
no native dependencies — verifying wiring, graceful degradation, and that the
local Piper provider is the default (edge-tts is never auto-selected).
"""

from __future__ import annotations

import pytest

from agents.contracts import AgentTask
from agents.voice_agent import VoiceAgent
from voice.pipeline import VoicePipeline, VoiceTurn


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeBrain:
    def __init__(self, reply: str = "The answer is 42.") -> None:
        self._reply = reply
        self.seen: list[str] = []

    async def run(self, user_input: str) -> str:
        self.seen.append(user_input)
        return self._reply


class _FakeTTS:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.spoken: list[str] = []

    async def speak(self, text: str) -> bytes:
        self.spoken.append(text)
        return b"RIFF....WAVEfake" if text and self.available else b""

    async def speak_stream(self, text: str):
        yield await self.speak(text)


class _FakeSTT:
    def __init__(self, transcript: str = "hello jarvis", available: bool = True) -> None:
        self.available = available
        self._transcript = transcript

    async def transcribe(self, audio_bytes: bytes) -> str:
        return self._transcript if self.available else ""

    async def transcribe_stream(self, audio_stream) -> str:
        return self._transcript if self.available else ""


class _FakeAudio:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.played: list[bytes] = []
        self._rate = 16_000

    def record(self, seconds: float) -> bytes:
        return b"\x00\x00" * 100 if self.available else b""

    def pcm_to_wav(self, pcm: bytes, sample_rate: int | None = None) -> bytes:
        return b"RIFF" + pcm

    def play_wav(self, wav_bytes: bytes) -> None:
        self.played.append(wav_bytes)


# --------------------------------------------------------------------------
# VoicePipeline
# --------------------------------------------------------------------------


async def test_full_turn_records_thinks_and_speaks():
    brain = _FakeBrain("Paris.")
    audio = _FakeAudio()
    pipeline = VoicePipeline(brain=brain, stt=_FakeSTT("capital of france"), tts=_FakeTTS(), audio=audio)

    turn = await pipeline.listen_and_respond()

    assert isinstance(turn, VoiceTurn)
    assert turn.transcript == "capital of france"
    assert turn.response == "Paris."
    assert turn.spoke is True
    assert brain.seen == ["capital of france"]
    assert audio.played  # something was played


async def test_respond_to_text_skips_stt():
    brain = _FakeBrain("hi there")
    tts = _FakeTTS()
    pipeline = VoicePipeline(brain=brain, stt=_FakeSTT(available=False), tts=tts, audio=_FakeAudio())

    turn = await pipeline.respond_to_text("hello")

    assert turn.response == "hi there"
    assert tts.spoken == ["hi there"]


async def test_no_mic_yields_empty_turn():
    pipeline = VoicePipeline(
        brain=_FakeBrain(), stt=_FakeSTT(), tts=_FakeTTS(), audio=_FakeAudio(available=False),
    )
    assert pipeline.can_listen is False
    turn = await pipeline.listen_and_respond()
    assert turn.transcript == ""
    assert turn.spoke is False


async def test_can_speak_requires_tts_and_audio():
    assert VoicePipeline(_FakeBrain(), _FakeSTT(), _FakeTTS(available=False), _FakeAudio()).can_speak is False
    assert VoicePipeline(_FakeBrain(), _FakeSTT(), _FakeTTS(), _FakeAudio(available=False)).can_speak is False
    assert VoicePipeline(_FakeBrain(), _FakeSTT(), _FakeTTS(), _FakeAudio()).can_speak is True


async def test_speak_returns_false_when_tts_produces_no_audio():
    pipeline = VoicePipeline(_FakeBrain(), _FakeSTT(), _FakeTTS(), _FakeAudio())
    assert await pipeline.speak("") is False


# --------------------------------------------------------------------------
# VoiceAgent
# --------------------------------------------------------------------------


async def test_voice_agent_output_synthesises():
    agent = VoiceAgent(tts=_FakeTTS(), stt=None, audio=_FakeAudio())
    result = await agent.handle(AgentTask(task_type="voice.output", payload={"text": "hello"}))
    assert result.success is True
    assert result.data["audio_bytes"] > 0
    assert result.data["played"] is True


async def test_voice_agent_output_unavailable():
    agent = VoiceAgent(tts=None, stt=None, audio=None)
    result = await agent.handle(AgentTask(task_type="voice.output", payload={"text": "hi"}))
    assert result.success is False
    assert result.data["status"] == "unavailable"


async def test_voice_agent_input_transcribes_supplied_audio():
    agent = VoiceAgent(tts=None, stt=_FakeSTT("transcribed text"), audio=None)
    result = await agent.handle(
        AgentTask(task_type="voice.input", payload={"audio": b"RIFF....WAVE"})
    )
    assert result.success is True
    assert result.data["transcript"] == "transcribed text"


async def test_voice_agent_rejects_unknown_task():
    agent = VoiceAgent()
    result = await agent.handle(AgentTask(task_type="voice.dance", payload={}))
    assert result.success is False


# --------------------------------------------------------------------------
# Piper provider (default, local) — no synthesis, just wiring/guards
# --------------------------------------------------------------------------


def test_piper_provider_is_unavailable_without_model(tmp_path):
    from voice.piper_tts import PiperTTSProvider

    provider = PiperTTSProvider(tmp_path / "does_not_exist.onnx")
    assert provider.available is False


async def test_piper_speak_returns_empty_when_unavailable(tmp_path):
    from voice.piper_tts import PiperTTSProvider

    provider = PiperTTSProvider(tmp_path / "missing.onnx")
    assert await provider.speak("hello") == b""
