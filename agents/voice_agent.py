"""Voice agent — text-to-speech output and speech-to-text input.

Wraps the local voice providers (Piper TTS, Whisper STT) so voice capabilities
are reachable through the standard agent/task interface. When no providers are
supplied the agent degrades gracefully and reports that voice is unavailable
rather than failing hard.

Supported task types:
    ``voice.output`` — payload ``{"text": str}`` → synthesises (and optionally
        plays) speech; returns audio byte length in ``data``.
    ``voice.input``  — records/transcribes and returns text in ``data``.
"""

from __future__ import annotations

import logging

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask
from voice.interfaces import ISTTProvider, ITTSProvider

_logger = logging.getLogger(__name__)


class VoiceAgent(Agent):
    """Agent responsible for voice input and output tasks."""

    def __init__(
        self,
        tts: ITTSProvider | None = None,
        stt: ISTTProvider | None = None,
        audio: "object | None" = None,
    ) -> None:
        super().__init__(name="voice", supported_task_types=("voice.input", "voice.output"))
        self._tts = tts
        self._stt = stt
        self._audio = audio  # AudioIO or None; kept loose to avoid a hard import

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"VoiceAgent cannot handle task type: {task.task_type}",
            )

        if task.task_type == "voice.output":
            return await self._handle_output(task)
        return await self._handle_input(task)

    async def _handle_output(self, task: AgentTask) -> AgentResult:
        text = str(task.payload.get("text", "")).strip()
        if not text:
            return self._fail(task, "No text provided for speech synthesis.")
        if self._tts is None or not getattr(self._tts, "available", False):
            return self._fail(task, "Text-to-speech is not available.")

        audio = await self._tts.speak(text)
        played = False
        if audio and self._audio is not None and getattr(self._audio, "available", False):
            try:
                self._audio.play_wav(audio)
                played = True
            except Exception:
                _logger.exception("Voice playback failed")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=bool(audio),
            message="",
            data={
                "status": "completed" if audio else "error",
                "audio_bytes": len(audio),
                "played": played,
                "response": text if audio else "I couldn't synthesise that audio.",
            },
        )

    async def _handle_input(self, task: AgentTask) -> AgentResult:
        # Allow a caller to pass pre-recorded WAV bytes; otherwise record live.
        wav: bytes = task.payload.get("audio", b"")
        if not wav and self._audio is not None and getattr(self._audio, "available", False):
            seconds = float(task.payload.get("seconds", 5.0))
            pcm = self._audio.record(seconds)
            if pcm:
                wav = self._audio.pcm_to_wav(pcm)

        if not wav:
            return self._fail(task, "No audio available to transcribe.")
        if self._stt is None or not getattr(self._stt, "available", False):
            return self._fail(task, "Speech-to-text is not available.")

        transcript = await self._stt.transcribe(wav)
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=bool(transcript),
            message="",
            data={
                "status": "completed" if transcript else "error",
                "transcript": transcript,
                "response": transcript,
            },
        )

    def _fail(self, task: AgentTask, message: str) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=False,
            message=message,
            data={"status": "unavailable", "response": message},
        )
