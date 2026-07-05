"""Voice system interface definitions for Jarvis.

These interfaces define contracts for speech-to-text, text-to-speech,
wake-word detection, and voice-activity detection.  Implementations
will be added in a future phase.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable


class ISTTProvider(ABC):
    """Interface for speech-to-text providers.

    Implementations: faster-whisper, whisper.cpp, OpenAI Whisper API.
    """

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""

    @abstractmethod
    async def transcribe_stream(self, audio_stream: AsyncIterable[bytes]) -> str:
        """Transcribe a stream of audio chunks to text."""


class ITTSProvider(ABC):
    """Interface for text-to-speech providers.

    Implementations: Piper TTS, edge-tts, OpenAI TTS.
    """

    @abstractmethod
    async def speak(self, text: str) -> bytes:
        """Synthesise text to audio bytes (WAV or MP3)."""

    @abstractmethod
    async def speak_stream(self, text: str) -> AsyncIterable[bytes]:
        """Synthesise text to a stream of audio chunks."""


class IWakeWordDetector(ABC):
    """Interface for wake-word / hotword detection.

    Implementations: OpenWakeWord, Porcupine, Silero VAD + keyword.
    """

    @abstractmethod
    async def detect(self, audio_stream: AsyncIterable[bytes]) -> AsyncIterable[str]:
        """Yield detected wake-word labels from an audio stream."""
