"""Voice package — 100% local speech I/O for JARVIS.

Providers are import-safe: constructing them without the optional native
dependencies installed will not raise; their ``available`` property reports
readiness so callers can degrade gracefully.
"""

from voice.interfaces import ISTTProvider, ITTSProvider, IWakeWordDetector
from voice.audio_io import AudioIO
from voice.piper_tts import PiperTTSProvider
from voice.whisper_stt import WhisperSTTProvider
from voice.wake_word import OpenWakeWordDetector
from voice.pipeline import VoicePipeline, VoiceTurn
from voice.vad import EnergyVAD, Endpointer, EndpointConfig
from voice.wake import (
    AlwaysAwake,
    TranscriptWakeWord,
    WakeResult,
    WakeStrategy,
    build_wake_strategy,
)
from voice.continuous_loop import (
    ContinuousVoiceLoop,
    LoopConfig,
    LoopState,
    LoopStats,
)

__all__ = [
    "ISTTProvider",
    "ITTSProvider",
    "IWakeWordDetector",
    "AudioIO",
    "PiperTTSProvider",
    "WhisperSTTProvider",
    "OpenWakeWordDetector",
    "VoicePipeline",
    "VoiceTurn",
    "EnergyVAD",
    "Endpointer",
    "EndpointConfig",
    "AlwaysAwake",
    "TranscriptWakeWord",
    "WakeResult",
    "WakeStrategy",
    "build_wake_strategy",
    "ContinuousVoiceLoop",
    "LoopConfig",
    "LoopState",
    "LoopStats",
]
