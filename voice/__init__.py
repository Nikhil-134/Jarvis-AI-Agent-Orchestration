"""Voice package exports."""

from typing import Any

__all__ = [
    "ISTTProvider",
    "ITTSProvider",
    "IWakeWordDetector",
]


def __getattr__(name: str) -> Any:
    if name in {"ISTTProvider", "ITTSProvider", "IWakeWordDetector"}:
        from voice.interfaces import ISTTProvider, ITTSProvider, IWakeWordDetector
        return {
            "ISTTProvider": ISTTProvider,
            "ITTSProvider": ITTSProvider,
            "IWakeWordDetector": IWakeWordDetector,
        }[name]
    raise AttributeError(f"module 'voice' has no attribute {name!r}")
