"""Wake-word strategies for the continuous voice loop.

A *wake strategy* decides whether a freshly transcribed utterance should
activate JARVIS. Kept deliberately simple and pluggable (Strategy pattern):

* :class:`TranscriptWakeWord` — the default. Reuses the existing Whisper STT:
  an utterance activates only if its transcript contains a wake word
  ("jarvis" / "computer"). Zero extra dependencies, fully local, and it also
  returns the *command* portion after the wake word so "Jarvis, what time is
  it?" activates and answers in one breath.

* :class:`AlwaysAwake` — no gating; every utterance is a command. Good for a
  dedicated hands-on session where you don't want to say the wake word each
  time.

An acoustic detector (openwakeword) can be added later behind this same
interface; it is intentionally *not* required, so the loop works out of the box.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_logger = logging.getLogger(__name__)

_DEFAULT_WAKE_WORDS: tuple[str, ...] = ("jarvis", "computer")


@dataclass(frozen=True, slots=True)
class WakeResult:
    """Outcome of testing an utterance for a wake word."""

    activated: bool
    command: str = ""  # text after the wake word, if any


@runtime_checkable
class WakeStrategy(Protocol):
    """Decides whether a transcript activates the assistant."""

    def check(self, transcript: str) -> WakeResult: ...


class TranscriptWakeWord:
    """Activate when the transcript contains a configured wake word.

    Matching is whole-word and case-insensitive. Everything after the wake
    word is returned as ``command`` so a single utterance can both wake and
    instruct ("Jarvis, set a reminder").
    """

    def __init__(self, wake_words: list[str] | tuple[str, ...] | None = None) -> None:
        words = tuple(w.strip().lower() for w in (wake_words or _DEFAULT_WAKE_WORDS) if w.strip())
        self._wake_words = words or _DEFAULT_WAKE_WORDS
        # Match "<wake>[,:]? <rest>" anywhere in the utterance.
        alternation = "|".join(re.escape(w) for w in self._wake_words)
        self._pattern = re.compile(rf"\b({alternation})\b[\s,:.!-]*", re.IGNORECASE)

    @property
    def wake_words(self) -> tuple[str, ...]:
        return self._wake_words

    def check(self, transcript: str) -> WakeResult:
        text = (transcript or "").strip()
        if not text:
            return WakeResult(activated=False)
        match = self._pattern.search(text)
        if not match:
            return WakeResult(activated=False)
        command = text[match.end():].strip(" ,.:!-")
        _logger.debug("Wake word matched (%s); command=%r", match.group(1), command)
        return WakeResult(activated=True, command=command)


class AlwaysAwake:
    """No wake gating — every utterance is treated as a command."""

    def check(self, transcript: str) -> WakeResult:
        text = (transcript or "").strip()
        return WakeResult(activated=bool(text), command=text)


def build_wake_strategy(
    mode: str, wake_words: list[str] | tuple[str, ...] | None = None,
) -> WakeStrategy:
    """Factory: ``"transcript"`` (default) or ``"none"`` → wake strategy."""
    if mode == "none":
        return AlwaysAwake()
    return TranscriptWakeWord(wake_words)
