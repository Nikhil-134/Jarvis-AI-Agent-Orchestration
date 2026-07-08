"""Reflection memory — turn raw conversation into durable insight.

Instead of retraining the model, JARVIS *reflects*: after a conversation it
distils a short summary plus the decisions made, tasks left open, and lessons
learned, and stores those as first-class memories. Over time this is what lets
retrieval answer "what did we decide?" or "what was left to do?" without the
full transcript.

Design:

* An LLM does the extraction when one is injected, constrained to emit strict
  JSON that is parsed defensively.
* If no LLM is available *or* the model returns unusable output, a deterministic
  keyword heuristic produces a real (if coarser) reflection — never a fabricated
  "success". The result always reports which path produced it via ``source``.

The engine performs no storage itself (single responsibility); the
:class:`~memory.persistent_memory.PersistentMemoryService` persists the result.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from memory.validation import sanitize_text

_logger = logging.getLogger(__name__)


@runtime_checkable
class _Summariser(Protocol):
    """Minimal LLM contract the reflection engine depends on."""

    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str: ...


@dataclass
class Reflection:
    """Structured insight distilled from a conversation."""

    summary: str = ""
    decisions: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    source: str = "heuristic"  # "llm" | "heuristic"

    def is_empty(self) -> bool:
        return not (self.summary or self.decisions or self.tasks or self.lessons)


_SYSTEM_PROMPT = (
    "You are a reflection module. Read a conversation transcript and extract "
    "durable insight. Respond with ONLY a JSON object with keys: "
    '"summary" (string), "decisions" (string array), "tasks" (string array), '
    '"lessons" (string array). No prose, no code fences.'
)

# Heuristic cue words for the no-LLM / fallback path.
_DECISION_CUES = ("decide", "decided", "we'll use", "we will use", "let's go with", "chosen", "agreed")
_TASK_CUES = ("todo", "to do", "need to", "should ", "next step", "fix ", "implement", "follow up", "follow-up")
_LESSON_CUES = ("learned", "lesson", "turns out", "realised", "realized", "note that", "gotcha")

_MAX_ITEMS = 12
_MAX_ITEM_CHARS = 300


class ReflectionEngine:
    """Extract a :class:`Reflection` from a conversation transcript."""

    def __init__(self, llm: _Summariser | None = None) -> None:
        self._llm = llm

    async def reflect(self, transcript: str) -> Reflection:
        """Return a :class:`Reflection` for *transcript* (never raises)."""
        text = sanitize_text(transcript, max_chars=20_000)
        if not text:
            return Reflection()

        if self._llm is not None:
            reflection = await self._reflect_with_llm(text)
            if reflection is not None and not reflection.is_empty():
                return reflection
            _logger.debug("Reflection LLM produced no usable output; using heuristic")

        return self._reflect_heuristic(text)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _reflect_with_llm(self, text: str) -> Reflection | None:
        prompt = f"Conversation transcript:\n\n{text}\n\nExtract the JSON reflection."
        try:
            raw = await self._llm.generate_text(prompt, system_prompt=_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001 - degrade to heuristic on any LLM error
            _logger.warning("Reflection LLM call failed; falling back to heuristic", exc_info=True)
            return None

        data = self._extract_json_object(raw or "")
        if data is None:
            return None

        return Reflection(
            summary=self._clean_str(data.get("summary", "")),
            decisions=self._clean_list(data.get("decisions")),
            tasks=self._clean_list(data.get("tasks")),
            lessons=self._clean_list(data.get("lessons")),
            source="llm",
        )

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any] | None:
        """Best-effort parse of a JSON object from a possibly noisy response."""
        raw = raw.strip()
        # Strip a leading/trailing code fence if the model added one.
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):] if "{" in raw else raw
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                obj = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                return None
        return obj if isinstance(obj, dict) else None

    # ------------------------------------------------------------------
    # Heuristic path
    # ------------------------------------------------------------------

    def _reflect_heuristic(self, text: str) -> Reflection:
        lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
        decisions: list[str] = []
        tasks: list[str] = []
        lessons: list[str] = []
        for line in lines:
            low = line.lower()
            if any(cue in low for cue in _DECISION_CUES):
                decisions.append(line)
            if any(cue in low for cue in _TASK_CUES):
                tasks.append(line)
            if any(cue in low for cue in _LESSON_CUES):
                lessons.append(line)

        # Summary = the first couple of sentences, bounded.
        summary = " ".join(re.split(r"(?<=[.!?])\s+", text)[:2])[:_MAX_ITEM_CHARS]

        return Reflection(
            summary=self._clean_str(summary),
            decisions=self._dedup_cap(decisions),
            tasks=self._dedup_cap(tasks),
            lessons=self._dedup_cap(lessons),
            source="heuristic",
        )

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_str(value: Any) -> str:
        return sanitize_text(str(value), max_chars=_MAX_ITEM_CHARS)

    @classmethod
    def _clean_list(cls, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return cls._dedup_cap([cls._clean_str(v) for v in value])

    @staticmethod
    def _dedup_cap(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for it in items:
            it = it.strip()
            key = it.lower()
            if it and key not in seen:
                seen.add(key)
                out.append(it[:_MAX_ITEM_CHARS])
            if len(out) >= _MAX_ITEMS:
                break
        return out
