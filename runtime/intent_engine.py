"""Intent Engine — multi-intent classification with confidence scoring.

Moves beyond single-intent keyword routing to support multiple simultaneous
intents with confidence-based resolution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from tools.intent_detector import IntentDetector

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Intent:
    """A single detected intent with label and confidence."""

    label: str
    confidence: float

    def __bool__(self) -> bool:
        return self.confidence >= 0.70


@dataclass(frozen=True, slots=True)
class IntentResult:
    """Result of multi-intent classification."""

    primary: Intent
    secondary: list[Intent] = field(default_factory=list)
    goal: str = ""
    requires_browser: bool = False
    requires_planning: bool = False
    requires_tool: bool = False
    requires_knowledge: bool = False
    requires_conversation: bool = False
    requires_vision: bool = False

    @property
    def is_conversational(self) -> bool:
        return self.primary.label in ("greeting", "conversation", "thanks", "goodbye", "how_are_you")

    @property
    def is_actionable(self) -> bool:
        return self.primary.label not in (
            "greeting", "conversation", "thanks", "goodbye", "how_are_you",
            "unknown", "follow_up",
        )


class IntentEngine:
    """Multi-intent classifier that wraps the existing IntentDetector.

    Supports compound intents (e.g., "search today's AI news and summarize it"
    → [browser, knowledge]) with confidence scoring.

    Usage::

        engine = IntentEngine(intent_detector)
        result = engine.classify("Search AI news and summarize it")
        result.primary.label  # "browser"
        result.secondary      # [Intent("knowledge", 0.80)]
    """

    _COMPOUND_SPLITTER = re.compile(r"\s+(and\s+then|then\s+)+|,\s*(and\s+)?", re.IGNORECASE)

    _BROWSER_KEYWORDS: list[str] = [
        "search", "browse", "look up", "find", "google", "navigate",
    ]

    _KNOWLEDGE_KEYWORDS: list[str] = [
        "summarize", "explain", "what is", "who is", "tell me about",
    ]

    _CODING_KEYWORDS: list[str] = [
        "write code", "generate code", "implement", "create a function",
        "program", "script", "debug", "refactor", "review code",
    ]

    _VISION_KEYWORDS: list[str] = [
        "image", "picture", "photo", "screenshot", "ocr",
        "what's in this", "describe this",
    ]

    def __init__(self, detector: IntentDetector | None) -> None:
        self._detector = detector

    @classmethod
    def _has_keyword(cls, text: str, keywords: list[str]) -> bool:
        """Whole-word (or phrase) keyword match.

        Uses word boundaries so ``photo`` does not match ``photosynthesis``
        and ``scan`` does not match inside unrelated words. Multi-word phrases
        are matched as substrings since they are already specific.
        """
        for kw in keywords:
            if " " in kw:
                if kw in text:
                    return True
            elif re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        return False

    def classify(self, goal: str) -> IntentResult:
        """Classify *goal* into primary and secondary intents.

        For simple single-intent queries, delegates to IntentDetector.
        For compound queries (detected by "and then", commas), splits
        and classifies each segment independently.
        """
        if not self._detector:
            return IntentResult(
                primary=Intent("plan", 0.50),
                goal=goal,
                requires_planning=True,
            )

        has_compound = bool(self._COMPOUND_SPLITTER.search(goal))

        if not has_compound:
            single = self._detector.classify(goal)
            result = self._to_intent(single, goal)
            return result

        # NOTE: _COMPOUND_SPLITTER has capturing groups, so re.split interleaves
        # the captured separators — including None for the optional group that
        # didn't match. Guard against non-string / None entries before .strip().
        segments = self._COMPOUND_SPLITTER.split(goal)
        segments = [s.strip() for s in segments if isinstance(s, str) and s.strip()]

        intents: list[Intent] = []
        for segment in segments:
            single = self._detector.classify(segment)
            intents.append(Intent(label=single.label, confidence=single.confidence))

        if not intents:
            return IntentResult(primary=Intent("unknown", 0.0), goal=goal)

        intents.sort(key=lambda i: i.confidence, reverse=True)
        primary_intent = intents[0]
        secondary = intents[1:] if len(intents) > 1 else []

        lower = goal.lower()

        return IntentResult(
            primary=primary_intent,
            secondary=secondary,
            goal=goal,
            requires_browser=self._has_keyword(lower, self._BROWSER_KEYWORDS)
            or primary_intent.label == "current_info",
            requires_planning=primary_intent.label in (
                "plan", "coding", "security", "devops", "shell",
            ) or (primary_intent.label == "browser" and self._has_keyword(lower, self._CODING_KEYWORDS)),
            requires_tool=primary_intent.label in ("tool",),
            requires_knowledge=primary_intent.label in (
                "knowledge_question", "follow_up",
            ) or primary_intent.confidence < 0.70,
            requires_conversation=primary_intent.label in (
                "greeting", "conversation",
            ),
            requires_vision=self._has_keyword(lower, self._VISION_KEYWORDS),
        )

    def _to_intent(self, classification: Any, goal: str) -> IntentResult:
        label = classification.label if hasattr(classification, "label") else str(classification)
        confidence = classification.confidence if hasattr(classification, "confidence") else 0.0

        lower = goal.lower()

        return IntentResult(
            primary=Intent(label=label, confidence=confidence),
            goal=goal,
            requires_browser=label == "current_info"
            or self._has_keyword(lower, self._BROWSER_KEYWORDS),
            requires_planning=label in ("plan", "coding", "security", "devops", "shell", "browser", "desktop", "calendar", "reminder", "email", "notes"),
            requires_tool=label == "tool",
            requires_knowledge=label in ("knowledge_question", "follow_up", "unknown") or confidence < 0.70,
            requires_conversation=label in ("greeting", "conversation"),
            requires_vision=self._has_keyword(lower, self._VISION_KEYWORDS),
        )
