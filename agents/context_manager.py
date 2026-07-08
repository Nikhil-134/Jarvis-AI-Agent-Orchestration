"""Context manager — tracks topics, entities, people, and pronoun references across conversation turns.

Enables follow-up question resolution like "Where was he born?" or
"Tell me more about that" by remembering what was discussed before.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


_PRONOMINAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(it|that|this|those|these)\b", re.IGNORECASE), "object"),
    (re.compile(r"\b(he|him|his)\b", re.IGNORECASE), "person_male"),
    (re.compile(r"\b(she|her|hers)\b", re.IGNORECASE), "person_female"),
    (re.compile(r"\b(they|them|their|theirs)\b", re.IGNORECASE), "person_plural"),
    (re.compile(r"\b(there)\b", re.IGNORECASE), "location"),
]

_FOLLOW_UP_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\b(all of it|all that|all those|all this)\b", re.IGNORECASE),
    re.compile(r"\b(the same|that one|these ones|those ones)\b", re.IGNORECASE),
    re.compile(r"\b(more (about|on|regarding)|tell me more|elaborate)\b", re.IGNORECASE),
    re.compile(r"\b(what about|how about|and)\s+(him|her|it|them|that)\b", re.IGNORECASE),
]

_EXTRACT_PERSON_RE = re.compile(
    r"\b(?:my name is|i am|i'm|called|name is|friends are|meet|know)\s+([A-Z][a-z]+)",
)

_EXTRACT_NAMED_ENTITIES_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
)
_EXTRACT_TOPIC_RE = re.compile(
    r"\b(?:about|regarding|concerning|on the topic of)\s+([A-Za-z]\S*(?:\s+[A-Za-z]\S*){0,5})",
)
_EXTRACT_ENTITY_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
)


@dataclass
class TurnContext:
    topic: str = ""
    entities: set[str] = field(default_factory=set)
    people: set[str] = field(default_factory=set)
    goal: str = ""
    response: str = ""


class ContextManager:
    """Tracks conversation context across turns for pronoun resolution
    and follow-up question handling.

    Maintains a short-term context window (last N turns) and exposes
    :meth:`enrich` to resolve references in new user input before
    it reaches the intent detector or planner.
    """

    def __init__(self, window_size: int = 5) -> None:
        self._window_size = window_size
        self._turns: list[TurnContext] = []

    @property
    def last_turn(self) -> TurnContext | None:
        return self._turns[-1] if self._turns else None

    @property
    def last_topic(self) -> str:
        for turn in reversed(self._turns):
            if turn.topic:
                return turn.topic
        return ""

    @property
    def last_entities(self) -> set[str]:
        merged: set[str] = set()
        for turn in reversed(self._turns[-3:]):
            merged.update(turn.entities)
        return merged

    @property
    def last_people(self) -> set[str]:
        merged: set[str] = set()
        for turn in reversed(self._turns):
            if turn.people:
                merged.update(turn.people)
                return merged
        return set()

    def update(self, goal: str, response: str) -> None:
        """Extract context from a completed turn and store it."""
        ctx = TurnContext(goal=goal, response=response)
        lower = goal.lower()

        people = _EXTRACT_PERSON_RE.findall(goal)
        for name in people:
            ctx.people.add(name.lower())

        # Also capture additional names after "friends are X and Y"
        if "friends are" in goal.lower():
            parts = goal.lower().split("friends are")[-1]
            for token in re.split(r"\s+and\s+|[,\s]+", parts):
                clean = token.strip("?!.,;:")
                if clean.istitle() and len(clean) > 1 and clean.lower() not in ("and",):
                    ctx.people.add(clean.lower())

        topics = _EXTRACT_TOPIC_RE.findall(goal)
        if topics:
            ctx.topic = topics[0].strip()

        if not ctx.topic and len(goal.split()) <= 6:
            ctx.topic = goal.strip().rstrip("?!.,")

        entities = _EXTRACT_ENTITY_RE.findall(goal)
        for e in entities:
            if e.lower() not in ("i", "my", "the", "a", "an", "it", "this", "that", "what", "why", "how", "who", "where", "when", "is", "are", "do", "does", "did", "can", "will", "would", "could", "should", "may", "might", "shall"):
                ctx.entities.add(e.lower())

        self._turns.append(ctx)
        if len(self._turns) > self._window_size:
            self._turns.pop(0)

        _logger.debug("Context updated: topic=%r people=%s entities=%d",
                      ctx.topic, ctx.people, len(ctx.entities))

    def enrich(self, goal: str) -> str:
        """Resolve pronouns and follow-up references in *goal* using stored context.

        Returns the enriched goal string with references replaced.
        """
        if not self._turns:
            return goal

        lower = goal.lower().strip()

        for phrase_re in _FOLLOW_UP_PHRASES:
            if phrase_re.search(lower):
                return self._resolve_follow_up(goal)

        for pronoun_re, target_type in _PRONOMINAL_PATTERNS:
            match = pronoun_re.search(lower)
            if match:
                resolved = self._resolve_pronoun(match.group(1), target_type)
                if resolved:
                    _logger.debug("Resolved '%s' → '%s'", match.group(1), resolved)
                    goal = goal[:match.start()] + resolved + goal[match.end():]
                    break

        return goal

    def _resolve_pronoun(self, pronoun: str, target_type: str) -> str:
        pronoun_lower = pronoun.lower()

        for turn in reversed(self._turns):
            if pronoun_lower in ("he", "him", "his") and turn.people:
                name = next(iter(turn.people))
                return name.title()
            if pronoun_lower in ("she", "her", "hers") and turn.people:
                name = next(iter(turn.people))
                return name.title()
            if pronoun_lower in ("they", "them", "their", "theirs") and turn.people:
                names = ", ".join(sorted(turn.people))
                return names.title()

        for turn in reversed(self._turns):
            if pronoun_lower in ("it", "that", "this") and turn.topic:
                return turn.topic

        for turn in reversed(self._turns):
            if pronoun_lower in ("there",) and turn.topic:
                return turn.topic

        return ""

    def _resolve_follow_up(self, goal: str) -> str:
        last = self.last_turn
        if last is None:
            return goal

        topic = last.topic or last.goal

        if re.search(r"\b(all of it|all that|all those)\b", goal, re.IGNORECASE):
            return f"Tell me everything about {topic}"
        if re.search(r"\b(tell me more|more about|elaborate|more on)\b", goal, re.IGNORECASE):
            return f"Tell me more about {topic}"
        if re.search(r"\b(what about|how about)\s+(him|her|it|them)\b", goal, re.IGNORECASE):
            people = self.last_people
            if people:
                return f"What about {', '.join(sorted(people))}"
            return f"Tell me about {topic}"

        return goal

    def clear(self) -> None:
        self._turns.clear()
        _logger.debug("Context cleared")

    def snapshot(self) -> dict[str, Any]:
        return {
            "turn_count": len(self._turns),
            "last_topic": self.last_topic,
            "last_entities": sorted(self.last_entities),
            "last_people": sorted(self.last_people),
        }
