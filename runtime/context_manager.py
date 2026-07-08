"""Enhanced context manager — multi-user, multi-session context tracking.

Tracks current topic, entities, people, pronouns, last browser result,
last file, last tool, last response across conversation turns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Context about the most recently used tool."""

    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = False
    timestamp: float = 0.0


@dataclass
class BrowserContext:
    """Context about the most recent browser interaction."""

    url: str = ""
    title: str = ""
    content_snippet: str = ""


@dataclass
class FileContext:
    """Context about the most recently accessed file."""

    path: str = ""
    operation: str = ""
    content_snippet: str = ""


@dataclass
class TurnRecord:
    """Complete record of one conversation turn."""

    goal: str = ""
    enriched_goal: str = ""
    response: str = ""
    topic: str = ""
    entities: set[str] = field(default_factory=set)
    people: set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    intent_label: str = ""
    intent_confidence: float = 0.0
    tool: ToolContext = field(default_factory=ToolContext)


@dataclass
class SessionContext:
    """All context for one user session."""

    session_id: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_tool: ToolContext = field(default_factory=ToolContext)
    last_browser: BrowserContext = field(default_factory=BrowserContext)
    last_file: FileContext = field(default_factory=FileContext)

    @property
    def last_turn(self) -> TurnRecord | None:
        return self.turns[-1] if self.turns else None

    @property
    def last_topic(self) -> str:
        for turn in reversed(self.turns):
            if turn.topic:
                return turn.topic
        return ""

    @property
    def last_people(self) -> set[str]:
        for turn in reversed(self.turns):
            if turn.people:
                return turn.people
        return set()

    @property
    def last_entities(self) -> set[str]:
        merged: set[str] = set()
        for turn in reversed(self.turns[-3:]):
            merged.update(turn.entities)
        return merged

    @property
    def turn_count(self) -> int:
        return len(self.turns)


_EXTRACT_PERSON_RE = re.compile(
    r"\b(?:my name is|i am|i'm|called|name is|friends are|meet|know)\s+([A-Z][a-z]+)",
)

_PRONOMINAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(it|that|this|those|these)\b", re.IGNORECASE), "object"),
    (re.compile(r"\b(he|him|his)\b", re.IGNORECASE), "person_male"),
    (re.compile(r"\b(she|her|hers)\b", re.IGNORECASE), "person_female"),
    (re.compile(r"\b(they|them|their|theirs)\b", re.IGNORECASE), "person_plural"),
]

_FOLLOW_UP_RE = re.compile(
    r"\b(all of it|all of that|all that|all those|all this|"
    r"tell me more|more about|elaborate|go on|continue|"
    r"the same|that one)\b",
    re.IGNORECASE,
)


class ContextManager:
    """Multi-user, multi-session context tracking.

    Manages per-session context state including topic tracking,
    pronoun resolution, entity tracking, and tool/browser/file context.

    Usage::

        cm = ContextManager()
        cm.update_session("session-1", "What is the weather in Bangalore", "Sunny", intent_label="current_info")
        enriched = cm.enrich("session-1", "What about tomorrow?")
        # Returns: "What about tomorrow in Bangalore?"
    """

    def __init__(self, max_turns_per_session: int = 20) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._max_turns = max_turns_per_session

    def get_or_create_session(self, session_id: str) -> SessionContext:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id=session_id)
        return self._sessions[session_id]

    def update_session(
        self,
        session_id: str,
        goal: str,
        response: str,
        *,
        intent_label: str = "",
        intent_confidence: float = 0.0,
        enriched_goal: str = "",
        tool_name: str = "",
        tool_args: dict[str, Any] | None = None,
        tool_result: str = "",
        tool_success: bool = False,
        browser_url: str = "",
        browser_title: str = "",
        browser_content: str = "",
        file_path: str = "",
        file_operation: str = "",
        file_content: str = "",
    ) -> TurnRecord:
        session = self.get_or_create_session(session_id)

        turn = TurnRecord(
            goal=goal,
            enriched_goal=enriched_goal or goal,
            response=response,
            intent_label=intent_label,
            intent_confidence=intent_confidence,
        )

        lower = goal.lower()

        people = _EXTRACT_PERSON_RE.findall(goal)
        for name in people:
            turn.people.add(name.lower())
        if "friends are" in lower:
            parts = lower.split("friends are")[-1]
            for token in re.split(r"\s+and\s+|[,\s]+", parts):
                clean = token.strip("?!.,;:")
                if clean.istitle() and len(clean) > 1:
                    turn.people.add(clean.lower())

        topic_match = re.search(
            r"\b(?:about|regarding|concerning|on the topic of)\s+(.+)", goal, re.IGNORECASE,
        )
        if topic_match:
            turn.topic = topic_match.group(1).strip().rstrip("?!.,")
        elif len(goal.split()) <= 8 and "?" not in goal:
            turn.topic = goal.strip().rstrip("?!.,")
        else:
            words = [w for w in lower.split() if len(w) > 3 and w not in (
                "what", "when", "where", "why", "how", "who", "which",
                "tell", "show", "give", "find", "search", "please",
                "could", "would", "should", "can", "will", "shall",
                "this", "that", "these", "those", "there", "their",
            )]
            if words:
                turn.topic = " ".join(words[:5])

        entity_matches = re.findall(r"\b([A-Z][a-z]{2,})\b", goal)
        for e in entity_matches:
            lower_e = e.lower()
            if lower_e not in ("the", "you", "your", "how", "why", "what", "who", "where", "when", "are", "can", "will", "did", "was", "were", "has", "had", "not", "but", "for", "and", "its", "all", "any", "may", "now"):
                turn.entities.add(lower_e)

        if tool_name:
            turn.tool = ToolContext(
                tool_name=tool_name,
                arguments=tool_args or {},
                result=tool_result,
                success=tool_success,
                timestamp=datetime.now(timezone.utc).timestamp(),
            )
            session.last_tool = turn.tool

        if browser_url:
            session.last_browser = BrowserContext(
                url=browser_url,
                title=browser_title,
                content_snippet=browser_content[:500],
            )

        if file_path:
            session.last_file = FileContext(
                path=file_path,
                operation=file_operation,
                content_snippet=file_content[:500],
            )

        session.turns.append(turn)
        if len(session.turns) > self._max_turns:
            session.turns.pop(0)

        _logger.debug(
            "Context[%s] updated: topic=%r people=%s entities=%d tools=%d",
            session_id, turn.topic, turn.people, len(turn.entities),
            1 if tool_name else 0,
        )

        return turn

    def enrich(self, session_id: str, goal: str) -> str:
        """Resolve pronouns and follow-up references in *goal*."""
        session = self._sessions.get(session_id)
        if not session or not session.turns:
            return goal

        lower = goal.lower().strip()

        if _FOLLOW_UP_RE.search(lower):
            return self._resolve_follow_up(session, goal)

        for pronoun_re, target_type in _PRONOMINAL_PATTERNS:
            match = pronoun_re.search(lower)
            if match:
                resolved = self._resolve_pronoun(session, match.group(1), target_type)
                if resolved:
                    _logger.debug("Resolved '%s' → '%s'", match.group(1), resolved)
                    return goal[:match.start()] + resolved + goal[match.end():]

        return goal

    def _resolve_pronoun(self, session: SessionContext, pronoun: str, target_type: str) -> str:
        p_lower = pronoun.lower()

        for turn in reversed(session.turns):
            if p_lower in ("he", "him", "his") and turn.people:
                return next(iter(turn.people)).title()
            if p_lower in ("she", "her", "hers") and turn.people:
                return next(iter(turn.people)).title()
            if p_lower in ("they", "them", "their", "theirs") and turn.people:
                return ", ".join(sorted(turn.people)).title()
            if p_lower in ("it", "that", "this") and turn.topic:
                return turn.topic

        return ""

    def _resolve_follow_up(self, session: SessionContext, goal: str) -> str:
        last = session.last_turn
        if last is None:
            return goal

        topic = last.topic or last.goal

        if re.search(r"\b(all of it|all that|all those)\b", goal, re.IGNORECASE):
            return f"Tell me everything about {topic}"
        if re.search(r"\b(tell me more|more about|elaborate|more on)\b", goal, re.IGNORECASE):
            return f"Tell me more about {topic}"
        if re.search(r"\b(what about|how about)\s+(him|her|it|them)\b", goal, re.IGNORECASE):
            people = session.last_people
            if people:
                return f"What about {', '.join(sorted(people))}"
            return f"Tell me about {topic}"

        return goal

    def get_context_summary(self, session_id: str) -> str:
        """Return a human-readable summary of current context."""
        session = self._sessions.get(session_id)
        if not session:
            return "No conversation context."

        parts = [f"Conversation turns: {session.turn_count}"]
        if session.last_topic:
            parts.append(f"Current topic: {session.last_topic}")
        if session.last_people:
            parts.append(f"People mentioned: {', '.join(sorted(session.last_people))}")
        if session.last_tool.tool_name:
            parts.append(f"Last tool: {session.last_tool.tool_name}")
        if session.last_browser.url:
            parts.append(f"Last browser: {session.last_browser.url}")
        if session.last_file.path:
            parts.append(f"Last file: {session.last_file.path}")
        return " | ".join(parts)

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        _logger.debug("Context cleared for session '%s'", session_id)

    def clear_all(self) -> None:
        self._sessions.clear()
        _logger.debug("All context cleared")
