"""Memory data models and value objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MemoryType(str, Enum):
    """Categorisation of stored memory entries.

    The first block is the original taxonomy (kept for backward compatibility
    with existing stored documents). The second block was added for the
    persistent-memory layer (sessions, projects, reflection, user profile).
    All values are stable strings — never rename an existing value, or old
    rows in the document store would fail to deserialise.
    """

    # Original taxonomy
    CONVERSATION = "conversation"
    FACT = "fact"
    SUMMARY = "summary"
    PREFERENCE = "preference"
    WORKING = "working"

    # Persistent-memory taxonomy
    PROJECT = "project"
    TASK = "task"
    DECISION = "decision"
    IDEA = "idea"
    REFLECTION = "reflection"
    USER_PROFILE = "user_profile"
    MEETING_NOTES = "meeting_notes"


_IMPORTANT_KEYWORDS = frozenset({
    "important", "critical", "urgent", "remember", "key",
    "essential", "vital", "crucial", "significant", "notable",
})


@dataclass
class MemoryItem:
    """A single unit of memory stored in the system.

    Combines metadata used by the vector store (embedding, content)
    and the document store (structured metadata).
    """

    content: str
    memory_type: MemoryType = MemoryType.CONVERSATION
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    embedding: list[float] | None = None

    @property
    def age_seconds(self) -> float:
        """Seconds since this memory was created."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def is_expired(self, max_age_days: int = 30) -> bool:
        """Return True if this memory is older than *max_age_days*."""
        return self.age_seconds > max_age_days * 86400

    def to_document(self) -> dict[str, Any]:
        """Serialise to a dict suitable for the document store."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "metadata": json.dumps(self.metadata, default=str),
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> MemoryItem:
        """Deserialise from a document store record."""
        return cls(
            id=doc["id"],
            content=doc["content"],
            memory_type=MemoryType(doc["memory_type"]),
            importance=doc["importance"],
            metadata=json.loads(doc.get("metadata", "{}")),
            created_at=datetime.fromisoformat(doc["created_at"]),
            accessed_at=datetime.fromisoformat(doc.get("accessed_at", doc["created_at"])),
            access_count=doc.get("access_count", 0),
        )

    def to_vector_metadata(self) -> dict[str, str | float | int]:
        """Serialise to metadata dict suitable for the vector store."""
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_vector_metadata(
        cls, metadata: dict[str, Any], embedding: list[float] | None = None, distance: float | None = None
    ) -> MemoryItem:
        """Reconstruct from vector store metadata, adding optional score."""
        item = cls(
            id=metadata.get("id", str(uuid4())),
            content=metadata.get("content", ""),
            memory_type=MemoryType(metadata.get("memory_type", "conversation")),
            importance=metadata.get("importance", 0.5),
            metadata={},
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now(timezone.utc).isoformat())),
            access_count=metadata.get("access_count", 0),
            embedding=embedding,
        )
        if distance is not None:
            item.metadata["relevance_score"] = 1.0 - min(distance, 1.0)
        return item


def calculate_importance(content: str, memory_type: MemoryType = MemoryType.CONVERSATION) -> float:
    """Calculate an importance score in [0, 1] for *content*.

    Factors considered:
    - Explicit importance keywords (+0.2)
    - Preferences (+0.2 over baseline)
    - Content length as a proxy for detail (+0.05 to +0.1)
    """
    score = 0.5

    words = set(content.lower().split())
    if words & _IMPORTANT_KEYWORDS:
        score += 0.2

    if memory_type == MemoryType.PREFERENCE:
        score += 0.2

    length = len(content)
    if length > 500:
        score += 0.1
    elif length > 200:
        score += 0.05

    return min(score, 1.0)
