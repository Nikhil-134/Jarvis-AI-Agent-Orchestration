"""Knowledge subsystems for JARVIS.

Currently exposes the internet retrieval layer (roadmap #9), a *context-only*
last-resort source that supplements — never replaces — local memory and the
local model. Reasoning always stays inside Qwen; the internet supplies facts,
not decisions.
"""

from __future__ import annotations

from knowledge.internet import (
    InternetKnowledgeService,
    IRetrievalProvider,
    RetrievalResult,
    needs_internet,
)

__all__ = [
    "InternetKnowledgeService",
    "IRetrievalProvider",
    "RetrievalResult",
    "needs_internet",
]
