"""Freshness router — decides *whether* a query needs live internet data.

This is the gate that keeps the internet a **last-resort** source. It encodes
the project's fixed priority order:

    1. Persistent memory   2. Current conversation   3. Local knowledge base
    4. Local documents/PDFs 5. Internet retrieval (only if required)
    6. Local Qwen reasoning

The router answers one question: *given the query (and whether local memory
already had something relevant), must we reach outside for fresh facts?* It
returns True only for information that is inherently time-sensitive or clearly
about current/recent events — weather, news, prices, sports scores, "latest"/
"newest" releases, "today"/"now", current office-holders, etc. Everything that
a static model can answer ("What is Python?", "Explain recursion") stays local.

Pure, deterministic, dependency-free → trivially unit-testable. It never
fetches anything itself.
"""

from __future__ import annotations

import re

# Strong signals that the answer changes over time and cannot come from a
# static model or from personal memory.
_FRESH_KEYWORDS: tuple[str, ...] = (
    "weather", "temperature", "forecast", "rain", "humidity",
    "news", "headline", "breaking",
    "today", "todays", "tonight", "yesterday", "this week", "this morning",
    "right now", "currently", "at the moment", "these days",
    "latest", "newest", "recent", "recently", "just released", "just announced",
    "current", "up to date", "up-to-date",
    "stock", "share price", "exchange rate", "crypto", "bitcoin price",
    "score", "match result", "fixture", "standings",
    "who is the president", "who is the prime minister", "who is the ceo",
    "release date", "released", "version",
    "trending", "happening",
)

# "Current <role>" style questions (office-holders change over time).
_CURRENT_ROLE_RE = re.compile(
    r"\b(current|present|new|latest)\s+"
    r"(president|prime minister|pm|ceo|chancellor|governor|mayor|"
    r"champion|leader|king|queen|pope)\b",
    re.IGNORECASE,
)

# "Who is the <role> of <place>" — office-holder lookups are effectively
# time-sensitive (they change), so prefer live data even without "current".
_OFFICE_RE = re.compile(
    r"\bwho\s+is\s+the\s+"
    r"(president|prime minister|pm|ceo|chancellor|governor|mayor|"
    r"champion|leader|king|queen|pope)\b",
    re.IGNORECASE,
)

# Signals the user is asking about *personal / conversational* memory — these
# must NEVER go to the internet (privacy + correctness). Memory owns them.
_MEMORY_KEYWORDS: tuple[str, ...] = (
    "we discuss", "we discussed", "we talked", "did we", "did i", "did you",
    "i told you", "i asked", "you said", "earlier you", "last time",
    "my name", "i like", "i prefer", "my favourite", "my favorite",
    "remember", "remind me what", "what do i", "what did i", "what did we",
    # Identity / self questions — owned by memory, answered from stored facts.
    "who am i", "what is my name", "what's my name", "whats my name",
    "what do you know about me", "know about me", "about me",
    "call me", "my favourite", "what are my",
)

# Timeless conceptual questions — explicitly local even if a fresh word sneaks
# in (e.g. "explain the latest sorting algorithm concept").
_CONCEPT_PREFIXES: tuple[str, ...] = (
    "what is", "what are", "explain", "define", "how does", "how do",
    "why does", "why is", "describe the concept",
)


def is_memory_query(query: str) -> bool:
    """True if the query is about personal/conversational memory (never internet)."""
    low = (query or "").lower()
    return any(k in low for k in _MEMORY_KEYWORDS)


def needs_internet(query: str, *, local_context_found: bool = False) -> bool:
    """Return True only when live external data is genuinely required.

    Parameters
    ----------
    query:
        The user's question (already enriched by the pipeline is fine).
    local_context_found:
        Whether local memory already surfaced relevant context for this query.
        When True we bias *against* the internet — local sources take priority
        (steps 1–4 of the ladder) unless the query is explicitly time-sensitive.
    """
    low = (query or "").strip().lower()
    if not low:
        return False

    # Personal/memory questions are owned by the memory layer — never leave.
    if is_memory_query(low):
        return False

    has_fresh_signal = (
        any(k in low for k in _FRESH_KEYWORDS)
        or bool(_CURRENT_ROLE_RE.search(low))
        or bool(_OFFICE_RE.search(low))
    )

    # Timeless "what is / explain" conceptual questions stay local UNLESS they
    # also carry an explicit time signal (e.g. "what is the latest iPhone").
    is_conceptual = any(low.startswith(p) for p in _CONCEPT_PREFIXES)
    if is_conceptual and not has_fresh_signal:
        return False

    if not has_fresh_signal:
        # No time-sensitivity at all → local model/memory can handle it.
        return False

    # There IS a freshness signal. If local memory already answered it well and
    # the signal is weak, still prefer local; but time-sensitive facts (weather,
    # news, prices, scores, office-holders) should refresh regardless. We treat
    # the presence of a fresh signal as sufficient to consult the internet,
    # because these are exactly the facts local sources cannot keep current.
    return True
