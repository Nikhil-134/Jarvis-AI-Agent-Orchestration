"""Shared user-facing response guards.

Single source of truth for the substrings that mark a message as internal
machinery (must never reach a user) and the helper that decides whether a
piece of text is safe to display.  Centralised here so the runtime response
composer and the planning ResponseVerifier share exactly one list — no drift.
"""

from __future__ import annotations

import re

# Substrings that mark a message as internal machinery — it must never reach
# the user (e.g. "memory_id is required", "not installed", "cannot handle task
# type", stack traces, exception class names).
INTERNAL_MESSAGE_MARKERS: tuple[str, ...] = (
    "required for",
    "is required",
    "not installed",
    "not configured",
    "not available",
    "cannot handle",
    "unknown task type",
    "requires a browser engine",
    "traceback",
    "exception",
    "agentresult",
    "task_id",
    "task_type",
    "nonetype",
    "attributeerror",
    "keyerror",
    "unavailable",
)

# Leaked prompt scaffolding — the plain-text conversation format used by the
# KnowledgeEngine ("User: ... / Jarvis: ... / System: ...").  A final answer
# echoing these role labels means internal prompt structure leaked out.
_SCAFFOLD_MARKERS: tuple[str, ...] = (
    "system:",
)

# A raw tool-call JSON blob rather than natural language.
_TOOL_JSON_RE = re.compile(r'^\s*[\[{].*"(name|arguments)"\s*:', re.DOTALL)


def is_user_safe(text: str) -> bool:
    """Return True if *text* is safe to show a user (not internal machinery).

    Rejects empty/whitespace text and any text containing an internal-message
    marker.  This is the shared gate used by both the runtime composer and the
    planning verifier.
    """
    if not text or not text.strip():
        return False
    low = text.lower()
    return not any(marker in low for marker in INTERNAL_MESSAGE_MARKERS)


def looks_like_tool_json(text: str) -> bool:
    """Return True if *text* looks like a raw tool-call JSON payload."""
    return bool(_TOOL_JSON_RE.match(text or ""))


def has_leaked_scaffolding(text: str) -> bool:
    """Return True if *text* contains leaked prompt scaffolding / role labels."""
    low = (text or "").lower()
    if any(marker in low for marker in _SCAFFOLD_MARKERS):
        return True
    # "User:" / "Jarvis:" only count as leaks when they head a line (the prompt
    # format), not when they appear mid-sentence.
    return bool(re.search(r"(?im)^\s*(user|jarvis|assistant)\s*:", text or ""))
