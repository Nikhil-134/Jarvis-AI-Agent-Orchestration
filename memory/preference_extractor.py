"""Preference extraction — turn natural conversation into structured preferences.

This closes roadmap #11: *"remember I like X" currently stores a conversation
turn (recalled semantically) but does not auto-promote to a structured
``set_preference``.*  The conversation runtime already records every meaningful
turn durably; this module adds the missing step — deterministically spotting
identity/preference statements ("call me boss", "my favourite language is Rust",
"I live in Bangalore") and mapping them to canonical ``(key, value)`` pairs so
the :class:`~memory.persistent_memory.PersistentMemoryService` can promote them
to fast, exact-recall structured preferences.

Design principles (matching the rest of the memory layer):

* **Deterministic, local, ₹0.** Pure regex — no LLM call, no network, instantly
  unit-testable. The local model is never asked to "extract preferences" (that
  would be slow, non-deterministic, and could hallucinate).
* **Precision over recall.** A *false* preference ("favourite = to think about
  it") pollutes the durable profile and is worse than a miss, so the patterns
  are deliberately tight and every value is validated. Ambiguous forms
  ("I am tired", "I like it") are intentionally rejected.
* **No new storage.** This module only *extracts*; storage stays the
  responsibility of the existing ``set_preference`` (single-writer discipline).
* **Statements only, never questions.** "What is my name?" must not set a name.

It does not duplicate any existing component — there was no preference
extraction anywhere in the codebase before this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical preference keys. Kept stable — they become durable ``pref:<key>``
# document ids (upsert), and the KnowledgeEngine injects the whole profile for
# personal/identity questions, so renaming one would orphan stored rows.
KEY_NAME = "name"
KEY_PREFERRED_LANGUAGE = "preferred_language"
KEY_LOCATION = "location"
KEY_OCCUPATION = "occupation"
KEY_CODING_STYLE = "coding_style"
KEY_LIKES = "likes"

# Values that are never a real preference target — pronouns / fillers that show
# up after "I like ..." / "call me ..." when the user isn't actually stating a
# durable fact ("I like it", "call me later").
_STOP_VALUES = frozenset({
    "it", "that", "this", "them", "these", "those", "you", "me", "him", "her",
    "us", "later", "back", "now", "then", "here", "there", "so", "too", "again",
    "a", "an", "the", "to", "of", "and", "or", "but", "if", "when", "why", "how",
    "tired", "busy", "sorry", "happy", "sad", "sure", "ok", "okay", "fine",
    "good", "great", "done", "ready", "right", "wrong", "hungry", "bored",
})

# Words that, if they *start* the captured value, mean it's a verb phrase or a
# clause rather than a concrete preference ("I like to code", "I prefer using X"
# is fine but "I like to think about it" is not a preference value).
_VERB_LEADS = frozenset({
    "to", "that", "when", "how", "being", "having", "getting", "doing",
    "thinking", "going", "working", "trying",
})

# Roles that make "I am a <X>" an occupation statement (vs. "I am tired").
_OCCUPATION_WORDS = (
    "developer", "engineer", "programmer", "designer", "architect", "scientist",
    "analyst", "manager", "student", "teacher", "professor", "researcher",
    "consultant", "founder", "ceo", "cto", "administrator", "admin", "devops",
    "tester", "writer", "author", "artist", "doctor", "lawyer", "accountant",
    "nurse", "freelancer", "intern",
)

_MAX_VALUE_CHARS = 60


@dataclass(frozen=True, slots=True)
class ExtractedPreference:
    """A single structured preference distilled from an utterance."""

    key: str
    value: str


class PreferenceExtractor:
    """Extract structured user preferences from a single user utterance.

    Usage::

        extractor = PreferenceExtractor()
        prefs = extractor.extract("call me Boss and my favourite language is Rust")
        # [ExtractedPreference("name", "Boss"),
        #  ExtractedPreference("preferred_language", "Rust")]
    """

    # Ordered most-specific → least-specific. Each entry is
    # (compiled pattern, canonical key, value-group index).
    # ``favou?rite`` accepts both spellings.
    _NAME_PATTERNS = (
        re.compile(r"\bcall me\s+([A-Za-z][\w'’\-]{0,29})", re.IGNORECASE),
        re.compile(r"\bmy name'?s?\s+is\s+([A-Za-z][\w'’\-]{0,29})", re.IGNORECASE),
        re.compile(r"\bmy name'?s\s+([A-Za-z][\w'’\-]{0,29})", re.IGNORECASE),
        re.compile(r"\byou can call me\s+([A-Za-z][\w'’\-]{0,29})", re.IGNORECASE),
    )

    _LANGUAGE_PATTERNS = (
        re.compile(
            r"\bmy (?:favou?rite|preferred)\s+(?:programming\s+|coding\s+)?language\s+is\s+([\w+#.\- ]{1,30})",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bi (?:like|love|prefer|enjoy)\s+(?:to\s+)?(?:code|coding|program|programming|write|writing)\s+in\s+([\w+#.\- ]{1,30})",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bi (?:like|love|prefer)\s+([\w+#.\-]{1,30})\s+(?:for|as)\s+(?:my\s+)?(?:coding|programming|main language)",
            re.IGNORECASE,
        ),
    )

    _LOCATION_PATTERNS = (
        re.compile(r"\bi live in\s+([\w'’\-][\w'’\- ]{1,40})", re.IGNORECASE),
        re.compile(r"\bi'?m (?:based|located)\s+in\s+([\w'’\-][\w'’\- ]{1,40})", re.IGNORECASE),
        re.compile(r"\bi'?m from\s+([\w'’\-][\w'’\- ]{1,40})", re.IGNORECASE),
    )

    _OCCUPATION_PATTERNS = (
        re.compile(
            r"\bi work as (?:an?\s+)?((?:[\w'’\-]+\s+){0,3}[\w'’\-]+)",
            re.IGNORECASE,
        ),
        # "I'm a senior data scientist", "I am a developer" — anchored on a role
        # word so "I am tired" / "I'm happy" never match. Up to 3 qualifier
        # words are captured before the role (e.g. "senior data scientist").
        re.compile(
            rf"\bi(?:'?m| am)\s+(?:an?\s+)((?:[\w'’\-]+\s+){{0,3}}(?:{'|'.join(_OCCUPATION_WORDS)}))\b",
            re.IGNORECASE,
        ),
    )

    _CODING_STYLE_PATTERNS = (
        re.compile(r"\bi (?:prefer|like|use)\s+(tabs|spaces|\d+[\- ]?space[s]?)\b", re.IGNORECASE),
        re.compile(r"\bmy (?:coding|code)\s+style\s+is\s+([\w'’\-][\w'’\- ]{1,40})", re.IGNORECASE),
    )

    # Generic "favourite <thing> is <value>" → key ``favorite_<thing>``.
    _FAVORITE_GENERIC = re.compile(
        r"\bmy favou?rite\s+([a-z][a-z ]{1,20}?)\s+is\s+([\w'’+#.\-][\w'’+#.\- ]{0,40})",
        re.IGNORECASE,
    )

    # Last-resort generic like/prefer → ``likes`` (lowest priority; guarded).
    _LIKE_GENERIC = re.compile(
        r"\bi (?:really\s+)?(?:like|love|prefer|enjoy)\s+([\w'’+#.\-][\w'’+#.\- ]{1,40})",
        re.IGNORECASE,
    )

    # Question forms — if the utterance is (or starts as) a question about the
    # user, we never treat it as a statement that sets a preference.
    _QUESTION_RE = re.compile(
        r"^\s*(what|who|where|when|why|how|do|does|did|is|are|am|can|could|"
        r"would|should|will|tell me)\b",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> list[ExtractedPreference]:
        """Return the structured preferences stated in *text* (may be empty).

        Never raises; unknown / ambiguous input simply yields ``[]``.
        """
        raw = (text or "").strip()
        if not raw or self._QUESTION_RE.match(raw):
            return []

        found: list[ExtractedPreference] = []
        seen_keys: set[str] = set()

        def _add(key: str, value: str) -> None:
            value = self._clean_value(value)
            if not value or key in seen_keys:
                return
            seen_keys.add(key)
            found.append(ExtractedPreference(key, value))

        # Specific, high-signal categories first.
        for pat in self._NAME_PATTERNS:
            m = pat.search(raw)
            if m:
                _add(KEY_NAME, m.group(1))
                break
        for pat in self._LANGUAGE_PATTERNS:
            m = pat.search(raw)
            if m:
                _add(KEY_PREFERRED_LANGUAGE, m.group(1))
                break
        for pat in self._LOCATION_PATTERNS:
            m = pat.search(raw)
            if m:
                _add(KEY_LOCATION, m.group(1))
                break
        for pat in self._OCCUPATION_PATTERNS:
            m = pat.search(raw)
            if m:
                _add(KEY_OCCUPATION, m.group(1))
                break
        for pat in self._CODING_STYLE_PATTERNS:
            m = pat.search(raw)
            if m:
                _add(KEY_CODING_STYLE, m.group(1))
                break

        # Generic "my favourite X is Y".
        for m in self._FAVORITE_GENERIC.finditer(raw):
            noun = self._clean_value(m.group(1)).lower().replace(" ", "_")
            # Don't double-emit the language we already captured specifically.
            if noun in ("language", "programming_language", "coding_language"):
                continue
            if noun:
                _add(f"favorite_{noun}", m.group(2))

        # Last-resort generic like/prefer → a single ``likes`` value, only when
        # nothing more specific matched (keeps the profile clean).
        if not found:
            m = self._LIKE_GENERIC.search(raw)
            if m:
                _add(KEY_LIKES, m.group(1))

        return found

    # ------------------------------------------------------------------
    # Value hygiene
    # ------------------------------------------------------------------

    @classmethod
    def _clean_value(cls, value: str) -> str:
        """Normalise and validate a captured value; '' means reject."""
        v = (value or "").strip().strip("\"'“”").strip()
        # Cut at a clause boundary so "Rust because it's fast" → "Rust".
        v = re.split(r"\b(?:because|since|as|so that|but|and then|,|;|\.| - )\b", v, maxsplit=1)[0]
        v = v.strip(" .,!?-—:")
        if not v:
            return ""
        if len(v) > _MAX_VALUE_CHARS:
            return ""
        # A single very long token is almost certainly noise, not a preference
        # ("I like aaaaaaaaaaaaaaaaaaaaaa...") — real values are short or spaced.
        if " " not in v and len(v) > 30:
            return ""
        low = v.lower()
        if low in _STOP_VALUES:
            return ""
        first = low.split()[0]
        if first in _VERB_LEADS:
            return ""
        # Must contain at least one letter or digit (reject pure punctuation).
        if not re.search(r"[A-Za-z0-9]", v):
            return ""
        return v
