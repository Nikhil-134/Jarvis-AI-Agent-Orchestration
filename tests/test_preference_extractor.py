"""Tests for PreferenceExtractor — deterministic preference extraction.

Precision matters more than recall here: a *false* preference pollutes the
durable profile, so the negative cases (questions, ambiguous statements,
pronoun/filler values) are as important as the positive ones.
"""

from __future__ import annotations

import pytest

from memory.preference_extractor import (
    KEY_CODING_STYLE,
    KEY_LIKES,
    KEY_LOCATION,
    KEY_NAME,
    KEY_OCCUPATION,
    KEY_PREFERRED_LANGUAGE,
    ExtractedPreference,
    PreferenceExtractor,
)


@pytest.fixture
def extractor() -> PreferenceExtractor:
    return PreferenceExtractor()


def _as_dict(prefs: list[ExtractedPreference]) -> dict[str, str]:
    return {p.key: p.value for p in prefs}


# =========================================================================
# Positive extraction
# =========================================================================

class TestPositiveExtraction:
    @pytest.mark.parametrize("text,key,value", [
        ("call me Boss", KEY_NAME, "Boss"),
        ("My name is Nikhil", KEY_NAME, "Nikhil"),
        ("you can call me Chief", KEY_NAME, "Chief"),
        ("my favourite language is Rust", KEY_PREFERRED_LANGUAGE, "Rust"),
        ("my favorite programming language is Python", KEY_PREFERRED_LANGUAGE, "Python"),
        ("I love coding in Go", KEY_PREFERRED_LANGUAGE, "Go"),
        ("I prefer Kotlin for my main language", KEY_PREFERRED_LANGUAGE, "Kotlin"),
        ("I live in Bangalore", KEY_LOCATION, "Bangalore"),
        ("I'm based in Berlin", KEY_LOCATION, "Berlin"),
        ("I'm from New York", KEY_LOCATION, "New York"),
        ("I work as a backend developer", KEY_OCCUPATION, "backend developer"),
        ("I'm a senior data scientist", KEY_OCCUPATION, "senior data scientist"),
        ("I am a machine learning engineer", KEY_OCCUPATION, "machine learning engineer"),
        ("I prefer tabs", KEY_CODING_STYLE, "tabs"),
        ("I like dark mode", KEY_LIKES, "dark mode"),
    ])
    def test_extracts_expected_pref(
        self, extractor: PreferenceExtractor, text: str, key: str, value: str
    ) -> None:
        result = _as_dict(extractor.extract(text))
        assert result.get(key) == value

    def test_generic_favourite(self, extractor: PreferenceExtractor) -> None:
        result = _as_dict(extractor.extract("my favourite editor is Neovim"))
        assert result.get("favorite_editor") == "Neovim"

    def test_multiple_prefs_in_one_utterance(self, extractor: PreferenceExtractor) -> None:
        result = _as_dict(
            extractor.extract("call me Boss and my favourite language is Rust")
        )
        assert result.get(KEY_NAME) == "Boss"
        assert result.get(KEY_PREFERRED_LANGUAGE) == "Rust"

    def test_case_insensitive(self, extractor: PreferenceExtractor) -> None:
        assert _as_dict(extractor.extract("CALL ME Boss")).get(KEY_NAME) == "Boss"


# =========================================================================
# Negative extraction — must NOT fire (precision guardrails)
# =========================================================================

class TestNegativeExtraction:
    @pytest.mark.parametrize("text", [
        "what is my name?",          # question about the user
        "who am I",
        "do you remember my name",
        "what do I like",
        "I am tired",                # adjective, not an occupation
        "I'm happy",
        "I like it",                 # pronoun value
        "I like that",
        "I like to think about this",  # verb-lead value
        "call me later",             # stop-word value
        "What is 2+2",               # a maths query
        "hello there",
        "explain recursion to me",
        "",
        "   ",
    ])
    def test_does_not_extract(self, extractor: PreferenceExtractor, text: str) -> None:
        assert extractor.extract(text) == []


# =========================================================================
# Value hygiene
# =========================================================================

class TestValueHygiene:
    def test_clause_boundary_is_trimmed(self, extractor: PreferenceExtractor) -> None:
        result = _as_dict(
            extractor.extract("my favourite language is Rust because it is fast")
        )
        assert result.get(KEY_PREFERRED_LANGUAGE) == "Rust"

    def test_generic_like_is_trimmed_at_clause(self, extractor: PreferenceExtractor) -> None:
        result = _as_dict(extractor.extract("I like Neovim because it is fast"))
        assert result.get(KEY_LIKES) == "Neovim"

    def test_trailing_punctuation_stripped(self, extractor: PreferenceExtractor) -> None:
        assert _as_dict(extractor.extract("call me Boss!")).get(KEY_NAME) == "Boss"

    def test_overlong_value_rejected(self, extractor: PreferenceExtractor) -> None:
        long_tail = "x" * 80
        assert extractor.extract(f"I like {long_tail}") == []

    def test_never_raises_on_weird_input(self, extractor: PreferenceExtractor) -> None:
        for weird in ["I like 🎉🎉🎉", "call me", "my name is", "((((", "I'm a "]:
            # Must not raise; may or may not extract.
            extractor.extract(weird)
