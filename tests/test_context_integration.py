"""Integration tests for ContextManager, IntentDetector additions, and cross-cutting fixes."""

from __future__ import annotations

import pytest

from agents.context_manager import ContextManager
from tools.intent_detector import IntentDetector
from tools.manager import ToolManager
from tools.builtins import register_all_builtins
from tools.registry import ToolRegistry


# =========================================================================
# ContextManager — pronoun resolution
# =========================================================================

class TestContextManagerPronouns:

    def test_resolves_he_to_last_person(self) -> None:
        cm = ContextManager()
        cm.update("My name is Nikhil", "Hello Nikhil!")
        enriched = cm.enrich("Where does he live?")
        assert "Nikhil" in enriched or "nikhil" in enriched.lower()
        assert "he" not in enriched.lower().split()

    def test_resolves_she_to_last_person(self) -> None:
        cm = ContextManager()
        cm.update("Her name is Priya", "Hello Priya!")
        enriched = cm.enrich("Where does she work?")
        assert "Priya" in enriched
        assert "she" not in enriched.lower().split()

    def test_resolves_it_to_last_topic(self) -> None:
        cm = ContextManager()
        cm.update("What is the weather in Bangalore", "It is sunny in Bangalore")
        enriched = cm.enrich("Is it going to rain?")
        assert "Bangalore" in enriched or "weather" in enriched.lower()

    def test_resolves_that_to_last_topic(self) -> None:
        cm = ContextManager()
        cm.update("Tell me about Python programming", "Python is a versatile language")
        enriched = cm.enrich("Tell me more about that")
        assert "Python" in enriched

    def test_no_context_returns_goal_unchanged(self) -> None:
        cm = ContextManager()
        assert cm.enrich("hello") == "hello"

    def test_resolves_they_to_plural_people(self) -> None:
        cm = ContextManager()
        cm.update("My friends are Alice and Bob", "Nice!")
        enriched = cm.enrich("Where do they work?")
        # Should resolve "they" to the people mentioned
        assert "Alice" in enriched or "Bob" in enriched

    def test_pronoun_not_resolved_when_not_in_context(self) -> None:
        cm = ContextManager()
        cm.update("What is 2+2", "4")
        enriched = cm.enrich("Where does he live?")
        # No people in context, "he" should remain
        assert "he" in enriched.lower() or enriched == "Where does he live?"


class TestContextManagerFollowUp:

    def test_all_of_it_resolves_to_last_topic(self) -> None:
        cm = ContextManager()
        cm.update("Tell me about the weather in Bangalore", "It is sunny")
        enriched = cm.enrich("Tell me all of it")
        assert "Bangalore" in enriched or "weather" in enriched

    def test_tell_me_more_resolves_to_last_topic(self) -> None:
        cm = ContextManager()
        cm.update("What is machine learning", "Machine learning is a subset of AI")
        enriched = cm.enrich("Tell me more about that")
        assert "machine learning" in enriched.lower()

    def test_elaborate_resolves_to_last_topic(self) -> None:
        cm = ContextManager()
        cm.update("Explain quantum computing", "Quantum computing uses qubits")
        enriched = cm.enrich("Elaborate")
        assert "quantum computing" in enriched.lower() or "quantum" in enriched.lower()

    def test_follow_up_without_context_unchanged(self) -> None:
        cm = ContextManager()
        enriched = cm.enrich("Tell me more about that")
        assert enriched == "Tell me more about that"

    def test_context_window_eviction(self) -> None:
        cm = ContextManager(window_size=2)
        cm.update("First topic A", "Response A")
        cm.update("Second topic B", "Response B")
        cm.update("Third topic C", "Response C")
        # Topic A should be evicted from window
        enriched = cm.enrich("Tell me more about that")
        assert "B" in enriched or "C" in enriched


class TestContextManagerSnapshot:

    def test_snapshot_after_updates(self) -> None:
        cm = ContextManager()
        cm.update("My name is Alice", "Hello Alice!")
        cm.update("What is the weather in London", "London is rainy")
        snap = cm.snapshot()
        assert snap["turn_count"] == 2
        assert "alice" in snap["last_people"]
        assert len(snap["last_entities"]) > 0


# =========================================================================
# IntentDetector — current_info routing
# =========================================================================

@pytest.fixture
def detector() -> IntentDetector:
    registry = ToolRegistry()
    register_all_builtins(registry)
    tm = ToolManager(registry=registry)
    return IntentDetector(tm)


class TestCurrentInfoDetection:

    @pytest.mark.parametrize("query", [
        "what is the weather today",
        "weather in London",
        "what is the temperature in Paris",
        "is it raining in Tokyo",
        "weather forecast for tomorrow",
    ])
    def test_weather_queries_detected(self, detector: IntentDetector, query: str) -> None:
        intent = detector.classify(query)
        assert intent.label == "current_info", f"Expected current_info for {query!r}, got {intent.label}"
        assert intent.confidence >= 0.85

    @pytest.mark.parametrize("query", [
        "what is the news today",
        "latest news headlines",
        "current news",
        "breaking news",
    ])
    def test_news_queries_detected(self, detector: IntentDetector, query: str) -> None:
        intent = detector.classify(query)
        assert intent.label == "current_info", f"Expected current_info for {query!r}, got {intent.label}"
        assert intent.confidence >= 0.85

    @pytest.mark.parametrize("query", [
        "what is the stock price of Apple",
        "share price of Microsoft",
        "market value of Tesla",
    ])
    def test_stock_queries_detected(self, detector: IntentDetector, query: str) -> None:
        intent = detector.classify(query)
        assert intent.label == "current_info", f"Expected current_info for {query!r}, got {intent.label}"

    @pytest.mark.parametrize("query", [
        "what is the current time in New York",
        "time in London",
        "current time in Tokyo",
    ])
    def test_time_queries_detected(self, detector: IntentDetector, query: str) -> None:
        intent = detector.classify(query)
        assert intent.label == "current_info", f"Expected current_info for {query!r}, got {intent.label}"

    def test_general_knowledge_not_current_info(self, detector: IntentDetector) -> None:
        """'who is' should NOT be classified as current_info."""
        intent = detector.classify("who is Nikhil")
        assert intent.label != "current_info"


class TestFollowUpDetection:

    @pytest.mark.parametrize("query", [
        "all of it",
        "tell me all of that",
        "all those details",
    ])
    def test_all_of_it_detected(self, detector: IntentDetector, query: str) -> None:
        intent = detector.classify(query)
        assert intent.label == "follow_up", f"Expected follow_up for {query!r}, got {intent.label}"
        assert intent.confidence >= 0.85

    @pytest.mark.parametrize("query", [
        "tell me more",
        "tell me more about that",
        "elaborate on that",
        "go on",
        "continue",
    ])
    def test_tell_me_more_detected(self, detector: IntentDetector, query: str) -> None:
        intent = detector.classify(query)
        assert intent.label == "follow_up", f"Expected follow_up for {query!r}, got {intent.label}"

    @pytest.mark.parametrize("query", [
        "what about him",
        "how about her",
        "what about them",
    ])
    def test_what_about_detected(self, detector: IntentDetector, query: str) -> None:
        intent = detector.classify(query)
        assert intent.label == "follow_up", f"Expected follow_up for {query!r}, got {intent.label}"


class TestToolIntentStillWorks:

    @pytest.mark.parametrize("query,tool", [
        ("what is 25 * 76", "calculator"),
        ("what's the time", "datetime"),
        ("show system info", "system_info"),
    ])
    def test_tool_intent_not_affected(self, detector: IntentDetector, query: str, tool: str) -> None:
        """Adding current_info and follow_up must not break existing tool detection."""
        result = detector.detect(query)
        assert result is not None
        assert result[0] == tool


# =========================================================================
# End-to-end: Context + Intent flow verification
# =========================================================================

class TestContextAndIntentFlow:

    def test_follow_up_respected_over_other_intents(self, detector: IntentDetector) -> None:
        """Follow-up phrases must take priority over tool or knowledge intents."""
        intent = detector.classify("tell me more")
        assert intent.label == "follow_up"

    def test_current_info_above_tool(self, detector: IntentDetector) -> None:
        """Current info must be detected before tool so weather goes to browser."""
        intent = detector.classify("what is the weather in London")
        assert intent.label == "current_info", f"Expected current_info, got {intent.label}"

    def test_unknown_remains_for_gibberish(self, detector: IntentDetector) -> None:
        intent = detector.classify("xyzabc 123")
        assert intent.label == "unknown"
        assert intent.confidence == 0.0
