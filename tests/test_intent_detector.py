"""Tests for the IntentDetector — natural language → tool routing."""

from __future__ import annotations

import pytest

from tools.intent_detector import IntentDetector
from tools.manager import ToolManager
from tools.builtins import register_all_builtins
from tools.registry import ToolRegistry


@pytest.fixture
def detector() -> IntentDetector:
    registry = ToolRegistry()
    register_all_builtins(registry)
    tm = ToolManager(registry=registry)
    return IntentDetector(tm)


# =========================================================================
# Calculator
# =========================================================================

class TestCalculatorDetection:
    def test_what_is_expression(self, detector: IntentDetector) -> None:
        result = detector.detect("what is 25 * 76")
        assert result is not None
        assert result[0] == "calculator"
        assert result[1]["expression"] == "25 * 76"

    @pytest.mark.parametrize("query,expected_expr", [
        ("what is 2+2?", "2+2"),
        ("calculate 100 / 5", "100 / 5"),
        ("solve 45 * 32", "45 * 32"),
        ("25 * 4 = ?", "25 * 4"),
        ("what is (12 + 34) * 2?", "(12 + 34) * 2"),
    ])
    def test_various_calculator_phrasings(self, detector: IntentDetector, query: str, expected_expr: str) -> None:
        result = detector.detect(query)
        assert result is not None
        assert result[0] == "calculator"
        assert result[1]["expression"] == expected_expr

    def test_no_match_for_non_calculator(self, detector: IntentDetector) -> None:
        assert detector.detect("hello world") is None
        assert detector.detect("what is the meaning of life") is None

    # ---- Pure math expressions (new) ----

    @pytest.mark.parametrize("query", [
        "234+56",
        "(234*89)+100",
        "sqrt(81)",
        "sin(45)",
        "factorial(20)",
        "log10(100000)",
        "2^100",
        "1e9*1e9",
        "2**100",
        "gcd(12, 8)",
        "lcm(4, 6)",
        "pi * 2",
        "sin(45) + cos(45)",
        "sqrt(abs(-100))",
        "(987654321*123456789)+(999999999/37)",
        "sqrt(987654321)",
        "factorial(20)",
    ])
    def test_pure_math_expressions(self, detector: IntentDetector, query: str) -> None:
        result = detector.detect(query)
        assert result is not None, f"Failed to detect: {query}"
        assert result[0] == "calculator"

    # ---- Explanation queries are NOT routed to calculator ----

    @pytest.mark.parametrize("query", [
        "explain why 2+2=4",
        "why does 2+2 equal 4",
        "explain how this formula works",
        "how does factorial work",
        "show me how to calculate sqrt",
    ])
    def test_explanation_queries_not_routed(self, detector: IntentDetector, query: str) -> None:
        assert detector.detect(query) is None, f"Should not route explanation: {query}"

    # ---- Natural language with math extraction ----

    @pytest.mark.parametrize("query,expected_expr", [
        ("what is (234*567)", "(234*567)"),
        ("what is sqrt(81)", "sqrt(81)"),
        ("compute 2^100", "2^100"),
        ("evaluate factorial(20)", "factorial(20)"),
        ("find sin(45)", "sin(45)"),
    ])
    def test_natural_language_with_math(self, detector: IntentDetector, query: str, expected_expr: str) -> None:
        result = detector.detect(query)
        assert result is not None, f"Failed to detect: {query}"
        assert result[0] == "calculator"

    # ---- Regression (cycle 9): pow()/exp()/log2() must reach the calculator,
    #      NOT the LLM. "The LLM must NEVER calculate mathematics." ----

    @pytest.mark.parametrize("query,expected_expr", [
        ("pow(2, 10)", "pow(2, 10)"),
        ("what is pow(2, 10)", "pow(2, 10)"),
        ("exp(1)", "exp(1)"),
        ("log2(1024)", "log2(1024)"),
        ("hypot(3, 4)", "hypot(3, 4)"),
    ])
    def test_pow_and_companions_route_to_calculator(
        self, detector: IntentDetector, query: str, expected_expr: str
    ) -> None:
        result = detector.detect(query)
        assert result is not None, f"pow/companion leaked to the LLM: {query!r}"
        assert result[0] == "calculator"
        assert result[1]["expression"] == expected_expr

    @pytest.mark.parametrize("query", ["pow(2, 10)", "what is pow(2, 10)"])
    def test_pow_not_classified_as_llm_intent(self, detector: IntentDetector, query: str) -> None:
        # Before the fix these classified as unknown / knowledge_question and
        # were answered (badly) by the LLM.
        assert detector.classify(query).label == "tool"


# =========================================================================
# UUID
# =========================================================================

class TestUuidDetection:
    def test_generate_uuid(self, detector: IntentDetector) -> None:
        result = detector.detect("generate a uuid")
        assert result is not None
        assert result[0] == "uuid"

    def test_create_guid(self, detector: IntentDetector) -> None:
        result = detector.detect("create a new guid")
        assert result is not None
        assert result[0] == "uuid"

    def test_no_match(self, detector: IntentDetector) -> None:
        assert detector.detect("what is a uuid") is None


# =========================================================================
# Base64
# =========================================================================

class TestBase64Detection:
    def test_encode(self, detector: IntentDetector) -> None:
        result = detector.detect("encode hello world to base64")
        assert result is not None
        assert result[0] == "base64"
        assert result[1]["operation"] == "encode"
        assert result[1]["data"] == "hello world"

    def test_decode(self, detector: IntentDetector) -> None:
        result = detector.detect("decode aGVsbG8= from base64")
        assert result is not None
        assert result[0] == "base64"
        assert result[1]["operation"] == "decode"
        assert result[1]["data"] == "aGVsbG8="

    def test_no_match(self, detector: IntentDetector) -> None:
        assert detector.detect("what is base64") is None


# =========================================================================
# Hash
# =========================================================================

class TestHashDetection:
    def test_sha256_hash(self, detector: IntentDetector) -> None:
        result = detector.detect("sha256 hash of hello")
        assert result is not None
        assert result[0] == "hash"
        assert result[1]["algorithm"] == "sha256"
        assert result[1]["data"] == "hello"

    def test_md5_hash(self, detector: IntentDetector) -> None:
        result = detector.detect("generate md5 hash of test data")
        assert result is not None
        assert result[0] == "hash"
        assert result[1]["algorithm"] == "md5"
        assert result[1]["data"] == "test data"

    def test_no_match(self, detector: IntentDetector) -> None:
        assert detector.detect("what is a hash") is None


# =========================================================================
# JSON
# =========================================================================

class TestJsonDetection:
    def test_pretty_print(self, detector: IntentDetector) -> None:
        result = detector.detect("pretty print json {\"a\":1}")
        assert result is not None
        assert result[0] == "json"
        assert result[1]["operation"] == "pretty_print"

    def test_validate(self, detector: IntentDetector) -> None:
        result = detector.detect("validate this json {\"a\":1}")
        assert result is not None
        assert result[0] == "json"
        assert result[1]["operation"] == "validate"

    def test_no_match(self, detector: IntentDetector) -> None:
        assert detector.detect("what is json") is None


# =========================================================================
# Datetime
# =========================================================================

class TestDatetimeDetection:
    @pytest.mark.parametrize("query", [
        "what is the current date and time",
        "what's the time",
        "current date",
        "what is today's date",
        "tell me the current datetime",
        "what is the date",
        "time",
        "today",
    ])
    def test_datetime_matches(self, detector: IntentDetector, query: str) -> None:
        result = detector.detect(query)
        assert result is not None
        assert result[0] == "datetime"

    def test_no_match(self, detector: IntentDetector) -> None:
        assert detector.detect("set an alarm for 5pm") is None


# =========================================================================
# System Info
# =========================================================================

class TestSystemInfoDetection:
    @pytest.mark.parametrize("query", [
        "system info",
        "show system information",
        "what is my os",
        "cpu info",
        "system details",
        "what is my processor",
        "memory",
        "display system info",
    ])
    def test_system_info_matches(self, detector: IntentDetector, query: str) -> None:
        result = detector.detect(query)
        assert result is not None
        assert result[0] == "system_info"

    def test_no_match(self, detector: IntentDetector) -> None:
        assert detector.detect("install python") is None


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    def test_empty_string(self, detector: IntentDetector) -> None:
        assert detector.detect("") is None

    def test_whitespace(self, detector: IntentDetector) -> None:
        assert detector.detect("   ") is None

    def test_special_characters(self, detector: IntentDetector) -> None:
        assert detector.detect("@#$%^&*()") is None

    def test_case_insensitivity(self, detector: IntentDetector) -> None:
        result = detector.detect("WHAT IS 25 * 4")
        assert result is not None
        assert result[0] == "calculator"

    def test_disabled_tool_not_detected(self, detector: IntentDetector) -> None:
        detector._tm.disable_tool("datetime")
        assert detector.detect("what is the time") is None

    def test_long_sentence_no_match(self, detector: IntentDetector) -> None:
        assert detector.detect("can you please help me figure out how to write a python script") is None


# =========================================================================
# Priority-ordered classify()
# =========================================================================

class TestClassify:
    """Tests for IntentDetector.classify() — priority-ordered intent labels."""

    @pytest.mark.parametrize("query", [
        "hi", "hello", "hey", "greetings", "howdy",
        "good morning", "good afternoon", "good evening", "good day",
        "hi jarvis", "hello jarvis", "hey jarvis",
        "hi there", "hello there", "hey there",
    ])
    def test_classify_greeting(self, detector: IntentDetector, query: str) -> None:
        assert detector.classify(query) == "greeting", f"Expected greeting for '{query}'"

    @pytest.mark.parametrize("query", [
        "thanks", "thank you", "cheers", "thanks jarvis",
        "you're welcome", "no problem",
        "bye", "goodbye", "good bye", "see you",
        "how are you", "how's it going", "what's up", "sup",
        "nice", "nice to meet you",
        "take care", "have a good day",
    ])
    def test_classify_conversation(self, detector: IntentDetector, query: str) -> None:
        assert detector.classify(query) == "conversation", f"Expected conversation for '{query}'"

    @pytest.mark.parametrize("query", [
        "what is 25 * 76",
        "generate a uuid",
        "encode hello world to base64",
        "sha256 hash of hello",
        "pretty print json {}",
        "what is the current date and time",
        "system info",
    ])
    def test_classify_tool(self, detector: IntentDetector, query: str) -> None:
        assert detector.classify(query) == "tool", f"Expected tool for '{query}'"

    @pytest.mark.parametrize("query", [
        "",
        "   ",
        "search for AI research papers",
    ])
    def test_classify_unknown(self, detector: IntentDetector, query: str) -> None:
        assert detector.classify(query) == "unknown", f"Expected unknown for '{query}'"

    # ── Knowledge question tests ──────────────────────────────────────

    @pytest.mark.parametrize("query,expected_conf", [
        ("who is Purna?", 0.85),
        ("what is AI", 0.80),
        ("what is the meaning of life", 0.80),
        ("explain recursion", 0.85),
        ("tell me about neural networks", 0.80),
        ("how does gravity work", 0.75),
        ("why is the sky blue", 0.80),
        ("where is Paris", 0.75),
        ("when was Python invented", 0.75),
        ("define photosynthesis", 0.85),
        ("tell me a joke", 0.90),
    ])
    def test_classify_knowledge_question(self, detector: IntentDetector, query: str, expected_conf: float) -> None:
        result = detector.classify(query)
        assert result.label == "knowledge_question", f"Expected knowledge_question for '{query}', got {result.label}"
        assert result.confidence == expected_conf, f"Expected confidence {expected_conf} for '{query}', got {result.confidence}"

    def test_classify_greeting_takes_priority_over_tool(self, detector: IntentDetector) -> None:
        """A greeting phrase should be classified as greeting even if it could
        also match a tool pattern (e.g. 'hi' should not match calculator)."""
        assert detector.classify("hi") == "greeting"
        assert detector.classify("hello") == "greeting"

    @pytest.mark.parametrize("query", [
        "scan my computer",
        "check for vulnerabilities",
        "malware scan",
        "exploit detection",
        "run a pentest",
        "security audit",
        "check firewall",
        "scan ports",
        "network scan",
    ])
    def test_classify_security(self, detector: IntentDetector, query: str) -> None:
        result = detector.classify(query)
        assert result.label == "security", f"Expected security for '{query}', got {result.label}"
        assert result.confidence == 0.99

    @pytest.mark.parametrize("query", [
        "deploy my docker container",
        "kubernetes cluster setup",
        "ci/cd pipeline",
        "terraform configuration",
        "ansible playbook",
        "build pipeline",
        "deployment to production",
    ])
    def test_classify_devops(self, detector: IntentDetector, query: str) -> None:
        result = detector.classify(query)
        assert result.label == "devops", f"Expected devops for '{query}', got {result.label}"
        assert result.confidence == 0.99

    @pytest.mark.parametrize("query", [
        "open chrome",
        "launch browser",
        "search the web",
        "open https://example.com",
    ])
    def test_classify_browser(self, detector: IntentDetector, query: str) -> None:
        result = detector.classify(query)
        assert result.label == "browser", f"Expected browser for '{query}', got {result.label}"
        assert result.confidence == 0.90

    @pytest.mark.parametrize("query,expected", [
        ("generate python code", "coding"),
        ("write a function to sort", "coding"),
        ("implement a binary search", "coding"),
    ])
    def test_classify_coding(self, detector: IntentDetector, query: str, expected: str) -> None:
        result = detector.classify(query)
        assert result.label == expected, f"Expected {expected} for '{query}', got {result.label}"

    def test_classify_screenshot_is_desktop(self, detector: IntentDetector) -> None:
        result = detector.classify("take a screenshot")
        assert result.label == "desktop", f"Got {result.label}"
        assert result.confidence == 0.85

    def test_classify_code_request_is_coding(self, detector: IntentDetector) -> None:
        """Code generation requests are classified as coding, not unknown."""
        result = detector.classify("generate python code for fibonacci")
        assert result.label == "coding", f"Got {result.label}"

    # ── IntentClassification API tests ────────────────────────────────

    def test_classification_str_equality(self, detector: IntentDetector) -> None:
        """IntentClassification supports == comparison against strings."""
        assert detector.classify("hello") == "greeting"
        assert not (detector.classify("hello") == "unknown")
        assert detector.classify("hello") != "unknown"

    def test_classification_label_and_confidence(self, detector: IntentDetector) -> None:
        result = detector.classify("who is purna")
        assert result.label == "knowledge_question"
        assert result.confidence == 0.85

    def test_classification_repr(self, detector: IntentDetector) -> None:
        result = detector.classify("hello")
        assert "IntentClassification" in repr(result)
        assert "greeting" in repr(result)

    def test_classify_disabled_tool_falls_to_knowledge(self, detector: IntentDetector) -> None:
        """When a tool is disabled, the query should fall through to knowledge
        question classification rather than remaining unknown."""
        detector._tm.disable_tool("datetime")
        result = detector.classify("what is the time")
        # Falls through to knowledge_question since datetime tool is disabled
        assert result.label == "knowledge_question", f"Got {result.label}"
        assert result.confidence == 0.80
        assert detector.classify("hi") == "greeting"  # still works
