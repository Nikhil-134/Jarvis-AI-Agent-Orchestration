"""Integration tests for the Runtime Layer pipeline components."""

from __future__ import annotations

import pytest

from runtime.personality_engine import PersonalityEngine
from runtime.intent_engine import IntentEngine, Intent
from runtime.fallback_engine import FallbackEngine
from runtime.response_formatter import ResponseFormatter
from runtime.response_synthesizer import ResponseSynthesizer
from runtime.llm_guard import LLMGuard, GuardConfig, RetryState
from runtime.context_manager import ContextManager
from runtime.routing_engine import RoutingEngine
from llm import BaseLLMProvider, LLMConfig
from llm.interfaces import LLMResponse


# =========================================================================
# Personality Engine
# =========================================================================

class TestPersonalityEngine:

    def setup_method(self) -> None:
        self.engine = PersonalityEngine()

    def test_greeting_returns_response(self) -> None:
        result = self.engine.process("Hello")
        assert result is not None
        assert len(result) > 5

    def test_thanks_returns_response(self) -> None:
        result = self.engine.process("Thank you")
        assert result is not None
        assert any(w in result.lower() for w in ("welcome", "help", "anytime", "glad", "pleasure", "happy"))

    def test_goodbye_returns_response(self) -> None:
        result = self.engine.process("Goodbye")
        assert result is not None
        assert any(w in result.lower() for w in ("goodbye", "bye", "later", "next time"))

    def test_joke_request_returns_joke(self) -> None:
        result = self.engine.process("Tell me a joke")
        assert result is not None
        assert len(result) > 20

    def test_how_are_you_returns_response(self) -> None:
        result = self.engine.process("How are you?")
        assert result is not None
        assert any(w in result.lower() for w in ("great", "operational", "well", "doing"))

    def test_morning_greeting(self) -> None:
        result = self.engine.process("Good morning")
        assert result is not None
        assert "morning" in result.lower()

    def test_evening_greeting(self) -> None:
        result = self.engine.process("Good evening")
        assert result is not None
        assert "evening" in result.lower()

    def test_non_conversation_returns_none(self) -> None:
        result = self.engine.process("What is the weather in Bangalore?")
        assert result is None

    def test_code_request_returns_none(self) -> None:
        result = self.engine.process("Write a Python function to sort a list")
        assert result is None

    def test_insult_returns_professional(self) -> None:
        result = self.engine.process("You are useless")
        assert result is not None
        assert any(w in result.lower() for w in ("sorry", "frustrated", "respect", "focus", "problem"))

    def test_compliment_returns_appreciation(self) -> None:
        result = self.engine.process("You are great!")
        assert result is not None
        assert any(w in result.lower() for w in ("thank", "appreciate", "kind"))

    def test_wake_up(self) -> None:
        result = self.engine.process("Wake up Jarvis")
        assert result is not None
        assert "online" in result.lower() or "awake" in result.lower() or "ready" in result.lower()


# =========================================================================
# Intent Engine
# =========================================================================

class TestIntentEngine:

    def test_requires_browser_for_weather(self) -> None:
        from tools.intent_detector import IntentDetector
        from tools.manager import ToolManager
        from tools.registry import ToolRegistry
        from tools.builtins import register_all_builtins
        registry = ToolRegistry()
        register_all_builtins(registry)
        detector = IntentDetector(ToolManager(registry=registry))
        engine = IntentEngine(detector)

        result = engine.classify("What is the weather in London?")
        assert result.requires_browser
        assert not result.requires_conversation
        assert not result.requires_vision

    def test_requires_knowledge_for_questions(self) -> None:
        from tools.intent_detector import IntentDetector
        from tools.manager import ToolManager
        from tools.registry import ToolRegistry
        from tools.builtins import register_all_builtins
        registry = ToolRegistry()
        register_all_builtins(registry)
        detector = IntentDetector(ToolManager(registry=registry))
        engine = IntentEngine(detector)

        result = engine.classify("Who is Nikhil?")
        assert result.requires_knowledge
        assert result.primary.label == "knowledge_question"

    def test_requires_tool_for_calculator(self) -> None:
        from tools.intent_detector import IntentDetector
        from tools.manager import ToolManager
        from tools.registry import ToolRegistry
        from tools.builtins import register_all_builtins
        registry = ToolRegistry()
        register_all_builtins(registry)
        detector = IntentDetector(ToolManager(registry=registry))
        engine = IntentEngine(detector)

        result = engine.classify("What is 2 + 2?")
        assert result.requires_tool
        assert result.primary.label == "tool"

    def test_handles_no_detector_gracefully(self) -> None:
        engine = IntentEngine(None)
        result = engine.classify("hello")
        assert result.primary.label == "plan"
        assert result.requires_planning

    def test_compound_goal_does_not_crash(self) -> None:
        # Regression: the compound splitter has capturing groups, so re.split
        # interleaves None for the optional group — this used to raise
        # AttributeError: 'NoneType' object has no attribute 'strip'.
        from tools.intent_detector import IntentDetector
        from tools.manager import ToolManager
        from tools.registry import ToolRegistry
        from tools.builtins import register_all_builtins
        registry = ToolRegistry()
        register_all_builtins(registry)
        detector = IntentDetector(ToolManager(registry=registry))
        engine = IntentEngine(detector)

        result = engine.classify("calculate 23*(18+7) and then explain the result")
        assert result.primary is not None
        assert len(result.secondary) >= 1  # compound → populated secondary


# =========================================================================
# Fallback Engine
# =========================================================================

class TestFallbackEngine:

    def setup_method(self) -> None:
        self.engine = FallbackEngine()

    def test_llm_error_returns_graceful_message(self) -> None:
        result = self.engine.on_llm_error("llm_timeout")
        assert result.success
        assert "language model" in result.data["response"].lower()

    def test_tool_error_includes_tool_name(self) -> None:
        result = self.engine.on_tool_error(tool_name="calculator")
        assert result.success
        assert "calculator" in result.data["response"]

    def test_exception_never_shows_traceback(self) -> None:
        result = self.engine.on_exception(ValueError("test error"))
        assert result.success
        response = result.data["response"]
        assert "Traceback" not in response
        assert "ValueError" not in response
        assert "issue" in response.lower()

    def test_wrap_failed_result(self) -> None:
        from agents.contracts import AgentResult
        failed = AgentResult(agent_name="test", task_id="", success=False, message="Something broke")
        wrapped = self.engine.wrap(failed, error_type="llm_timeout")
        assert wrapped.success
        assert "language model" in wrapped.data["response"].lower()


# =========================================================================
# Response Formatter
# =========================================================================

class TestResponseFormatter:

    def setup_method(self) -> None:
        self.formatter = ResponseFormatter()

    def test_strips_tool_call_json(self) -> None:
        dirty = '{"name":"calculator","arguments":{"expression":"2+2"}} returned 4'
        cleaned = self.formatter.format(dirty)
        assert '{"name"' not in cleaned
        assert "returned 4" in cleaned

    def test_strips_internal_prompts(self) -> None:
        dirty = "You are Jarvis, a helpful AI assistant. What is 2+2?"
        cleaned = self.formatter.format(dirty)
        assert "You are Jarvis" not in cleaned
        assert "What is 2+2" in cleaned

    def test_normalizes_whitespace(self) -> None:
        dirty = "Line 1\n\n\n\nLine 2\n   \nLine 3"
        cleaned = self.formatter.format(dirty)
        lines = cleaned.split("\n")
        assert len(lines) <= 5

    def test_format_code_block(self) -> None:
        result = self.formatter.format_code_block("print('hello')", "python")
        assert "```python" in result
        assert "print('hello')" in result
        assert "```" in result

    def test_format_list(self) -> None:
        result = self.formatter.format_list(["Item 1", "Item 2", "Item 3"])
        assert "- Item 1" in result
        assert "- Item 3" in result

    def test_format_table(self) -> None:
        result = self.formatter.format_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        assert "|Name|Age|" in result
        assert "|Alice|30|" in result
        assert "|Bob|25|" in result

    def test_is_response_clean_rejects_internal_artifacts(self) -> None:
        assert not self.formatter.is_response_clean("You are Jarvis, an AI")
        assert not self.formatter.is_response_clean("AgentResult(success=True)")
        assert not self.formatter.is_response_clean("Tool execution completed")
        assert self.formatter.is_response_clean('{"name":"test","arguments":{}}')
        assert self.formatter.is_response_clean("Hello! How can I help?")


# =========================================================================
# Context Manager
# =========================================================================

class TestRuntimeContextManager:

    def setup_method(self) -> None:
        self.cm = ContextManager()

    def test_multi_session_isolation(self) -> None:
        self.cm.update_session("user_a", "My name is Alice", "Hello Alice!")
        self.cm.update_session("user_b", "My name is Bob", "Hello Bob!")

        enriched_a = self.cm.enrich("user_a", "Where does she work?")
        enriched_b = self.cm.enrich("user_b", "Where does he live?")

        assert "Alice" in enriched_a
        assert "Bob" in enriched_b

    def test_tool_context_tracking(self) -> None:
        self.cm.update_session(
            "default", "Calculate 2+2", "4",
            tool_name="calculator", tool_args={"expression": "2+2"}, tool_result="4", tool_success=True,
        )
        session = self.cm._sessions["default"]
        assert session.last_tool.tool_name == "calculator"
        assert session.last_tool.result == "4"

    def test_browser_context_tracking(self) -> None:
        self.cm.update_session(
            "default", "Search for AI news", "Results found",
            browser_url="https://google.com", browser_title="Google Search",
        )
        session = self.cm._sessions["default"]
        assert session.last_browser.url == "https://google.com"

    def test_file_context_tracking(self) -> None:
        self.cm.update_session(
            "default", "Read config.py", "File contents",
            file_path="/path/to/config.py", file_operation="read",
        )
        session = self.cm._sessions["default"]
        assert session.last_file.path == "/path/to/config.py"

    def test_context_summary(self) -> None:
        self.cm.update_session("default", "What is the weather in London", "London is rainy")
        summary = self.cm.get_context_summary("default")
        assert "Conversation turns" in summary
        assert "London" in summary or "weather" in summary

    def test_clear_session(self) -> None:
        self.cm.update_session("test", "Hello", "Hi")
        self.cm.clear_session("test")
        assert "test" not in self.cm._sessions

    def test_clear_all(self) -> None:
        self.cm.update_session("a", "Hello", "Hi")
        self.cm.update_session("b", "Hello", "Hi")
        self.cm.clear_all()
        assert len(self.cm._sessions) == 0


# =========================================================================
# LLM Guard
# =========================================================================

class FakeProvider(BaseLLMProvider):
    """A test LLM provider that returns a fixed response."""

    def __init__(self, fail_count: int = 0) -> None:
        super().__init__(LLMConfig(provider="fake", model="fake"))
        self._call_count = 0
        self._fail_count = fail_count

    @property
    def name(self) -> str:
        return "fake"

    async def _generate_once(self, prompt: str, system_prompt: str | None, tools=None) -> LLMResponse:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            from llm.errors import LLMTimeoutError
            raise LLMTimeoutError("Simulated timeout")
        return LLMResponse(content=f"Response to: {prompt[:50]}")

    async def _stream_once(self, prompt: str, system_prompt: str | None, tools=None) -> None:
        yield "streaming "


class TestLLMGuard:

    @pytest.mark.asyncio
    async def test_successful_call_returns_response(self) -> None:
        provider = FakeProvider()
        guard = LLMGuard(provider)
        response = await guard.generate("Hello")
        assert response.content.startswith("Response to:")

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self) -> None:
        provider = FakeProvider(fail_count=1)
        guard = LLMGuard(provider, config=GuardConfig(max_retries=2, retry_backoff_seconds=0.01))
        response = await guard.generate("Hello")
        assert response.content.startswith("Response to:")
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_graceful_message_on_all_failures(self) -> None:
        provider = FakeProvider(fail_count=3)
        guard = LLMGuard(provider, config=GuardConfig(max_retries=0, retry_backoff_seconds=0.01))
        response = await guard.generate("Hello")
        assert "trouble" in response.content.lower() or "issue" in response.content.lower()

    @pytest.mark.asyncio
    async def test_no_provider_returns_graceful_message(self) -> None:
        guard = LLMGuard(None)
        response = await guard.generate("Hello")
        assert "No LLM provider configured" in response.content

    def test_is_available(self) -> None:
        assert LLMGuard(FakeProvider()).is_available
        assert not LLMGuard(None).is_available


# =========================================================================
# ResponseSynthesizer Tests
# =========================================================================

class TestResponseSynthesizer:

    def setup_method(self) -> None:
        self.synth = ResponseSynthesizer()

    def test_calculator_returns_answer(self) -> None:
        result = self.synth.synthesize("calculator", "24")
        assert "answer" in result.lower()
        assert "24" in result

    def test_calculator_decimal(self) -> None:
        result = self.synth.synthesize("calculator", "3.14")
        assert "answer" in result.lower()
        assert "3.14" in result

    def test_calculator_negative(self) -> None:
        result = self.synth.synthesize("calculator", "-5")
        assert "answer" in result.lower()
        assert "-5" in result

    def test_calculator_large_integer_kept_exact(self) -> None:
        # Regression (cycle 9): a big-integer result (2**5000) must not be
        # routed through float() — that overflowed to inf and raised on
        # int(inf), losing the "answer" formatting.
        big = str(2 ** 5000)
        result = self.synth.synthesize("calculator", big)
        assert "answer" in result.lower()
        assert big in result  # kept exact, in full

    def test_datetime_returns_formatted(self) -> None:
        result = self.synth.synthesize("datetime", "2026-07-07 10:30:00")
        assert "date and time" in result.lower()
        assert "2026" in result

    def test_uuid_returns_value(self) -> None:
        result = self.synth.synthesize("uuid", "550e8400-e29b-41d4-a716-446655440000")
        assert "550e8400" in result

    def test_clipboard_returns_prefixed(self) -> None:
        result = self.synth.synthesize("clipboard", "some text")
        assert "clipboard" in result.lower()
        assert "some text" in result

    def test_screenshot_returns_confirmation(self) -> None:
        result = self.synth.synthesize("screenshot", "image data")
        assert "captured" in result.lower()

    def test_notification_returns_confirmation(self) -> None:
        result = self.synth.synthesize("notification", "sent")
        assert "sent" in result.lower()

    def test_shell_returns_code_block(self) -> None:
        result = self.synth.synthesize("shell", "ls -la")
        assert "```" in result
        assert "ls -la" in result

    def test_system_info_preserves_output(self) -> None:
        result = self.synth.synthesize("system_info", "OS: Windows")
        assert "OS: Windows" in result or "Windows" in result

    def test_hash_returns_value(self) -> None:
        result = self.synth.synthesize("hash", "abc123def456")
        assert "abc123def456" in result

    def test_base64_returns_value(self) -> None:
        result = self.synth.synthesize("base64", "SGVsbG8=")
        assert "SGVsbG8=" in result

    def test_json_returns_code_block(self) -> None:
        result = self.synth.synthesize("json", '{"key": "value"}')
        assert "```json" in result.lower() or "```" in result

    def test_text_returns_output(self) -> None:
        result = self.synth.synthesize("text", "Some text output")
        assert "Some text output" in result

    def test_file_system_returns_output(self) -> None:
        result = self.synth.synthesize("file_system", "File created: test.txt")
        assert "File created" in result

    def test_generic_preserves_output(self) -> None:
        result = self.synth.synthesize("unknown_tool", "raw output data")
        assert "raw output data" in result

    def test_empty_output_returns_message(self) -> None:
        result = self.synth.synthesize("calculator", "")
        assert "no output" in result.lower() or "completed" in result.lower()

    def test_tool_aliases_work(self) -> None:
        result = self.synth.synthesize("calc", "42")
        assert "answer" in result.lower()
        assert "42" in result

    def test_json_dict_output_extraction(self) -> None:
        result = self.synth.synthesize("calculator", '{"output": "42", "success": true}')
        assert "42" in result

    def test_json_dict_result_fallback(self) -> None:
        result = self.synth.synthesize("calculator", '{"result": "success", "value": "100"}')
        assert "success" in result

    def test_weather_json_output(self) -> None:
        weather_json = '{"temperature": 25, "humidity": 60, "wind": "10 km/h", "forecast": "Sunny"}'
        result = self.synth.synthesize("weather", weather_json)
        assert "25" in result or "temperature" in result.lower() or "humidity" in result.lower()

    def test_browser_json_output(self) -> None:
        browser_json = '{"output": "Search results for Python decorators:\\n- @decorator syntax\\n- functools.wraps"}'
        result = self.synth.synthesize("browser", browser_json)
        assert "decorator" in result


class TestRoutingEngineToolDetection:

    def test_detect_calculator(self) -> None:
        tool_name, args = RoutingEngine._detect_tool("Calculate 234+(5*7)")
        assert tool_name == "calculator"
        assert "expression" in args
        assert "234+(5*7)" in args["expression"] or "234+(5" in args["expression"]

    def test_detect_calculator_what_is(self) -> None:
        tool_name, args = RoutingEngine._detect_tool("What is 25 * 76")
        assert tool_name == "calculator"
        assert "expression" in args

    def test_detect_search(self) -> None:
        tool_name, args = RoutingEngine._detect_tool("Search Python decorators")
        assert tool_name == "search"
        assert "query" in args

    def test_detect_weather(self) -> None:
        tool_name, args = RoutingEngine._detect_tool("Weather Bangalore")
        assert tool_name == "weather"
        assert "location" in args

    def test_detect_datetime(self) -> None:
        tool_name, args = RoutingEngine._detect_tool("What is the current date and time")
        assert tool_name == "datetime"

    def test_detect_direct_calculate(self) -> None:
        tool_name, args = RoutingEngine._detect_tool("Calculate 2+2")
        assert tool_name == "calculator"
        assert args.get("expression") == "2+2"

    def test_detect_no_tool(self) -> None:
        tool_name, args = RoutingEngine._detect_tool("Tell me about Python")
        assert tool_name == ""
        assert args == {}
