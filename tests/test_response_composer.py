"""Tests for the ResponseComposer service — merging and formatting agent outputs."""

from __future__ import annotations

from collections.abc import AsyncIterable

import pytest

from agents import AgentResult, ResponseComposer
from llm import BaseLLMProvider, LLMConfig


class StaticProvider(BaseLLMProvider):
    """Provider that returns a fixed response for merge testing."""

    def __init__(self, content: str = "MERGED: all results combined") -> None:
        super().__init__(LLMConfig(provider="static", model="static"))
        self._content = content

    @property
    def name(self) -> str:
        return "static"

    async def _generate_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> LLMResponse:
        from llm.interfaces import LLMResponse
        return LLMResponse(content=self._content)

    async def _stream_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> AsyncIterable[str]:
        yield self._content


@pytest.mark.asyncio
class TestResponseComposer:
    async def test_empty_results_returns_fallback(self) -> None:
        composer = ResponseComposer()
        result = await composer.merge("hello", [])
        assert "How can I assist you" in result

    async def test_all_failed_formats_errors(self) -> None:
        composer = ResponseComposer()
        results = [
            AgentResult(
                agent_name="friday", task_id="1", success=False,
                message="Failed", data={"error": "timeout"},
            ),
        ]
        result = await composer.merge("do something", results)
        assert "couldn't complete" in result.lower()
        assert "friday" in result
        assert "timeout" in result

    async def test_single_successful_result_extracts_response(self) -> None:
        composer = ResponseComposer()
        results = [
            AgentResult(
                agent_name="friday", task_id="1", success=True,
                message="ok", data={"response": "Here is the info."},
            ),
        ]
        result = await composer.merge("research AI", results)
        assert result == "Here is the info."

    async def test_single_result_falls_back_to_output(self) -> None:
        composer = ResponseComposer()
        results = [
            AgentResult(
                agent_name="tool", task_id="1", success=True,
                message="ok", data={"output": "tool output"},
            ),
        ]
        result = await composer.merge("run tool", results)
        assert result == "tool output"

    async def test_single_result_falls_back_to_message(self) -> None:
        composer = ResponseComposer()
        results = [
            AgentResult(
                agent_name="agent", task_id="1", success=True,
                message="Operation completed successfully.",
            ),
        ]
        result = await composer.merge("do thing", results)
        assert result == "Operation completed successfully."

    async def test_single_result_empty_data_returns_default(self) -> None:
        composer = ResponseComposer()
        results = [
            AgentResult(
                agent_name="agent", task_id="1", success=True,
                message="", data={},
            ),
        ]
        result = await composer.merge("do thing", results)
        assert result == "Task completed."

    async def test_simple_merge_multiple_results(self) -> None:
        composer = ResponseComposer()
        results = [
            AgentResult(
                agent_name="friday", task_id="1", success=True,
                message="ok", data={"response": "Research done."},
            ),
            AgentResult(
                agent_name="veronica", task_id="2", success=True,
                message="ok", data={"response": "Code generated."},
            ),
        ]
        result = await composer.merge("build feature", results)
        assert "Friday" in result or "friday" in result
        assert "Research done" in result
        assert "Veronica" in result or "veronica" in result
        assert "Code generated" in result

    async def test_all_failed_no_error_details(self) -> None:
        composer = ResponseComposer()
        results = [
            AgentResult(
                agent_name="agent", task_id="1", success=False,
                message="", data={},
            ),
        ]
        result = await composer.merge("task", results)
        assert "issue" in result

    async def test_llm_merge_with_mock_provider(self) -> None:
        provider = StaticProvider("Combined response from LLM.")
        composer = ResponseComposer(llm_provider=provider)
        results = [
            AgentResult(
                agent_name="friday", task_id="1", success=True,
                message="ok", data={"response": "Research A"},
            ),
            AgentResult(
                agent_name="veronica", task_id="2", success=True,
                message="ok", data={"response": "Code B"},
            ),
        ]
        result = await composer.merge("build feature", results)
        assert "Combined response" in result

    async def test_llm_merge_fallback_on_failure(self) -> None:
        class FailingProvider(BaseLLMProvider):
            def __init__(self) -> None:
                super().__init__(LLMConfig(provider="fail", model="fail"))

            @property
            def name(self) -> str:
                return "fail"

            async def _generate_once(self, prompt, system_prompt=None, tools=None):
                msg = "LLM error"
                raise RuntimeError(msg)

            async def _stream_once(self, prompt, system_prompt=None, tools=None):
                msg = "LLM error"
                raise RuntimeError(msg)

        composer = ResponseComposer(llm_provider=FailingProvider())
        results = [
            AgentResult(
                agent_name="friday", task_id="1", success=True,
                message="ok", data={"response": "Fallback result"},
            ),
        ]
        result = await composer.merge("test", results)
        assert "Fallback result" in result


@pytest.mark.asyncio
async def test_response_composer_integration_with_planner() -> None:
    """Verify ResponseComposer works end-to-end with a planner-like result."""
    composer = ResponseComposer()

    result = await composer.merge(
        "test goal",
        [
            AgentResult(
                agent_name="planner", task_id="p1", success=True,
                message="Planning completed.",
                data={"response": "This is the final response."},
            ),
        ],
    )
    assert result == "This is the final response."
