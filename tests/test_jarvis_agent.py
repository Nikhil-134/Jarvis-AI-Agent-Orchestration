"""Tests for JarvisPrimeAgent — single entry point orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

import pytest

from agents import AgentResult, AgentTask, JarvisPrimeAgent, ResponseComposer
from agents.conversation_manager import ConversationManager
from agents.planner import PlannerAgent
from llm import BaseLLMProvider, LLMConfig, PromptManager
from orchestrator.workflow import WorkflowEngine, WorkflowPlan, WorkflowStep


class EchoPlannerProvider(BaseLLMProvider):
    """Provider that echoes back the prompt for testing."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(provider="echo", model="echo"))

    @property
    def name(self) -> str:
        return "echo"

    async def _generate_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> LLMResponse:
        from llm.interfaces import LLMResponse
        return LLMResponse(content=f"Response: {prompt[:50]}")

    async def _stream_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> AsyncIterable[str]:
        yield f"Response: {prompt[:50]}"


def _mock_route(task: AgentTask) -> AgentResult:
    return AgentResult(
        agent_name="mock_agent",
        task_id=task.task_id,
        success=True,
        message=f"Handled {task.task_type}",
        data={"response": f"Result for {task.task_type}"},
    )


@pytest.mark.asyncio
async def test_jarvis_agent_handles_jarvis_process() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="jarvis.process", payload={"goal": "hello", "task_type": "plan"})
    )
    assert result.success
    assert result.agent_name == "jarvis"
    assert "response" in result.data


@pytest.mark.asyncio
async def test_jarvis_agent_handles_plan_task() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="plan", payload={"goal": "hello"})
    )
    assert result.success
    assert "response" in result.data


@pytest.mark.asyncio
async def test_jarvis_agent_returns_fallback_for_unknown_task_type() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="unknown", payload={"goal": "test"})
    )
    assert not result.success
    assert "cannot handle" in result.message


@pytest.mark.asyncio
async def test_jarvis_agent_conversation_turns_increment() -> None:
    agent = JarvisPrimeAgent()
    assert agent.conversation_manager.turn_count == 0

    await agent.handle(AgentTask(task_type="jarvis.process", payload={"goal": "hello"}))
    assert agent.conversation_manager.turn_count == 1

    await agent.handle(AgentTask(task_type="jarvis.process", payload={"goal": "how are you?"}))
    assert agent.conversation_manager.turn_count == 2


@pytest.mark.asyncio
async def test_jarvis_agent_workflow_engine_integration() -> None:
    """Verify multi-agent orchestration through the workflow engine."""
    route_fn = _mock_route
    workflow_engine = WorkflowEngine(route_fn=route_fn)
    agent = JarvisPrimeAgent(workflow_engine=workflow_engine)

    task_types = ["research", "code.generate"]
    response = await agent._execute_and_merge("build a feature", task_types)
    assert response
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_jarvis_agent_rule_based_decompose_research() -> None:
    agent = JarvisPrimeAgent()
    types = await agent._rule_based_decompose("search for AI papers")
    assert "information.retrieve" in types


@pytest.mark.asyncio
async def test_jarvis_agent_rule_based_decompose_code() -> None:
    agent = JarvisPrimeAgent()
    types = await agent._rule_based_decompose("generate python code for fibonacci")
    assert "code.generate" in types


@pytest.mark.asyncio
async def test_jarvis_agent_rule_based_decompose_empty_for_greeting() -> None:
    agent = JarvisPrimeAgent()
    types = await agent._rule_based_decompose("hello, how are you?")
    assert types == []


@pytest.mark.asyncio
async def test_jarvis_agent_response_composer_integration() -> None:
    """Verify the response_composer is properly wired."""
    agent = JarvisPrimeAgent()
    assert agent.response_composer is not None
    assert isinstance(agent.response_composer, ResponseComposer)


@pytest.mark.asyncio
async def test_jarvis_agent_planner_agent_integration() -> None:
    """Verify the planner_agent is properly wired."""
    agent = JarvisPrimeAgent()
    assert agent.planner_agent is not None
    assert isinstance(agent.planner_agent, PlannerAgent)


@pytest.mark.asyncio
async def test_jarvis_agent_health_check() -> None:
    agent = JarvisPrimeAgent()
    health = await agent.health_check()
    assert health["name"] == "jarvis"
    assert "conversation_turns" in health
    assert "has_workflow_engine" in health


@pytest.mark.asyncio
async def test_jarvis_agent_tool_execution_delegation() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(
            task_type="tool.execute",
            payload={"tool_name": "system_info", "arguments": {}},
        )
    )
    # Falls back to planner handling since no workflow engine
    assert result.success


@pytest.mark.asyncio
async def test_jarvis_agent_single_agent_fallback() -> None:
    """Without workflow engine, JPA should fall back to planner for multi-agent requests."""
    agent = JarvisPrimeAgent()
    response = await agent._execute_and_merge("do research and generate code", ["information.retrieve", "code.generate"])
    assert response
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_jarvis_agent_handle_tool_execution_task_type() -> None:
    """When task_type is tool.execute in payload, it should be handled."""
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(
            task_type="jarvis.process",
            payload={"goal": "run system_info tool", "task_type": "tool.execute", "tool_name": "system_info", "tool_arguments": {}},
        )
    )
    assert result.success


# =========================================================================
# Simple conversation detection & handling
# =========================================================================

class TestIsSimpleConversation:

    @pytest.mark.parametrize("phrase", [
        "hi", "hello", "hey", "greetings", "howdy",
        "good morning", "good afternoon", "good evening", "good day",
        "hi jarvis", "hello jarvis", "hey jarvis",
        "hi there", "hello there", "hey there",
        "thanks", "thank you", "cheers", "thanks jarvis",
        "bye", "goodbye", "good bye", "see you", "see you later", "bye jarvis",
        "how are you", "how's it going", "how are you doing", "what's up", "sup",
        "nice", "nice to meet you",
        "have a good day", "take care",
        "hello, how are you?",
    ])
    def test_simple_conversation_true(self, phrase: str) -> None:
        assert JarvisPrimeAgent._is_simple_conversation(phrase), f"Expected True for '{phrase}'"

    @pytest.mark.parametrize("phrase", [
        "hello world program in python",
        "hi, can you search for AI papers",
        "good morning class",
        "hey everyone, let's start",
        "thank you for the help earlier",
        "how are you doing on this fine day",
        "what is the weather like",
        "search for python tutorial",
        "generate code for fibonacci",
        "what is 2+2",
        "",
        "   ",
        "a very long greeting phrase that should not match",
    ])
    def test_simple_conversation_false(self, phrase: str) -> None:
        assert not JarvisPrimeAgent._is_simple_conversation(phrase), f"Expected False for '{phrase}'"


@pytest.mark.asyncio
async def test_jarvis_agent_greeting_returns_hello() -> None:
    """Plain greetings should return a friendly hello."""
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="jarvis.process", payload={"goal": "hi jarvis"})
    )
    assert result.success
    assert "Hello" in result.data.get("response", "")


@pytest.mark.asyncio
async def test_jarvis_agent_goodbye_returns_farewell() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="jarvis.process", payload={"goal": "bye"})
    )
    assert result.success
    assert "Goodbye" in result.data.get("response", "")


@pytest.mark.asyncio
async def test_jarvis_agent_thanks_returns_acknowledgment() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="jarvis.process", payload={"goal": "thank you"})
    )
    assert result.success
    assert "welcome" in result.data.get("response", "").lower()


@pytest.mark.asyncio
async def test_jarvis_agent_how_are_you_returns_response() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="jarvis.process", payload={"goal": "how are you?"})
    )
    assert result.success
    assert "well" in result.data.get("response", "").lower()


@pytest.mark.asyncio
async def test_jarvis_agent_greeting_no_memory_enrichment() -> None:
    """Greetings should not trigger memory enrichment."""
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="jarvis.process", payload={"goal": "hello"})
    )
    assert result.data.get("memory_enriched") is False
    assert result.data.get("memory_count") == 0


@pytest.mark.asyncio
async def test_jarvis_agent_greeting_does_not_route_to_friday() -> None:
    """Greetings must never produce 'Query is required' — the FridayAgent
    error message that indicates incorrect specialist routing."""
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="jarvis.process", payload={"goal": "hi jarvis"})
    )
    response = result.data.get("response", "")
    assert "Query is required" not in response, f"Greeting incorrectly routed to FridayAgent: {response}"
    assert result.success


# =========================================================================
# Knowledge question routing
# =========================================================================

@pytest.mark.asyncio
async def test_knowledge_question_via_knowledge_task_type() -> None:
    """A 'who is' query via knowledge task_type should route to LLM directly
    without hitting specialist agents."""
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="knowledge", payload={"goal": "Who is Purna?"})
    )
    assert result.success
    response = result.data.get("response", "")
    # Must never contain FridayAgent error or specialist routing artifacts
    assert "Query is required" not in response
    assert "Security scan" not in response
    assert result.data.get("memory_enriched") is False


@pytest.mark.asyncio
async def test_knowledge_question_what_is_ai() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="knowledge", payload={"goal": "What is AI?"})
    )
    assert result.success
    response = result.data.get("response", "")
    assert "Query is required" not in response
    assert "Security scan" not in response


@pytest.mark.asyncio
async def test_knowledge_question_explain_recursion() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="knowledge", payload={"goal": "Explain recursion"})
    )
    assert result.success


@pytest.mark.asyncio
async def test_knowledge_question_tell_joke() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="knowledge", payload={"goal": "Tell me a joke"})
    )
    assert result.success


@pytest.mark.asyncio
async def test_knowledge_question_who_invented_python() -> None:
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="knowledge", payload={"goal": "Who invented Python?"})
    )
    assert result.success


# =========================================================================
# Routing correctness: specialist agents must not activate for general queries
# =========================================================================

@pytest.mark.asyncio
async def test_rule_based_decompose_no_research_for_who_is() -> None:
    """'who is' should NOT trigger information.retrieve — it's a knowledge question."""
    agent = JarvisPrimeAgent()
    types = await agent._rule_based_decompose("who is Purna")
    assert "information.retrieve" not in types
    assert types == []


@pytest.mark.asyncio
async def test_rule_based_decompose_no_research_for_what_is() -> None:
    """'what is' should NOT trigger information.retrieve — it's a knowledge question."""
    agent = JarvisPrimeAgent()
    types = await agent._rule_based_decompose("what is AI")
    assert "information.retrieve" not in types
    assert types == []


@pytest.mark.asyncio
async def test_rule_based_decompose_no_research_for_tell_me_about() -> None:
    agent = JarvisPrimeAgent()
    types = await agent._rule_based_decompose("tell me about neural networks")
    assert "information.retrieve" not in types
    assert types == []


@pytest.mark.asyncio
async def test_rule_based_decompose_still_matches_research() -> None:
    """'search for' should still trigger information.retrieve."""
    agent = JarvisPrimeAgent()
    types = await agent._rule_based_decompose("search for AI papers")
    assert "information.retrieve" in types


@pytest.mark.asyncio
async def test_knowledge_task_never_uses_decomposition() -> None:
    """Knowledge task_type must bypass _process_goal and decomposition entirely."""
    agent = JarvisPrimeAgent()
    result = await agent.handle(
        AgentTask(task_type="knowledge", payload={"goal": "Who is Virat Kohli?"})
    )
    assert result.success
    response = result.data.get("response", "")
    # The response should be a real answer, not a routing artifact
    assert len(response) > 10, f"Response too short: {response}"
    assert "Query is required" not in response
