from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents import (
    AthenaAgent,
    FridayAgent,
    GeckoAgent,
    HerculesAgent,
    HulkAgent,
    JeromeAgent,
    OracleAgent,
    PepperAgent,
    StarkAgent,
    SteveAgent,
    UltronAgent,
    VeronicaAgent,
    VisionAgent,
)
from agents.capabilities import (
    CAPABILITY_BROWSER,
    CAPABILITY_CODE_ENGINEERING,
    CAPABILITY_COMPUTATION,
    CAPABILITY_DEVOPS,
    CAPABILITY_ENGINEERING,
    CAPABILITY_KNOWLEDGE,
    CAPABILITY_RESEARCH,
    CAPABILITY_SECURITY,
    CAPABILITY_STORAGE,
    CAPABILITY_STRATEGY,
    CAPABILITY_TESTING,
    CAPABILITY_USER_EXPERIENCE,
    CAPABILITY_VISION,
    Capability,
)
from agents.contracts import AgentResult, AgentTask
from llm.base import BaseLLMProvider, LLMConfig
from llm.interfaces import LLMResponse
from tools.engine import ToolExecutionEngine, ToolResult


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(LLMConfig(provider="test", model="test"))

    @property
    def name(self) -> str:
        return "test"

    async def _generate_once(
        self, prompt: str, system_prompt: str | None = None, tools=None
    ) -> LLMResponse:
        return LLMResponse(content="mock response")

    async def _stream_once(
        self, prompt: str, system_prompt: str | None = None, tools=None
    ) -> Any:
        yield "mock"


class _MockToolResult:
    success: bool = True
    output: str = "mock output"
    error: str | None = None
    execution_time_ms: float = 1.0


class _MockToolEngine:
    async def execute(self, name: str, **kwargs: Any) -> _MockToolResult:
        return _MockToolResult()


class _FailingToolResult:
    success: bool = False
    output: str = ""
    error: str | None = "boom"
    execution_time_ms: float = 1.0


class _FailingToolEngine:
    async def execute(self, name: str, **kwargs: Any) -> _FailingToolResult:
        return _FailingToolResult()


# ---------------------------------------------------------------------------
# Parametrized capability map: agent_class -> (expected_capability, task_types)
# ---------------------------------------------------------------------------

AGENT_CAPABILITIES: list[tuple[type, Capability, tuple[str, ...]]] = [
    (FridayAgent, CAPABILITY_RESEARCH, ("research", "information.retrieve", "information.synthesize")),
    (VeronicaAgent, CAPABILITY_CODE_ENGINEERING, ("code.generate", "code.review", "code.refactor", "code.analyze")),
    (VisionAgent, CAPABILITY_VISION, ("vision.analyze", "vision.ocr", "vision.screenshot", "vision.describe")),
    (UltronAgent, CAPABILITY_SECURITY, ("security.scan", "system.monitor", "security.analyze")),
    (AthenaAgent, CAPABILITY_STRATEGY, ("strategy.plan", "task.decompose", "workflow.design")),
    (StarkAgent, CAPABILITY_ENGINEERING, ("build.compile", "build.deploy", "project.setup")),
    (SteveAgent, CAPABILITY_TESTING, ("test.run", "test.create", "test.analyze", "coverage.report")),
    (OracleAgent, CAPABILITY_KNOWLEDGE, ("knowledge.store", "knowledge.query", "knowledge.search")),
    (GeckoAgent, CAPABILITY_BROWSER, ("web.fetch", "browser.navigate", "browser.scrape", "web.automate")),
    (HerculesAgent, CAPABILITY_COMPUTATION, ("compute.process", "data.transform", "batch.execute")),
    (PepperAgent, CAPABILITY_USER_EXPERIENCE, ("ux.notify", "ux.display", "ux.interact", "ux.speak")),
    (HulkAgent, CAPABILITY_STORAGE, ("storage.organize", "storage.backup", "storage.cleanup", "storage.analyze")),
    (JeromeAgent, CAPABILITY_DEVOPS, ("devops.configure", "devops.deploy", "system.admin", "devops.monitor")),
]


# ---------------------------------------------------------------------------
# Oracle's supported types includes knowledge.index which is not in the
# requirement list above, so we add it here.
# ---------------------------------------------------------------------------

ORACLE_EXTRA_TYPES = ("knowledge.index",)


# ===================================================================
# 1. FridayAgent – Research & Information Synthesis
# ===================================================================

class TestFridayAgent:
    agent_cls = FridayAgent
    capability = CAPABILITY_RESEARCH
    task_types = ("research", "information.retrieve", "information.synthesize")

    @pytest.fixture
    def agent(self) -> FridayAgent:
        return FridayAgent(llm_provider=MockLLMProvider())

    def test_capabilities(self, agent: FridayAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: FridayAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: FridayAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_research_success(self, agent: FridayAgent) -> None:
        result = await agent.handle(AgentTask(task_type="research", payload={"topic": "AI"}))
        assert result.success is True
        assert result.agent_name == "friday"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_information_retrieve_success(self, agent: FridayAgent) -> None:
        result = await agent.handle(AgentTask(task_type="information.retrieve", payload={"query": "test"}))
        assert result.agent_name == "friday"

    @pytest.mark.asyncio
    async def test_handle_information_synthesize_success(self, agent: FridayAgent) -> None:
        result = await agent.handle(AgentTask(task_type="information.synthesize", payload={"pieces": ["a", "b"]}))
        assert result.success is True
        assert result.agent_name == "friday"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: FridayAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 2. VeronicaAgent – Code Engineering
# ===================================================================

class TestVeronicaAgent:
    agent_cls = VeronicaAgent
    capability = CAPABILITY_CODE_ENGINEERING
    task_types = ("code.generate", "code.review", "code.refactor", "code.analyze")

    @pytest.fixture
    def agent(self) -> VeronicaAgent:
        return VeronicaAgent(llm_provider=MockLLMProvider())

    def test_capabilities(self, agent: VeronicaAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: VeronicaAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: VeronicaAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_code_generate_success(self, agent: VeronicaAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="code.generate",
            payload={"specification": "hello world", "language": "python"},
        ))
        assert result.success is True
        assert result.agent_name == "veronica"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_code_review_success(self, agent: VeronicaAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="code.review",
            payload={"code": "def foo(): pass", "language": "python"},
        ))
        assert result.success is True
        assert result.agent_name == "veronica"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_code_refactor_success(self, agent: VeronicaAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="code.refactor",
            payload={"code": "def foo(): pass", "language": "python"},
        ))
        assert result.success is True
        assert result.agent_name == "veronica"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_code_analyze_success(self, agent: VeronicaAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="code.analyze",
            payload={"code": "def foo(): pass", "language": "python"},
        ))
        assert result.success is True
        assert result.agent_name == "veronica"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: VeronicaAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 3. VisionAgent – Computer Vision
# ===================================================================

class TestVisionAgent:
    agent_cls = VisionAgent
    capability = CAPABILITY_VISION
    task_types = ("vision.analyze", "vision.ocr", "vision.describe", "vision.screenshot")

    @pytest.fixture
    def agent(self) -> VisionAgent:
        return VisionAgent(llm_provider=MockLLMProvider())

    def test_capabilities(self, agent: VisionAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: VisionAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: VisionAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_vision_analyze_success(self, agent: VisionAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="vision.analyze",
            payload={"path": "/fake/image.png"},
        ))
        assert result.success is True
        assert result.agent_name == "vision"

    @pytest.mark.asyncio
    async def test_handle_vision_ocr_success(self, agent: VisionAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="vision.ocr",
            payload={"path": "/fake/image.png"},
        ))
        assert result.success is True
        assert result.agent_name == "vision"

    @pytest.mark.asyncio
    async def test_handle_vision_describe_success(self, agent: VisionAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="vision.describe",
            payload={"path": "/fake/image.png"},
        ))
        assert result.success is True
        assert result.agent_name == "vision"

    @pytest.mark.asyncio
    async def test_handle_vision_screenshot(self, agent: VisionAgent) -> None:
        result = await agent.handle(AgentTask(task_type="vision.screenshot"))
        assert result.agent_name == "vision"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: VisionAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 4. UltronAgent – Security & Monitoring
# ===================================================================

class TestUltronAgent:
    agent_cls = UltronAgent
    capability = CAPABILITY_SECURITY
    task_types = ("security.scan", "system.monitor", "security.analyze")

    @pytest.fixture
    def agent(self) -> UltronAgent:
        return UltronAgent()

    def test_capabilities(self, agent: UltronAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: UltronAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: UltronAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_security_scan(self, agent: UltronAgent, tmp_path: Any) -> None:
        f = tmp_path / "log.txt"
        f.write_text("line one\nsecret token here\nline three")
        result = await agent.handle(AgentTask(
            task_type="security.scan",
            payload={"target": str(tmp_path), "pattern": "secret"},
        ))
        assert result.success is True
        assert result.agent_name == "ultron"
        assert result.data["status"] == "completed"
        assert result.data["match_count"] >= 1

    @pytest.mark.asyncio
    async def test_handle_security_scan_no_target(self, agent: UltronAgent) -> None:
        result = await agent.handle(AgentTask(task_type="security.scan", payload={"pattern": "x"}))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_handle_security_scan_missing_path(self, agent: UltronAgent, tmp_path: Any) -> None:
        missing = tmp_path / "does_not_exist"
        result = await agent.handle(AgentTask(
            task_type="security.scan",
            payload={"target": str(missing), "pattern": "x"},
        ))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_handle_system_monitor_no_engine(self, agent: UltronAgent) -> None:
        # Fixture agent has no tool engine: metrics cannot be read for real.
        result = await agent.handle(AgentTask(task_type="system.monitor"))
        assert result.success is False
        assert result.agent_name == "ultron"

    @pytest.mark.asyncio
    async def test_handle_system_monitor_with_engine(self) -> None:
        agent = UltronAgent(tool_engine=_MockToolEngine())
        result = await agent.handle(AgentTask(task_type="system.monitor"))
        assert result.success is True
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_security_analyze(self, agent: UltronAgent) -> None:
        result = await agent.handle(AgentTask(task_type="security.analyze"))
        assert result.success is True
        assert result.agent_name == "ultron"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: UltronAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 5. AthenaAgent – Strategic Planning
# ===================================================================

class TestAthenaAgent:
    agent_cls = AthenaAgent
    capability = CAPABILITY_STRATEGY
    task_types = ("strategy.plan", "task.decompose", "workflow.design")

    @pytest.fixture
    def agent(self) -> AthenaAgent:
        return AthenaAgent()

    def test_capabilities(self, agent: AthenaAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: AthenaAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: AthenaAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_strategy_plan(self, agent: AthenaAgent) -> None:
        result = await agent.handle(AgentTask(task_type="strategy.plan"))
        assert result.success is True
        assert result.agent_name == "athena"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_task_decompose(self, agent: AthenaAgent) -> None:
        result = await agent.handle(AgentTask(task_type="task.decompose"))
        assert result.success is True
        assert result.agent_name == "athena"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_workflow_design(self, agent: AthenaAgent) -> None:
        result = await agent.handle(AgentTask(task_type="workflow.design"))
        assert result.success is True
        assert result.agent_name == "athena"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: AthenaAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 6. StarkAgent – Engineering & Build
# ===================================================================

class TestStarkAgent:
    agent_cls = StarkAgent
    capability = CAPABILITY_ENGINEERING
    task_types = ("build.compile", "build.deploy", "project.setup")

    @pytest.fixture
    def agent(self) -> StarkAgent:
        return StarkAgent(tool_engine=_MockToolEngine())

    def test_capabilities(self, agent: StarkAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: StarkAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: StarkAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_build_compile(self, agent: StarkAgent) -> None:
        result = await agent.handle(AgentTask(task_type="build.compile", payload={"command": "make"}))
        assert result.success is True
        assert result.agent_name == "stark"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_build_compile_no_command(self, agent: StarkAgent) -> None:
        result = await agent.handle(AgentTask(task_type="build.compile"))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_handle_build_compile_no_engine(self) -> None:
        agent = StarkAgent()
        result = await agent.handle(AgentTask(task_type="build.compile", payload={"command": "make"}))
        assert result.success is False
        assert result.data["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_handle_build_compile_propagates_failure(self) -> None:
        agent = StarkAgent(tool_engine=_FailingToolEngine())
        result = await agent.handle(AgentTask(task_type="build.compile", payload={"command": "make"}))
        assert result.success is False
        assert result.data["status"] == "error"

    @pytest.mark.asyncio
    async def test_handle_build_deploy(self, agent: StarkAgent) -> None:
        result = await agent.handle(AgentTask(task_type="build.deploy", payload={"command": "deploy.sh"}))
        assert result.success is True
        assert result.agent_name == "stark"
        assert result.data["deployed"] is True

    @pytest.mark.asyncio
    async def test_handle_build_deploy_no_command(self, agent: StarkAgent) -> None:
        result = await agent.handle(AgentTask(task_type="build.deploy"))
        assert result.success is False
        assert result.data["deployed"] is False

    @pytest.mark.asyncio
    async def test_handle_project_setup(self, agent: StarkAgent) -> None:
        result = await agent.handle(AgentTask(task_type="project.setup", payload={"project_name": "proj"}))
        assert result.success is True
        assert result.agent_name == "stark"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_project_setup_no_engine(self) -> None:
        agent = StarkAgent()
        result = await agent.handle(AgentTask(task_type="project.setup", payload={"project_name": "proj"}))
        assert result.success is False
        assert result.data["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: StarkAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 7. SteveAgent – Testing & QA
# ===================================================================

class TestSteveAgent:
    agent_cls = SteveAgent
    capability = CAPABILITY_TESTING
    task_types = ("test.run", "test.create", "test.analyze", "coverage.report")

    @pytest.fixture
    def agent(self) -> SteveAgent:
        return SteveAgent(tool_engine=_MockToolEngine())

    def test_capabilities(self, agent: SteveAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: SteveAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: SteveAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_test_run(self, agent: SteveAgent) -> None:
        result = await agent.handle(AgentTask(task_type="test.run"))
        assert result.success is True
        assert result.agent_name == "steve"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_test_run_no_engine(self) -> None:
        agent = SteveAgent()
        result = await agent.handle(AgentTask(task_type="test.run"))
        assert result.success is False
        assert result.data["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_handle_test_run_propagates_failure(self) -> None:
        agent = SteveAgent(tool_engine=_FailingToolEngine())
        result = await agent.handle(AgentTask(task_type="test.run"))
        assert result.success is False
        assert result.data["status"] == "error"

    @pytest.mark.asyncio
    async def test_handle_test_create(self, agent: SteveAgent) -> None:
        result = await agent.handle(AgentTask(task_type="test.create"))
        assert result.success is True
        assert result.agent_name == "steve"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_test_analyze(self, agent: SteveAgent) -> None:
        result = await agent.handle(AgentTask(task_type="test.analyze"))
        assert result.success is True
        assert result.agent_name == "steve"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_coverage_report(self, agent: SteveAgent) -> None:
        result = await agent.handle(AgentTask(task_type="coverage.report"))
        assert result.success is True
        assert result.agent_name == "steve"
        assert result.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_coverage_report_no_engine(self) -> None:
        agent = SteveAgent()
        result = await agent.handle(AgentTask(task_type="coverage.report"))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: SteveAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 8. OracleAgent – Knowledge Management
# ===================================================================

class TestOracleAgent:
    agent_cls = OracleAgent
    capability = CAPABILITY_KNOWLEDGE
    task_types = ("knowledge.store", "knowledge.query", "knowledge.search")

    @pytest.fixture
    def agent(self) -> OracleAgent:
        return OracleAgent()

    def test_capabilities(self, agent: OracleAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: OracleAgent) -> None:
        for t in (*self.task_types, "knowledge.index"):
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: OracleAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_knowledge_store(self, agent: OracleAgent) -> None:
        result = await agent.handle(AgentTask(task_type="knowledge.store"))
        assert result.success is True
        assert result.agent_name == "oracle"

    @pytest.mark.asyncio
    async def test_handle_knowledge_query(self, agent: OracleAgent) -> None:
        result = await agent.handle(AgentTask(task_type="knowledge.query"))
        assert result.success is True
        assert result.agent_name == "oracle"

    @pytest.mark.asyncio
    async def test_handle_knowledge_search(self, agent: OracleAgent) -> None:
        result = await agent.handle(AgentTask(task_type="knowledge.search"))
        assert result.success is True
        assert result.agent_name == "oracle"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: OracleAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 9. GeckoAgent – Web & Browser
# ===================================================================

class TestGeckoAgent:
    agent_cls = GeckoAgent
    capability = CAPABILITY_BROWSER
    task_types = ("web.fetch", "browser.navigate", "browser.scrape", "web.automate")

    @pytest.fixture
    def agent(self) -> GeckoAgent:
        return GeckoAgent()

    def test_capabilities(self, agent: GeckoAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: GeckoAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: GeckoAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_web_fetch(self, agent: GeckoAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="web.fetch",
            payload={"url": "http://localhost:0/nonexistent"},
        ))
        assert result.agent_name == "gecko"

    @pytest.mark.asyncio
    async def test_handle_browser_navigate(self, agent: GeckoAgent) -> None:
        # No browser engine installed: navigation must not fake success.
        result = await agent.handle(AgentTask(task_type="browser.navigate"))
        assert result.success is False
        assert result.agent_name == "gecko"

    @pytest.mark.asyncio
    async def test_handle_browser_scrape(self, agent: GeckoAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="browser.scrape",
            payload={"url": "http://localhost:0/nonexistent"},
        ))
        assert result.agent_name == "gecko"

    @pytest.mark.asyncio
    async def test_handle_web_automate(self, agent: GeckoAgent) -> None:
        # No browser engine installed: automation must not fake success.
        result = await agent.handle(AgentTask(task_type="web.automate"))
        assert result.success is False
        assert result.agent_name == "gecko"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: GeckoAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 10. HerculesAgent – Computation & Data Processing
# ===================================================================

class TestHerculesAgent:
    agent_cls = HerculesAgent
    capability = CAPABILITY_COMPUTATION
    task_types = ("compute.process", "data.transform", "batch.execute")

    @pytest.fixture
    def agent(self) -> HerculesAgent:
        return HerculesAgent()

    def test_capabilities(self, agent: HerculesAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: HerculesAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: HerculesAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_compute_process_no_engine(self, agent: HerculesAgent) -> None:
        # Fixture agent has no tool engine: computation cannot run for real.
        result = await agent.handle(AgentTask(task_type="compute.process"))
        assert result.success is False
        assert result.agent_name == "hercules"

    @pytest.mark.asyncio
    async def test_handle_compute_process_with_engine(self) -> None:
        agent = HerculesAgent(tool_engine=_MockToolEngine())
        result = await agent.handle(AgentTask(
            task_type="compute.process",
            payload={"operation": "calculator", "input": "1+1"},
        ))
        assert result.success is True
        assert result.agent_name == "hercules"

    @pytest.mark.asyncio
    async def test_handle_data_transform(self, agent: HerculesAgent) -> None:
        result = await agent.handle(AgentTask(task_type="data.transform"))
        assert result.success is True
        assert result.agent_name == "hercules"

    @pytest.mark.asyncio
    async def test_handle_batch_execute_no_ops(self, agent: HerculesAgent) -> None:
        result = await agent.handle(AgentTask(task_type="batch.execute"))
        assert result.success is False
        assert result.agent_name == "hercules"

    @pytest.mark.asyncio
    async def test_handle_batch_execute_with_engine(self) -> None:
        agent = HerculesAgent(tool_engine=_MockToolEngine())
        result = await agent.handle(AgentTask(
            task_type="batch.execute",
            payload={"operations": [{"operation": "calculator", "parameters": {}}]},
        ))
        assert result.success is True
        assert result.data["total"] == 1
        assert result.data["results"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: HerculesAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 11. PepperAgent – User Experience
# ===================================================================

class TestPepperAgent:
    agent_cls = PepperAgent
    capability = CAPABILITY_USER_EXPERIENCE
    task_types = ("ux.notify", "ux.display", "ux.interact", "ux.speak")

    @pytest.fixture
    def agent(self) -> PepperAgent:
        return PepperAgent()

    def test_capabilities(self, agent: PepperAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: PepperAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: PepperAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_ux_notify(self, agent: PepperAgent) -> None:
        result = await agent.handle(AgentTask(task_type="ux.notify"))
        assert result.success is True
        assert result.agent_name == "pepper"

    @pytest.mark.asyncio
    async def test_handle_ux_display(self, agent: PepperAgent) -> None:
        result = await agent.handle(AgentTask(task_type="ux.display"))
        assert result.success is True
        assert result.agent_name == "pepper"

    @pytest.mark.asyncio
    async def test_handle_ux_interact(self, agent: PepperAgent) -> None:
        result = await agent.handle(AgentTask(task_type="ux.interact"))
        assert result.success is True
        assert result.agent_name == "pepper"

    @pytest.mark.asyncio
    async def test_handle_ux_speak(self, agent: PepperAgent) -> None:
        # PepperAgent has no TTS backend; speech is owned by the voice subsystem.
        result = await agent.handle(AgentTask(task_type="ux.speak", payload={"text": "hi"}))
        assert result.success is False
        assert result.agent_name == "pepper"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: PepperAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 12. HulkAgent – Storage Management
# ===================================================================

class TestHulkAgent:
    agent_cls = HulkAgent
    capability = CAPABILITY_STORAGE
    task_types = ("storage.organize", "storage.backup", "storage.cleanup", "storage.analyze")

    @pytest.fixture
    def agent(self) -> HulkAgent:
        return HulkAgent()

    def test_capabilities(self, agent: HulkAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: HulkAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: HulkAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_storage_organize(self, agent: HulkAgent, tmp_path: Any) -> None:
        d = tmp_path / "organize_test"
        d.mkdir()
        (d / "hello.py").write_text("x")
        result = await agent.handle(AgentTask(
            task_type="storage.organize",
            payload={"directory": str(d)},
        ))
        assert result.success is True
        assert result.agent_name == "hulk"

    @pytest.mark.asyncio
    async def test_handle_storage_backup(self, agent: HulkAgent, tmp_path: Any) -> None:
        src = tmp_path / "backup_src"
        src.mkdir()
        (src / "f.txt").write_text("data")
        dst = tmp_path / "backup_dst"
        result = await agent.handle(AgentTask(
            task_type="storage.backup",
            payload={"source": str(src), "destination": str(dst)},
        ))
        assert result.success is True
        assert result.agent_name == "hulk"

    @pytest.mark.asyncio
    async def test_handle_storage_cleanup(self, agent: HulkAgent, tmp_path: Any) -> None:
        d = tmp_path / "cleanup_test"
        d.mkdir()
        result = await agent.handle(AgentTask(
            task_type="storage.cleanup",
            payload={"directory": str(d), "age_days": 0},
        ))
        assert result.success is True
        assert result.agent_name == "hulk"

    @pytest.mark.asyncio
    async def test_handle_storage_analyze(self, agent: HulkAgent, tmp_path: Any) -> None:
        d = tmp_path / "analyze_test"
        d.mkdir()
        (d / "a.txt").write_text("hello")
        result = await agent.handle(AgentTask(
            task_type="storage.analyze",
            payload={"directory": str(d)},
        ))
        assert result.success is True
        assert result.agent_name == "hulk"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: HulkAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message


# ===================================================================
# 13. JeromeAgent – DevOps & Deployment
# ===================================================================

class TestJeromeAgent:
    agent_cls = JeromeAgent
    capability = CAPABILITY_DEVOPS
    task_types = ("devops.configure", "devops.deploy", "system.admin", "devops.monitor")

    @pytest.fixture
    def agent(self) -> JeromeAgent:
        return JeromeAgent(tool_engine=_MockToolEngine())

    def test_capabilities(self, agent: JeromeAgent) -> None:
        assert agent.capabilities == [self.capability]

    def test_can_handle_known(self, agent: JeromeAgent) -> None:
        for t in self.task_types:
            assert agent.can_handle(AgentTask(task_type=t)), f"should handle {t}"

    def test_can_handle_unknown(self, agent: JeromeAgent) -> None:
        assert not agent.can_handle(AgentTask(task_type="unknown"))

    @pytest.mark.asyncio
    async def test_handle_devops_configure(self, agent: JeromeAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="devops.configure",
            payload={"command": "echo test"},
        ))
        assert result.success is True
        assert result.agent_name == "jerome"

    @pytest.mark.asyncio
    async def test_handle_devops_deploy(self, agent: JeromeAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="devops.deploy",
            payload={"command": "deploy.sh"},
        ))
        assert result.success is True
        assert result.agent_name == "jerome"

    @pytest.mark.asyncio
    async def test_handle_system_admin(self, agent: JeromeAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="system.admin",
            payload={"command": "whoami"},
        ))
        assert result.success is True
        assert result.agent_name == "jerome"

    @pytest.mark.asyncio
    async def test_handle_devops_monitor(self, agent: JeromeAgent) -> None:
        result = await agent.handle(AgentTask(
            task_type="devops.monitor",
            payload={"command": "top -n1"},
        ))
        assert result.success is True
        assert result.agent_name == "jerome"

    @pytest.mark.asyncio
    async def test_handle_unknown_task_type(self, agent: JeromeAgent) -> None:
        result = await agent.handle(AgentTask(task_type="unknown"))
        assert result.success is False
        assert "cannot handle" in result.message
