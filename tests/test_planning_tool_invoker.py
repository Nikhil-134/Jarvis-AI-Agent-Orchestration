"""Tests for planning.tool_invoker.ToolInvoker — delegation + honest failure."""

from __future__ import annotations

from tools.engine import ToolResult
from tools.interfaces import PermissionLevel
from planning.capabilities import CapabilityCatalog
from planning.models import TaskNode, TaskStatus
from planning.tool_invoker import ToolInvoker


# --- Fakes ---------------------------------------------------------------

class _Spec:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name


class _FakeTool:
    def __init__(self, name: str, dangerous: bool = False) -> None:
        self.spec = _Spec(name)
        self.permission_level = (
            PermissionLevel.DANGEROUS if dangerous else PermissionLevel.SAFE
        )


class FakeRegistry:
    def __init__(self, tools: dict[str, _FakeTool]) -> None:
        self._tools = tools

    def list_specs(self):
        return [t.spec for t in self._tools.values()]

    def get(self, name: str):
        return self._tools.get(name)


class FakeToolEngine:
    def __init__(self, tools: dict[str, _FakeTool], output: str = "ok",
                 success: bool = True) -> None:
        self.registry = FakeRegistry(tools)
        self.calls: list[tuple[str, dict]] = []
        self._output = output
        self._success = success

    async def execute(self, name, _context=None, **kwargs):
        self.calls.append((name, kwargs))
        return ToolResult(
            success=self._success, output=self._output, tool_name=name,
            execution_time_ms=1.0, error=None if self._success else "boom",
        )


class FakeMemory:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def enrich_prompt(self, query, **kw):
        self.calls.append(query)
        return ("", [])


class FakeInternet:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.calls: list[str] = []

    async def build_context(self, query, max_results=5):
        self.calls.append(query)
        return f"live facts about {query}"


class FakeKnowledge:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: list[str] = []

    @property
    def available(self) -> bool:
        return self._available

    async def answer(self, text: str) -> str:
        self.calls.append(text)
        return f"reasoned: {text}"


def _node(tool: str, desc: str = "step", args: dict | None = None) -> TaskNode:
    return TaskNode(id="n1", description=desc, required_tool=tool, args=args or {})


# --- Tests ---------------------------------------------------------------

class TestServiceDelegation:
    async def test_memory_delegates_to_enrich_prompt(self) -> None:
        mem = FakeMemory()
        cat = CapabilityCatalog(memory_service=mem)
        inv = ToolInvoker(cat, memory_service=mem)
        res = await inv.invoke(_node("memory", "what do I like", {"query": "likes"}))
        assert res.status is TaskStatus.SUCCEEDED
        assert mem.calls == ["likes"]

    async def test_internet_delegates_when_gated_true(self) -> None:
        net = FakeInternet(available=True)
        cat = CapabilityCatalog(internet_service=net)
        inv = ToolInvoker(cat, internet_service=net)
        # "weather" trips the needs_internet freshness gate.
        res = await inv.invoke(_node("internet", "weather in Paris",
                                     {"query": "weather in Paris"}))
        assert res.status is TaskStatus.SUCCEEDED
        assert net.calls  # build_context was called

    async def test_reasoning_delegates_to_knowledge(self) -> None:
        kn = FakeKnowledge(available=True)
        cat = CapabilityCatalog(reasoning_available=True)
        inv = ToolInvoker(cat, knowledge_engine=kn)
        res = await inv.invoke(_node("reasoning", "explain recursion"))
        assert res.status is TaskStatus.SUCCEEDED
        assert kn.calls and "recursion" in res.output


class TestToolDelegation:
    async def test_calculator_delegates_with_expression(self) -> None:
        eng = FakeToolEngine({"calculator": _FakeTool("calculator")}, output="575")
        cat = CapabilityCatalog(tool_engine=eng)
        inv = ToolInvoker(cat, tool_engine=eng)
        res = await inv.invoke(_node("calculator", "23*(18+7)",
                                     {"expression": "23*(18+7)"}))
        assert res.status is TaskStatus.SUCCEEDED
        assert eng.calls == [("calculator", {"expression": "23*(18+7)"})]
        assert res.output == "575"

    async def test_calculator_extracts_expression_from_prose(self) -> None:
        # Planner named the tool but passed the whole sentence (no expression arg).
        eng = FakeToolEngine({"calculator": _FakeTool("calculator")}, output="575")
        cat = CapabilityCatalog(tool_engine=eng)
        inv = ToolInvoker(cat, tool_engine=eng)
        res = await inv.invoke(_node("calculator",
                                     "calculate 23*(18+7) and then explain it"))
        assert res.status is TaskStatus.SUCCEEDED
        # The arithmetic sub-string was extracted, not the prose.
        assert eng.calls[0] == ("calculator", {"expression": "23*(18+7)"})

    async def test_filesystem_uses_real_tool_name(self) -> None:
        eng = FakeToolEngine({"file_system": _FakeTool("file_system")})
        cat = CapabilityCatalog(tool_engine=eng)
        inv = ToolInvoker(cat, tool_engine=eng)
        await inv.invoke(_node("filesystem", "list files", {"path": "/tmp"}))
        assert eng.calls[0][0] == "file_system"

    async def test_tool_failure_is_honest(self) -> None:
        eng = FakeToolEngine({"calculator": _FakeTool("calculator")}, success=False)
        cat = CapabilityCatalog(tool_engine=eng)
        inv = ToolInvoker(cat, tool_engine=eng)
        res = await inv.invoke(_node("calculator", "bad", {"expression": "?"}))
        assert res.status is TaskStatus.FAILED


class TestUnavailable:
    async def test_python_is_unavailable_no_call(self) -> None:
        eng = FakeToolEngine({"shell": _FakeTool("shell", dangerous=True)})
        cat = CapabilityCatalog(tool_engine=eng)
        inv = ToolInvoker(cat, tool_engine=eng)
        res = await inv.invoke(_node("python", "run code"))
        assert res.status is TaskStatus.FAILED
        assert eng.calls == []  # never silently routed to shell

    async def test_dangerous_tool_gated_without_auto_approve(self) -> None:
        eng = FakeToolEngine({"browser": _FakeTool("browser", dangerous=True)})
        cat = CapabilityCatalog(tool_engine=eng, allow_dangerous=False)
        inv = ToolInvoker(cat, tool_engine=eng)
        res = await inv.invoke(_node("browser", "open site", {"url": "x"}))
        assert res.status is TaskStatus.FAILED
        assert eng.calls == []  # refused before executing

    async def test_dangerous_tool_allowed_with_auto_approve(self) -> None:
        eng = FakeToolEngine({"browser": _FakeTool("browser", dangerous=True)})
        cat = CapabilityCatalog(tool_engine=eng, allow_dangerous=True)
        inv = ToolInvoker(cat, tool_engine=eng)
        res = await inv.invoke(_node("browser", "open", {"url": "x"}))
        assert res.status is TaskStatus.SUCCEEDED
        assert eng.calls and eng.calls[0][0] == "browser"

    async def test_unknown_capability_no_reasoning_is_unavailable(self) -> None:
        cat = CapabilityCatalog(reasoning_available=False)
        inv = ToolInvoker(cat)
        res = await inv.invoke(_node("quantum_flux", "?"))
        assert res.status is TaskStatus.FAILED

    async def test_invoker_never_raises_on_backend_error(self) -> None:
        class Boom(FakeMemory):
            async def enrich_prompt(self, query, **kw):
                raise RuntimeError("kaboom")

        mem = Boom()
        cat = CapabilityCatalog(memory_service=mem)
        inv = ToolInvoker(cat, memory_service=mem)
        res = await inv.invoke(_node("memory", "x"))
        assert res.status is TaskStatus.FAILED  # captured, not raised
