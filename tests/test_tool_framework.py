"""Comprehensive tests for Tool Framework (Phase 6).

Covers: ToolContext, ToolManager, timeout support, file_system tool,
text tool, json tool, uuid tool, hash tool, base64 tool, permissions,
error handling, enable/disable configuration, and memory integration.
"""

from __future__ import annotations

import asyncio
import json as stdjson
import os
import re
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from tools.context import ToolContext
from tools.engine import ToolExecutionEngine, ToolResult
from tools.exceptions import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
    ToolValidationError,
)
from tools.interfaces import ITool, PermissionLevel, ToolSpec
from tools.manager import ToolManager
from tools.permissions import PermissionManager
from tools.registry import ToolRegistry


# =========================================================================
# Dummy tools for testing
# =========================================================================


class SlowTool(ITool):
    """Tool that sleeps for testing timeouts."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="slow", description="A slow tool.", parameters={})

    @property
    def category(self) -> str:
        return "test"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        delay = float(kwargs.get("delay", 0.1))
        await asyncio.sleep(delay)
        return {"success": True, "output": f"slept {delay}s"}


class FailingTool(ITool):
    """Tool that always fails."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="fail", description="Always fails.", parameters={})

    @property
    def category(self) -> str:
        return "test"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Intentional failure")


class DeniedTool(ITool):
    """Tool that requires DANGEROUS permission."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="danger", description="Dangerous tool.", parameters={})

    @property
    def category(self) -> str:
        return "test"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.DANGEROUS

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"success": True, "output": "dangerous action done"}


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def engine(registry: ToolRegistry) -> ToolExecutionEngine:
    return ToolExecutionEngine(registry=registry)


@pytest.fixture
def manager(registry: ToolRegistry) -> ToolManager:
    return ToolManager(registry=registry)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# =========================================================================
# ToolContext tests
# =========================================================================


class TestToolContext:
    def test_default_creation(self) -> None:
        ctx = ToolContext()
        assert ctx.timeout_seconds == 30.0
        assert ctx.env_vars == {}
        assert ctx.working_directory is None
        assert ctx.log_level == "INFO"

    def test_custom_values(self) -> None:
        ctx = ToolContext(timeout_seconds=5.0, env_vars={"PATH": "/usr/bin"}, working_directory="/tmp", log_level="DEBUG")
        assert ctx.timeout_seconds == 5.0
        assert ctx.get_env("PATH") == "/usr/bin"
        assert ctx.working_directory == "/tmp"
        assert ctx.log_level == "DEBUG"

    def test_get_env_fallback(self) -> None:
        ctx = ToolContext()
        assert ctx.get_env("MISSING", "default") == "default"
        assert ctx.get_env("MISSING") == ""


# =========================================================================
# ToolManager tests
# =========================================================================


class TestToolManager:
    async def test_register_and_list(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        assert manager.tool_count == 1
        specs = manager.list_tools()
        assert len(specs) == 1
        assert specs[0].name == "slow"

    async def test_register_twice_raises(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        with pytest.raises(ToolAlreadyRegisteredError):
            manager.register_tool(SlowTool())

    async def test_get_tool(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        tool = manager.get_tool("slow")
        assert tool is not None
        assert tool.spec.name == "slow"
        assert manager.get_tool("nonexistent") is None

    async def test_register_many(self, registry: ToolRegistry) -> None:
        mgr = ToolManager(registry=registry)
        mgr.register_tools([SlowTool(), FailingTool()])
        assert mgr.tool_count == 2

    async def test_unregister(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        manager.unregister_tool("slow")
        assert manager.tool_count == 0
        with pytest.raises(ToolNotFoundError):
            manager.unregister_tool("slow")

    async def test_list_categories(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        assert "test" in manager.list_categories()

    async def test_get_by_category(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        tools = manager.get_tools_by_category("test")
        assert len(tools) == 1

    async def test_enable_disable_tool(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        assert manager.is_tool_enabled("slow") is True
        manager.disable_tool("slow")
        assert manager.is_tool_enabled("slow") is False
        manager.enable_tool("slow")
        assert manager.is_tool_enabled("slow") is True

    async def test_disabled_tool_execution_returns_error(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        manager.disable_tool("slow")
        result = await manager.execute("slow")
        assert result.success is False
        assert "disabled" in (result.error or "").lower()

    async def test_enabled_tools_filter(self, registry: ToolRegistry) -> None:
        mgr = ToolManager(registry=registry, enabled_tools={"slow"})
        mgr.register_tool(SlowTool())
        mgr.register_tool(FailingTool())
        assert mgr.is_tool_enabled("slow") is True
        assert mgr.is_tool_enabled("fail") is False

    async def test_execute_many(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        results = await manager.execute_many([("slow", {"delay": 0.01}), ("slow", {"delay": 0.01})])
        assert len(results) == 2
        assert results[0].success is True

    async def test_health(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        health = manager.health()
        assert health["tool_count"] == 1
        assert "categories" in health
        assert "tools" in health

    async def test_list_specs_for_llm(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        specs = manager.list_specs_for_llm()
        assert len(specs) == 1
        assert specs[0]["type"] == "function"
        assert specs[0]["function"]["name"] == "slow"

    async def test_disabled_tool_excluded_from_llm_specs(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        manager.disable_tool("slow")
        specs = manager.list_specs_for_llm()
        assert len(specs) == 0

    async def test_execute_missing_tool(self, manager: ToolManager) -> None:
        result = await manager.execute("nonexistent")
        assert result.success is False

    async def test_set_enabled_disabled(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        manager.register_tool(FailingTool())
        manager.set_enabled_tools({"slow"})
        assert manager.is_tool_enabled("slow") is True
        assert manager.is_tool_enabled("fail") is False
        manager.set_disabled_tools({"slow"})
        assert manager.is_tool_enabled("slow") is False


# =========================================================================
# Timeout tests
# =========================================================================


class TestTimeout:
    async def test_timeout_raises_error(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        slow = SlowTool()
        registry.register(slow)
        ctx = ToolContext(timeout_seconds=0.05)
        result = await engine.execute("slow", _context=ctx, delay=10)
        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    async def test_no_timeout_for_fast_tool(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        registry.register(SlowTool())
        ctx = ToolContext(timeout_seconds=5.0)
        result = await engine.execute("slow", _context=ctx, delay=0.01)
        assert result.success is True

    async def test_timeout_through_manager(self, manager: ToolManager) -> None:
        manager.register_tool(SlowTool())
        ctx = ToolContext(timeout_seconds=0.05)
        result = await manager.execute("slow", context=ctx, delay=10)
        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    async def test_default_timeout(self, registry: ToolRegistry) -> None:
        mgr = ToolManager(registry=registry, default_timeout=0.05)
        mgr.register_tool(SlowTool())
        result = await mgr.execute("slow", delay=10)
        assert result.success is False
        assert "timed out" in (result.error or "").lower()


# =========================================================================
# Permission tests
# =========================================================================


class TestPermissions:
    async def test_safe_tool_passes_without_callback(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        registry.register(SlowTool())
        result = await engine.execute("slow")
        assert result.success is True

    async def test_dangerous_tool_denied_without_callback(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        registry.register(DeniedTool())
        result = await engine.execute("danger")
        assert result.success is False
        assert "denied" in (result.error or "").lower() or "permission" in (result.error or "").lower()

    async def test_dangerous_tool_with_callback_approve(self, registry: ToolRegistry) -> None:
        async def approve(name: str, reason: str) -> bool:
            return True
        pm = PermissionManager(confirmation_callback=approve)
        engine = ToolExecutionEngine(registry=registry, permission_manager=pm)
        registry.register(DeniedTool())
        result = await engine.execute("danger")
        assert result.success is True

    async def test_dangerous_tool_with_callback_deny(self, registry: ToolRegistry) -> None:
        async def deny(name: str, reason: str) -> bool:
            return False
        pm = PermissionManager(confirmation_callback=deny)
        engine = ToolExecutionEngine(registry=registry, permission_manager=pm)
        registry.register(DeniedTool())
        result = await engine.execute("danger")
        assert result.success is False
        assert "denied" in (result.error or "").lower()

    async def test_auto_approve(self, registry: ToolRegistry) -> None:
        pm = PermissionManager(auto_approve_dangerous=True)
        engine = ToolExecutionEngine(registry=registry, permission_manager=pm)
        registry.register(DeniedTool())
        result = await engine.execute("danger")
        assert result.success is True


# =========================================================================
# Error handling tests
# =========================================================================


class TestErrorHandling:
    async def test_tool_not_found(self, engine: ToolExecutionEngine) -> None:
        result = await engine.execute("nonexistent")
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    async def test_tool_execution_exception(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        registry.register(FailingTool())
        result = await engine.execute("fail")
        assert result.success is False
        assert "failed" in (result.error or "").lower()

    async def test_tool_validation_missing_args(self, registry: ToolRegistry) -> None:
        tool = SlowTool()
        registry.register(tool)
        with pytest.raises(ToolValidationError):
            ToolExecutionEngine._validate_args({"name": "test"}, {"required": ["missing"], "properties": {}})

    async def test_tool_result_fields(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        registry.register(SlowTool())
        result = await engine.execute("slow", delay=0.01)
        assert isinstance(result, ToolResult)
        assert result.tool_name == "slow"
        assert result.success is True
        assert result.execution_time_ms > 0
        assert result.error is None


# =========================================================================
# FileSystemTool tests
# =========================================================================


class TestFileSystemTool:
    @pytest.mark.asyncio
    async def test_read_write_file(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        test_file = tmp_dir / "test.txt"
        test_file.write_text("hello world")

        result = await tool.execute(operation="read", path=str(test_file))
        assert result["success"] is True
        assert "hello world" in result["output"]

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        test_file = tmp_dir / "write_test.txt"

        result = await tool.execute(operation="write", path=str(test_file), content="new content")
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_create_folder(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        folder = tmp_dir / "new_folder"

        result = await tool.execute(operation="create_folder", path=str(folder))
        assert result["success"] is True
        assert folder.exists()
        assert folder.is_dir()

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        (tmp_dir / "a.txt").write_text("a")
        (tmp_dir / "b.txt").write_text("b")

        result = await tool.execute(operation="list_directory", path=str(tmp_dir))
        assert result["success"] is True
        assert "a.txt" in result["output"]
        assert "b.txt" in result["output"]

    @pytest.mark.asyncio
    async def test_move_file(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        src = tmp_dir / "source.txt"
        src.write_text("move me")
        dst = tmp_dir / "dest.txt"

        result = await tool.execute(operation="move", path=str(src), destination=str(dst))
        assert result["success"] is True
        assert not src.exists()
        assert dst.exists()

    @pytest.mark.asyncio
    async def test_copy_file(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        src = tmp_dir / "original.txt"
        src.write_text("copy me")
        dst = tmp_dir / "copy.txt"

        result = await tool.execute(operation="copy", path=str(src), destination=str(dst))
        assert result["success"] is True
        assert src.exists()
        assert dst.exists()

    @pytest.mark.asyncio
    async def test_delete_file(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        test_file = tmp_dir / "delete_me.txt"
        test_file.write_text("delete")

        result = await tool.execute(operation="delete", path=str(test_file))
        assert result["success"] is True
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_search_files(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        (tmp_dir / "alpha.py").write_text("")
        (tmp_dir / "beta.py").write_text("")

        result = await tool.execute(operation="search", path=str(tmp_dir), pattern="*.py")
        assert result["success"] is True
        assert "alpha.py" in result["output"]
        assert "beta.py" in result["output"]

    @pytest.mark.asyncio
    async def test_unknown_operation(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        result = await tool.execute(operation="unknown", path=str(tmp_dir))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_operation(self) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        result = await tool.execute(path="/tmp")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, tmp_dir: Path) -> None:
        from tools.builtins.file_system_tool import FileSystemTool

        tool = FileSystemTool()
        result = await tool.execute(operation="read", path=str(tmp_dir / "nope.txt"))
        assert result["success"] is False
        assert "not found" in result.get("error", "").lower()


# =========================================================================
# TextTool tests
# =========================================================================


class TestTextTool:
    @pytest.mark.asyncio
    async def test_word_count(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(operation="word_count", text="hello world foo bar")
        assert result["success"] is True
        assert result["data"]["word_count"] == 4

    @pytest.mark.asyncio
    async def test_word_count_empty(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(operation="word_count", text="")
        assert result["data"]["word_count"] == 0

    @pytest.mark.asyncio
    async def test_char_count(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(operation="char_count", text="hello")
        assert result["success"] is True
        assert result["data"]["character_count"] == 5

    @pytest.mark.asyncio
    async def test_regex_search(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(operation="regex_search", text="hello 123 world 456", pattern=r"\d+")
        assert result["success"] is True
        assert result["data"]["count"] == 2

    @pytest.mark.asyncio
    async def test_regex_search_invalid_pattern(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(operation="regex_search", text="hello", pattern=r"[invalid")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_regex_search_no_pattern(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(operation="regex_search", text="hello")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_summarize_short_text(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(operation="summarize", text="Short text.")
        assert result["success"] is True
        assert result["output"] == "Short text."

    @pytest.mark.asyncio
    async def test_summarize_long_text(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        long_text = "This is a very long text. " * 50
        result = await tool.execute(operation="summarize", text=long_text, max_length=100)
        assert result["success"] is True
        assert len(result["output"]) <= 120

    @pytest.mark.asyncio
    async def test_no_operation(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(text="hello")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_unknown_operation(self) -> None:
        from tools.builtins.text_tool import TextTool

        tool = TextTool()
        result = await tool.execute(operation="unknown", text="hello")
        assert result["success"] is False


# =========================================================================
# JsonTool tests
# =========================================================================


class TestJsonTool:
    @pytest.mark.asyncio
    async def test_validate_valid(self) -> None:
        from tools.builtins.json_tool import JsonTool

        tool = JsonTool()
        result = await tool.execute(operation="validate", data='{"key": "value"}')
        assert result["success"] is True
        assert result["data"]["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_invalid(self) -> None:
        from tools.builtins.json_tool import JsonTool

        tool = JsonTool()
        result = await tool.execute(operation="validate", data="not json")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_pretty_print(self) -> None:
        from tools.builtins.json_tool import JsonTool

        tool = JsonTool()
        result = await tool.execute(operation="pretty_print", data='{"a":1,"b":2}')
        assert result["success"] is True
        assert '"a":' in result["output"]

    @pytest.mark.asyncio
    async def test_pretty_print_invalid(self) -> None:
        from tools.builtins.json_tool import JsonTool

        tool = JsonTool()
        result = await tool.execute(operation="pretty_print", data="bad")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_convert_dict(self) -> None:
        from tools.builtins.json_tool import JsonTool

        tool = JsonTool()
        result = await tool.execute(operation="convert", data='{"a": 1, "b": 2}')
        assert result["success"] is True
        assert "a:" in result["output"]

    @pytest.mark.asyncio
    async def test_convert_list(self) -> None:
        from tools.builtins.json_tool import JsonTool

        tool = JsonTool()
        result = await tool.execute(operation="convert", data='["a", "b"]')
        assert result["success"] is True
        assert "a" in result["output"]

    @pytest.mark.asyncio
    async def test_no_operation(self) -> None:
        from tools.builtins.json_tool import JsonTool

        tool = JsonTool()
        result = await tool.execute(data='{}')
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_unknown_operation(self) -> None:
        from tools.builtins.json_tool import JsonTool

        tool = JsonTool()
        result = await tool.execute(operation="unknown", data='{}')
        assert result["success"] is False


# =========================================================================
# UuidTool tests
# =========================================================================


class TestUuidTool:
    @pytest.mark.asyncio
    async def test_generate_single(self) -> None:
        from tools.builtins.uuid_tool import UuidTool

        tool = UuidTool()
        result = await tool.execute()
        assert result["success"] is True
        assert result["data"]["count"] == 1
        UUID(result["output"].strip())

    @pytest.mark.asyncio
    async def test_generate_multiple(self) -> None:
        from tools.builtins.uuid_tool import UuidTool

        tool = UuidTool()
        result = await tool.execute(count=5)
        assert result["success"] is True
        assert result["data"]["count"] == 5
        assert len(result["data"]["uuids"]) == 5

    @pytest.mark.asyncio
    async def test_generate_version_1(self) -> None:
        from tools.builtins.uuid_tool import UuidTool

        tool = UuidTool()
        result = await tool.execute(version=1)
        assert result["success"] is True
        UUID(result["output"].strip())

    @pytest.mark.asyncio
    async def test_invalid_version(self) -> None:
        from tools.builtins.uuid_tool import UuidTool

        tool = UuidTool()
        result = await tool.execute(version=3)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_max_count(self) -> None:
        from tools.builtins.uuid_tool import UuidTool

        tool = UuidTool()
        result = await tool.execute(count=200)
        assert result["success"] is True
        assert result["data"]["count"] == 100


# =========================================================================
# HashTool tests
# =========================================================================


class TestHashTool:
    @pytest.mark.asyncio
    async def test_sha256_hex(self) -> None:
        from tools.builtins.hash_tool import HashTool

        tool = HashTool()
        result = await tool.execute(algorithm="sha256", data="hello")
        assert result["success"] is True
        assert len(result["output"]) == 64

    @pytest.mark.asyncio
    async def test_md5_hex(self) -> None:
        from tools.builtins.hash_tool import HashTool

        tool = HashTool()
        result = await tool.execute(algorithm="md5", data="hello")
        assert result["success"] is True
        assert len(result["output"]) == 32

    @pytest.mark.asyncio
    async def test_sha256_base64(self) -> None:
        from tools.builtins.hash_tool import HashTool

        tool = HashTool()
        result = await tool.execute(algorithm="sha256", data="hello", encoding="base64")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_unsupported_algorithm(self) -> None:
        from tools.builtins.hash_tool import HashTool

        tool = HashTool()
        result = await tool.execute(algorithm="sha1", data="hello")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_algorithm(self) -> None:
        from tools.builtins.hash_tool import HashTool

        tool = HashTool()
        result = await tool.execute(data="hello")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_data(self) -> None:
        from tools.builtins.hash_tool import HashTool

        tool = HashTool()
        result = await tool.execute(algorithm="sha256")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_deterministic(self) -> None:
        from tools.builtins.hash_tool import HashTool

        tool = HashTool()
        r1 = await tool.execute(algorithm="sha256", data="hello")
        r2 = await tool.execute(algorithm="sha256", data="hello")
        assert r1["output"] == r2["output"]


# =========================================================================
# Base64Tool tests
# =========================================================================


class TestBase64Tool:
    @pytest.mark.asyncio
    async def test_encode(self) -> None:
        from tools.builtins.base64_tool import Base64Tool

        tool = Base64Tool()
        result = await tool.execute(operation="encode", data="hello")
        assert result["success"] is True
        assert result["output"] == "aGVsbG8="

    @pytest.mark.asyncio
    async def test_decode(self) -> None:
        from tools.builtins.base64_tool import Base64Tool

        tool = Base64Tool()
        result = await tool.execute(operation="decode", data="aGVsbG8=")
        assert result["success"] is True
        assert result["output"] == "hello"

    @pytest.mark.asyncio
    async def test_decode_invalid(self) -> None:
        from tools.builtins.base64_tool import Base64Tool

        tool = Base64Tool()
        result = await tool.execute(operation="decode", data="not-valid-base64!!!")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_operation(self) -> None:
        from tools.builtins.base64_tool import Base64Tool

        tool = Base64Tool()
        result = await tool.execute(data="hello")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_data(self) -> None:
        from tools.builtins.base64_tool import Base64Tool

        tool = Base64Tool()
        result = await tool.execute(operation="encode")
        assert result["success"] is False


# =========================================================================
# Registry tests
# =========================================================================


class TestToolRegistry:
    def test_get_all(self, registry: ToolRegistry) -> None:
        registry.register(SlowTool())
        registry.register(FailingTool())
        all_tools = registry.get_all()
        assert len(all_tools) == 2

    def test_get_by_category(self, registry: ToolRegistry) -> None:
        registry.register(SlowTool())
        tools = registry.get_by_category("test")
        assert len(tools) == 1
        tools = registry.get_by_category("other")
        assert len(tools) == 0

    def test_count(self, registry: ToolRegistry) -> None:
        assert registry.count == 0
        registry.register(SlowTool())
        assert registry.count == 1

    def test_list_specs_for_llm(self, registry: ToolRegistry) -> None:
        registry.register(SlowTool())
        specs = registry.list_specs_for_llm()
        assert len(specs) == 1
        assert specs[0]["type"] == "function"

    def test_get_categories(self, registry: ToolRegistry) -> None:
        registry.register(SlowTool())
        assert registry.get_categories() == {"test"}

    def test_register_duplicate_raises(self, registry: ToolRegistry) -> None:
        registry.register(SlowTool())
        with pytest.raises(ToolAlreadyRegisteredError):
            registry.register(SlowTool())

    def test_unregister_missing_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolNotFoundError):
            registry.unregister("nonexistent")


# =========================================================================
# Execution engine tests
# =========================================================================


class TestToolExecutionEngine:
    async def test_execute_success(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        registry.register(SlowTool())
        result = await engine.execute("slow", delay=0.01)
        assert result.success is True
        assert "slept" in result.output

    async def test_execute_not_found(self, engine: ToolExecutionEngine) -> None:
        result = await engine.execute("nonexistent")
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    async def test_execute_exception(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        registry.register(FailingTool())
        result = await engine.execute("fail")
        assert result.success is False

    async def test_validation(self, registry: ToolRegistry) -> None:
        engine = ToolExecutionEngine(registry=registry)
        registry.register(SlowTool())
        with pytest.raises(ToolValidationError):
            ToolExecutionEngine._validate_args(
                {"name": "test"},
                {"required": ["missing"], "properties": {"name": {"type": "string"}}},
            )

    async def test_validation_type_mismatch(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolValidationError):
            ToolExecutionEngine._validate_args(
                {"count": "not_a_number"},
                {"properties": {"count": {"type": "integer"}}},
            )


# =========================================================================
# Memory integration tests (via ToolAgent)
# =========================================================================


class TestToolMemoryIntegration:
    @pytest.mark.asyncio
    async def test_tool_agent_stores_results_in_memory(self, tmp_path: Path) -> None:
        from agents import ToolAgent
        from agents.contracts import AgentTask
        from memory.document_store import SQLiteDocumentStore
        from memory.memory_manager import MemoryManager
        from memory.memory_service import MemoryService
        from memory.vector_store import ChromaVectorStore

        vs = ChromaVectorStore(str(tmp_path / "tm_vectors"))
        ds = SQLiteDocumentStore(str(tmp_path / "tm_docs.db"))
        mm = MemoryManager(vector_store=vs, document_store=ds, importance_threshold=0.0)
        await mm.initialize()
        svc = MemoryService(mm)

        agent = ToolAgent(memory_service=svc, store_results=True)
        result = await agent.handle(
            AgentTask(
                task_type="tool.execute",
                payload={
                    "tool_name": "system_info",
                    "arguments": {},
                    "task_id": "test-run-1",
                },
            )
        )
        assert result.success is True

        stats = await mm.get_stats()
        assert stats["document_count"] >= 1

    @pytest.mark.asyncio
    async def test_tool_agent_stores_disabled(self, tmp_path: Path) -> None:
        from agents import ToolAgent
        from agents.contracts import AgentTask
        from memory.document_store import SQLiteDocumentStore
        from memory.memory_manager import MemoryManager
        from memory.memory_service import MemoryService
        from memory.vector_store import ChromaVectorStore

        vs = ChromaVectorStore(str(tmp_path / "tm2_vectors"))
        ds = SQLiteDocumentStore(str(tmp_path / "tm2_docs.db"))
        mm = MemoryManager(vector_store=vs, document_store=ds, importance_threshold=0.0)
        await mm.initialize()
        svc = MemoryService(mm)

        agent = ToolAgent(memory_service=svc, store_results=False)
        result = await agent.handle(
            AgentTask(
                task_type="tool.execute",
                payload={"tool_name": "system_info", "arguments": {}},
            )
        )
        assert result.success is True

        stats = await mm.get_stats()
        assert stats["document_count"] == 0
