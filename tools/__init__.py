"""Tools package exports."""

from typing import Any

__all__ = [
    "ITool",
    "IToolRegistry",
    "ToolSpec",
]


def __getattr__(name: str) -> Any:
    if name in {"ITool", "IToolRegistry", "ToolSpec"}:
        from tools.interfaces import ITool, IToolRegistry, ToolSpec
        return {
            "ITool": ITool,
            "IToolRegistry": IToolRegistry,
            "ToolSpec": ToolSpec,
        }[name]
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
