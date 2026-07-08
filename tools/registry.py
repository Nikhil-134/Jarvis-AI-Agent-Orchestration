"""Tool registry — in-memory tool storage with category support."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from tools.exceptions import ToolAlreadyRegisteredError, ToolNotFoundError
from tools.interfaces import ITool, IToolRegistry, ToolSpec

_logger = logging.getLogger(__name__)


class ToolRegistry(IToolRegistry):
    """Thread-safe in-memory tool registry.

    Supports categories via ``ITool.category``, and maintains metadata
    for capability discovery.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ITool] = {}
        self._lock = Lock()

    def register(self, tool: ITool) -> None:
        name = tool.spec.name
        with self._lock:
            if name in self._tools:
                raise ToolAlreadyRegisteredError(
                    f"Tool '{name}' is already registered."
                )
            self._tools[name] = tool
        _logger.info("Registered tool '%s' (category=%s, permission=%s)", name, tool.category, tool.permission_level.name)

    def unregister(self, name: str) -> None:
        with self._lock:
            if name not in self._tools:
                raise ToolNotFoundError(f"Tool '{name}' is not registered.")
            del self._tools[name]
        _logger.info("Unregistered tool '%s'", name)

    def get(self, name: str) -> ITool | None:
        with self._lock:
            return self._tools.get(name)

    def get_all(self) -> list[ITool]:
        with self._lock:
            return list(self._tools.values())

    def list_specs(self) -> list[ToolSpec]:
        with self._lock:
            return [t.spec for t in self._tools.values()]

    def get_by_category(self, category: str) -> list[ITool]:
        with self._lock:
            return [t for t in self._tools.values() if t.category == category]

    def get_categories(self) -> set[str]:
        with self._lock:
            return {t.category for t in self._tools.values()}

    def list_specs_for_llm(self) -> list[dict[str, Any]]:
        """Return tool specs in the format LLM providers expect (OpenAI-compatible)."""
        with self._lock:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.spec.name,
                        "description": t.spec.description,
                        "parameters": t.spec.parameters,
                    },
                }
                for t in self._tools.values()
            ]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._tools)
