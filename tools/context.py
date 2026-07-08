"""Tool execution context — carries runtime configuration to tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Runtime context passed to tool execution.

    Provides timeout, environment, and logging configuration that
    tools can use during execution.  Created by ToolManager and
    propagated through the execution pipeline.
    """

    timeout_seconds: float = 30.0
    env_vars: dict[str, str] = field(default_factory=dict)
    working_directory: str | None = None
    log_level: str = "INFO"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_env(self, key: str, default: str = "") -> str:
        return self.env_vars.get(key, default)

    @property
    def logger(self) -> logging.Logger:
        return _logger
