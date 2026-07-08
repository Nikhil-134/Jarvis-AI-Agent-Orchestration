"""Ultron agent — Security & Monitoring specialist."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_SECURITY, Capability
from agents.contracts import AgentResult, AgentTask

_logger = logging.getLogger(__name__)

# Bounds for the local filesystem scan — keep it responsive and DoS-resistant.
_MAX_SCAN_FILES = 2000
_MAX_FILE_BYTES = 1_000_000
_MAX_MATCHES = 500


class UltronAgent(Agent):
    """Agent responsible for security scanning, system monitoring, and log analysis."""

    def __init__(
        self,
        llm_provider: Any | None = None,
        memory_service: Any | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="ultron",
            supported_task_types=("security.scan", "system.monitor", "security.analyze"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Capability]:
        return [CAPABILITY_SECURITY]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"UltronAgent cannot handle task type: {task.task_type}",
            )

        handlers = {
            "security.scan": self._handle_security_scan,
            "system.monitor": self._handle_system_monitor,
            "security.analyze": self._handle_security_analyze,
        }

        handler = handlers.get(task.task_type)
        if handler is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Unknown task type: {task.task_type}",
            )

        try:
            return await handler(task)
        except Exception:
            _logger.exception("UltronAgent failed to handle task %s", task.task_id)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Internal error while processing task.",
            )

    async def _handle_security_scan(self, task: AgentTask) -> AgentResult:
        target = str(task.payload.get("target", "")).strip()
        pattern = str(task.payload.get("pattern", "")).strip()
        _logger.info("Scanning target=%s for pattern=%s", target, pattern)

        if not target:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No scan target provided.",
                data={"status": "error"},
            )
        if not pattern:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No search pattern provided.",
                data={"status": "error"},
            )

        root = Path(target)
        if not root.exists():
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Scan target does not exist: {target}",
                data={"status": "error", "target": target},
            )

        # Pure-Python scan: no shell, so the caller-supplied target/pattern can
        # never be interpreted as a command (removes command-injection risk).
        matches, files_scanned, truncated = await asyncio.to_thread(
            self._scan_paths, root, pattern
        )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message=f"Security scan completed: {len(matches)} match(es) in {files_scanned} file(s).",
            data={
                "status": "completed",
                "target": target,
                "pattern": pattern,
                "match_count": len(matches),
                "matches": matches,
                "files_scanned": files_scanned,
                "truncated": truncated,
            },
        )

    @staticmethod
    def _scan_paths(root: Path, pattern: str) -> tuple[list[dict[str, Any]], int, bool]:
        """Case-insensitive substring scan of *root* for *pattern*.

        Runs synchronously in a worker thread. Bounded by ``_MAX_SCAN_FILES``,
        ``_MAX_FILE_BYTES`` and ``_MAX_MATCHES`` so a huge tree cannot hang or
        exhaust memory.
        """
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        matches: list[dict[str, Any]] = []
        files_scanned = 0
        truncated = False
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if files_scanned >= _MAX_SCAN_FILES or len(matches) >= _MAX_MATCHES:
                truncated = True
                break
            if not path.is_file():
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            files_scanned += 1
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append({"file": str(path), "line": lineno, "text": line.strip()[:200]})
                    if len(matches) >= _MAX_MATCHES:
                        truncated = True
                        break
        return matches, files_scanned, truncated

    async def _handle_system_monitor(self, task: AgentTask) -> AgentResult:
        _logger.info("Gathering system resource information")

        if self._tool_engine is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="System monitoring requires a tool engine to read metrics.",
                data={"status": "unavailable"},
            )

        cpu = await self._tool_engine.execute("system_info", metric="cpu")
        memory = await self._tool_engine.execute("system_info", metric="memory")
        disk = await self._tool_engine.execute("system_info", metric="disk")
        info = {
            "cpu": cpu.output if cpu.success else "unavailable",
            "memory": memory.output if memory.success else "unavailable",
            "disk": disk.output if disk.success else "unavailable",
        }

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="System monitor data collected.",
            data={"status": "completed", "system_info": info},
        )

    async def _handle_security_analyze(self, task: AgentTask) -> AgentResult:
        content = task.payload.get("content", "")
        source = task.payload.get("source", "unknown")
        _logger.info("Analyzing security content from source=%s", source)

        concerns = []
        keywords = {
            "error": "Error or failure detected",
            "failed": "Failure detected",
            "exception": "Exception thrown",
            "unauthorized": "Unauthorized access attempt",
            "breach": "Potential security breach",
            "malware": "Malware signature detected",
            "intrusion": "Intrusion attempt detected",
        }

        lower = content.lower()
        for keyword, concern in keywords.items():
            if keyword in lower:
                concerns.append(concern)

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Security analysis completed.",
            data={
                "status": "completed",
                "source": source,
                "concerns": concerns,
                "concern_count": len(concerns),
            },
        )
