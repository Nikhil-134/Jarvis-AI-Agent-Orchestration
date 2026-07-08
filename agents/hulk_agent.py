"""Hulk agent — storage management."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_STORAGE
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider
from memory import MemoryService

_logger = logging.getLogger(__name__)


class HulkAgent(Agent):
    """Agent responsible for file organization, backup, cleanup, and analysis."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="hulk",
            supported_task_types=("storage.organize", "storage.backup", "storage.cleanup", "storage.analyze"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Any]:
        return [CAPABILITY_STORAGE]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"HulkAgent cannot handle task type: {task.task_type}",
            )

        match task.task_type:
            case "storage.organize":
                return await self._organize(task)
            case "storage.backup":
                return await self._backup(task)
            case "storage.cleanup":
                return await self._cleanup(task)
            case "storage.analyze":
                return await self._analyze(task)
            case _:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message=f"Unknown task type: {task.task_type}",
                )

    async def _organize(self, task: AgentTask) -> AgentResult:
        directory = Path(task.payload.get("directory", ""))
        pattern = task.payload.get("pattern", "*")
        _logger.info("Organizing files in %s with pattern %s", directory, pattern)
        if not directory.is_dir():
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Directory does not exist: {directory}",
            )
        organized = 0
        for f in directory.glob(pattern):
            if f.is_file():
                ext = f.suffix.lstrip(".") or "no_extension"
                ext_dir = directory / ext
                ext_dir.mkdir(exist_ok=True)
                dest = ext_dir / f.name
                if not dest.exists():
                    f.rename(dest)
                    organized += 1
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message=f"Organized {organized} file(s) in {directory}.",
            data={"directory": str(directory), "organized": organized},
        )

    async def _backup(self, task: AgentTask) -> AgentResult:
        source = Path(task.payload.get("source", ""))
        destination = Path(task.payload.get("destination", ""))
        _logger.info("Backing up %s to %s", source, destination)
        if not source.exists():
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Source does not exist: {source}",
            )
        try:
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(destination))
            elif source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(source), str(destination), dirs_exist_ok=True)
        except Exception as exc:
            _logger.exception("Backup failed")
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Backup failed: {exc}",
            )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Backup completed.",
            data={"source": str(source), "destination": str(destination)},
        )

    async def _cleanup(self, task: AgentTask) -> AgentResult:
        directory = Path(task.payload.get("directory", ""))
        age_days = task.payload.get("age_days", 30)
        patterns = task.payload.get("patterns", ["*.tmp", "*.log", "*.temp", "__pycache__"])
        _logger.info("Cleaning up %s older than %d days", directory, age_days)
        if not directory.is_dir():
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Directory does not exist: {directory}",
            )
        import time
        cutoff = time.time() - age_days * 86400
        removed = 0
        freed_bytes = 0
        for pat in patterns:
            for f in directory.rglob(pat):
                try:
                    if f.is_file():
                        if f.stat().st_mtime < cutoff:
                            freed_bytes += f.stat().st_size
                            f.unlink()
                            removed += 1
                    elif f.is_dir() and not any(f.iterdir()):
                        f.rmdir()
                        removed += 1
                except Exception:
                    _logger.warning("Could not remove %s", f)
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message=f"Cleanup removed {removed} item(s), freed {freed_bytes} bytes.",
            data={"directory": str(directory), "removed": removed, "freed_bytes": freed_bytes},
        )

    async def _analyze(self, task: AgentTask) -> AgentResult:
        directory = Path(task.payload.get("directory", ""))
        _logger.info("Analyzing disk usage in %s", directory)
        if not directory.is_dir():
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Directory does not exist: {directory}",
            )
        total_size = 0
        file_count = 0
        dir_count = 0
        largest_files: list[dict[str, Any]] = []
        for f in directory.rglob("*"):
            try:
                if f.is_file():
                    file_count += 1
                    sz = f.stat().st_size
                    total_size += sz
                    largest_files.append({"name": str(f), "size_bytes": sz})
                elif f.is_dir():
                    dir_count += 1
            except Exception:
                continue
        largest_files.sort(key=lambda x: x["size_bytes"], reverse=True)
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Disk usage analysis completed.",
            data={
                "directory": str(directory),
                "total_size_bytes": total_size,
                "file_count": file_count,
                "dir_count": dir_count,
                "largest_files": largest_files[:10],
            },
        )
