"""Stark agent — Engineering & Build specialist."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_ENGINEERING, Capability
from agents.contracts import AgentResult, AgentTask

_logger = logging.getLogger(__name__)


class StarkAgent(Agent):
    """Agent responsible for compiling, deploying, and setting up projects."""

    def __init__(
        self,
        llm_provider: Any | None = None,
        memory_service: Any | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="stark",
            supported_task_types=("build.compile", "build.deploy", "project.setup"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Capability]:
        return [CAPABILITY_ENGINEERING]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"StarkAgent cannot handle task type: {task.task_type}",
            )

        handlers = {
            "build.compile": self._handle_build_compile,
            "build.deploy": self._handle_build_deploy,
            "project.setup": self._handle_project_setup,
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
            _logger.exception("StarkAgent failed to handle task %s", task.task_id)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Internal error while processing task.",
            )

    async def _handle_build_compile(self, task: AgentTask) -> AgentResult:
        command = task.payload.get("command", "")
        working_dir = task.payload.get("working_dir", "")
        _logger.info("Compiling with command=%s in %s", command, working_dir)

        if not command:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No build command provided.",
                data={"status": "error"},
            )

        if self._tool_engine is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Build requires a tool engine to execute the shell command.",
                data={"status": "unavailable", "command": command},
            )

        result = await self._tool_engine.execute("shell", command=command, workdir=working_dir)
        output = result.output if result.success else (result.error or "Build failed")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            # Honest result: the build only "succeeds" if the shell command did.
            success=result.success,
            message="Build compilation completed." if result.success else "Build compilation failed.",
            data={
                "status": "completed" if result.success else "error",
                "command": command,
                "working_dir": working_dir,
                "output": output,
            },
        )

    async def _handle_build_deploy(self, task: AgentTask) -> AgentResult:
        artifact = task.payload.get("artifact", "")
        target = task.payload.get("target", "")
        command = task.payload.get("command", "")
        _logger.info("Deploying artifact=%s to target=%s", artifact, target)

        # Deployment is only real when a concrete command is executed via the
        # tool engine. Without both, refuse rather than fabricate success.
        if not command:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No deployment command provided; nothing was deployed.",
                data={"status": "error", "artifact": artifact, "target": target, "deployed": False},
            )

        if self._tool_engine is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Deployment requires a tool engine to execute the command.",
                data={"status": "unavailable", "artifact": artifact, "target": target, "deployed": False},
            )

        result = await self._tool_engine.execute("shell", command=command)
        output = result.output if result.success else (result.error or "Deployment failed")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=result.success,
            message="Deployment completed." if result.success else "Deployment failed.",
            data={
                "status": "completed" if result.success else "error",
                "artifact": artifact,
                "target": target,
                "command": command,
                "output": output,
                "deployed": result.success,
            },
        )

    async def _handle_project_setup(self, task: AgentTask) -> AgentResult:
        project_name = task.payload.get("project_name", "")
        template = task.payload.get("template", "default")
        components = task.payload.get("components", [])
        _logger.info("Setting up project=%s with template=%s", project_name, template)

        if not project_name:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No project_name provided for setup.",
                data={"status": "error"},
            )

        structure = {
            "project_name": project_name,
            "template": template,
            "directories": [
                f"{project_name}/src",
                f"{project_name}/tests",
                f"{project_name}/docs",
                f"{project_name}/config",
            ],
            "files": [
                f"{project_name}/README.md",
                f"{project_name}/requirements.txt",
                f"{project_name}/src/__init__.py",
                f"{project_name}/tests/__init__.py",
            ],
            "components": components,
        }

        # Creating the scaffold requires a tool engine that can touch the
        # filesystem; without it nothing is created, so report honestly.
        if self._tool_engine is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Project setup requires a tool engine to create files and directories.",
                data={"status": "unavailable", "project_name": project_name, "structure": structure},
            )

        created_dirs: list[str] = []
        created_files: list[str] = []
        errors: list[str] = []
        for directory in structure["directories"]:
            res = await self._tool_engine.execute("file_system", action="mkdir", path=directory)
            if res.success:
                created_dirs.append(directory)
            else:
                errors.append(f"mkdir {directory}: {res.error or 'failed'}")
        for file_path in structure["files"]:
            res = await self._tool_engine.execute("file_system", action="touch", path=file_path)
            if res.success:
                created_files.append(file_path)
            else:
                errors.append(f"touch {file_path}: {res.error or 'failed'}")

        success = not errors
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=success,
            message="Project setup completed." if success else "Project setup completed with errors.",
            data={
                "status": "completed" if success else "error",
                "project_name": project_name,
                "structure": structure,
                "created_directories": created_dirs,
                "created_files": created_files,
                "errors": errors,
            },
        )
