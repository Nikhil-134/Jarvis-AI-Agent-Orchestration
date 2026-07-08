"""Gecko agent — web and browser automation."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agents.base import Agent
from agents.capabilities import CAPABILITY_BROWSER
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider
from memory import MemoryService

_logger = logging.getLogger(__name__)


class GeckoAgent(Agent):
    """Agent responsible for web fetching, scraping, and browser automation."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="gecko",
            supported_task_types=("browser.navigate", "browser.scrape", "web.automate", "web.fetch"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Any]:
        return [CAPABILITY_BROWSER]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"GeckoAgent cannot handle task type: {task.task_type}",
            )

        match task.task_type:
            case "web.fetch":
                return await self._fetch(task)
            case "browser.navigate":
                return await self._navigate(task)
            case "browser.scrape":
                return await self._scrape(task)
            case "web.automate":
                return await self._automate(task)
            case _:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message=f"Unknown task type: {task.task_type}",
                )

    async def _fetch(self, task: AgentTask) -> AgentResult:
        url = task.payload.get("url", "")
        _logger.info("Fetching URL: %s", url)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                text = resp.text
        except Exception as exc:
            _logger.exception("Failed to fetch URL: %s", url)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Failed to fetch URL: {exc}",
            )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="URL fetched successfully.",
            data={"url": url, "content": text, "status_code": resp.status_code},
        )

    async def _navigate(self, task: AgentTask) -> AgentResult:
        url = task.payload.get("url", "")
        _logger.info("Browser navigation requested for URL: %s", url)
        # Real interactive navigation needs a browser engine (playwright/selenium),
        # which is not installed. Do not pretend it succeeded — point the caller at
        # the working `web.fetch` path for static content.
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=False,
            message="Interactive browser navigation requires a browser engine "
            "(not installed). Use 'web.fetch' to retrieve static page content.",
            data={"url": url, "status": "unavailable"},
        )

    async def _scrape(self, task: AgentTask) -> AgentResult:
        url = task.payload.get("url", "")
        selector = task.payload.get("selector")
        _logger.info("Scraping URL (stub): %s selector=%s", url, selector)
        content = ""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                content = resp.text
        except Exception as exc:
            _logger.exception("Failed to scrape URL: %s", url)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Failed to scrape URL: {exc}",
            )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Content scraped successfully.",
            data={"url": url, "content": content[:10000], "selector": selector},
        )

    async def _automate(self, task: AgentTask) -> AgentResult:
        action = task.payload.get("action", "")
        url = task.payload.get("url", "")
        _logger.info("Browser automation requested: action=%s url=%s", action, url)
        # Clicking/typing/automating a live page needs a browser engine
        # (playwright/selenium), which is not installed.
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=False,
            message="Browser automation requires a browser engine (not installed).",
            data={"action": action, "url": url, "status": "unavailable"},
        )
