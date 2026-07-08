"""BrowserAgent — browser automation and web interaction.

Provides browser session management, tab control, navigation,
content extraction, screenshot capture, and download management
via Playwright with Selenium fallback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask

_logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright
    _HAS_PLAYWRIGHT = True
except ImportError:
    async_playwright = None
    _HAS_PLAYWRIGHT = False


class BrowserAgent(Agent):
    """Browser automation agent using Playwright with Selenium fallback.

    Manages browser sessions, tabs, navigation, content extraction,
    screenshots, and downloads.
    """

    def __init__(self) -> None:
        super().__init__(
            name="browser",
            supported_task_types=(
                "browser.navigate",
                "browser.screenshot",
                "browser.get_content",
                "browser.search",
                "browser.click",
                "browser.type",
                "browser.close",
            ),
        )
        self._playwright = None
        self._browser = None
        self._page = None
        self._current_url: str = ""

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"BrowserAgent cannot handle task type: {task.task_type}",
            )

        task_type = task.task_type

        try:
            if task_type == "browser.navigate":
                return await self._navigate(task)
            if task_type == "browser.screenshot":
                return await self._screenshot(task)
            if task_type == "browser.get_content":
                return await self._get_content(task)
            if task_type == "browser.search":
                return await self._search(task)
            if task_type == "browser.click":
                return await self._click(task)
            if task_type == "browser.type":
                return await self._type_text(task)
            if task_type == "browser.close":
                return await self._close(task)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Unknown browser task: {task_type}",
            )
        except Exception as exc:
            _logger.exception("Browser operation failed")
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=str(exc),
                data={"error": str(exc)},
            )

    async def _ensure_browser(self) -> None:
        if self._page is not None:
            return

        if not _HAS_PLAYWRIGHT:
            raise RuntimeError(
                "Playwright is not installed. "
                "Install with: pip install playwright && python -m playwright install chromium"
            )

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self._page = await context.new_page()

    async def _navigate(self, task: AgentTask) -> AgentResult:
        url = str(task.payload.get("url", ""))
        if not url:
            return AgentResult(
                agent_name=self.name, task_id=task.task_id,
                success=False, message="No URL provided",
            )

        await self._ensure_browser()
        await self._page.goto(url, wait_until="domcontentloaded")
        self._current_url = self._page.url
        title = await self._page.title()

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True,
            message=f"Navigated to {url}",
            data={"url": self._current_url, "title": title},
        )

    async def _screenshot(self, task: AgentTask) -> AgentResult:
        await self._ensure_browser()

        save_path = task.payload.get("save_path")
        if save_path:
            await self._page.screenshot(path=str(save_path), full_page=True)
            return AgentResult(
                agent_name=self.name, task_id=task.task_id,
                success=True, message=f"Screenshot saved to {save_path}",
                data={"path": str(save_path)},
            )

        import base64, io
        screenshot_bytes = await self._page.screenshot(full_page=True)
        b64 = base64.b64encode(screenshot_bytes).decode("ascii")
        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message="Screenshot captured",
            data={"base64": b64, "format": "png"},
        )

    async def _get_content(self, task: AgentTask) -> AgentResult:
        await self._ensure_browser()

        content_type = task.payload.get("type", "text")
        if content_type == "html":
            content = await self._page.content()
        elif content_type == "markdown":
            content = await self._page.evaluate("document.body.innerText")
        else:
            content = await self._page.evaluate("document.body.innerText")

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message="Content extracted",
            data={"content": content[:10000], "url": self._current_url, "length": len(content)},
        )

    async def _search(self, task: AgentTask) -> AgentResult:
        query = str(task.payload.get("query", ""))
        engine = task.payload.get("engine", "https://www.google.com/search?q=")

        if not query:
            return AgentResult(
                agent_name=self.name, task_id=task.task_id,
                success=False, message="No search query provided",
            )

        search_url = f"{engine}{query.replace(' ', '+')}"
        await self._ensure_browser()
        await self._page.goto(search_url, wait_until="domcontentloaded")
        self._current_url = self._page.url
        content = await self._page.evaluate("document.body.innerText")

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message=f"Search results for '{query}'",
            data={"query": query, "content": content[:5000], "url": self._current_url},
        )

    async def _click(self, task: AgentTask) -> AgentResult:
        await self._ensure_browser()

        selector = str(task.payload.get("selector", ""))
        if selector:
            await self._page.click(selector)
        else:
            x = task.payload.get("x")
            y = task.payload.get("y")
            if x is not None and y is not None:
                await self._page.click(f"body", position={"x": int(x), "y": int(y)})
            else:
                return AgentResult(
                    agent_name=self.name, task_id=task.task_id,
                    success=False, message="No selector or coordinates provided",
                )

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message="Click performed",
        )

    async def _type_text(self, task: AgentTask) -> AgentResult:
        await self._ensure_browser()

        text = str(task.payload.get("text", ""))
        selector = task.payload.get("selector")

        if selector:
            await self._page.fill(selector, text)
        else:
            await self._page.keyboard.type(text)

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message=f"Typed {len(text)} characters",
            data={"length": len(text)},
        )

    async def _close(self, task: AgentTask) -> AgentResult:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

        self._playwright = None
        self._browser = None
        self._page = None
        self._current_url = ""

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message="Browser closed",
        )

    async def stop(self) -> None:
        await self._close(AgentTask(task_type="browser.close"))
        await super().stop()
