"""Browser automation tool — stub implementation using Playwright or Selenium."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:
    import selenium.webdriver as webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False


class BrowserTool(ITool):
    """Automate a web browser (stub).

    Supports Playwright and Selenium backends.
    Operations: open_url, search, get_page_content, screenshot.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser",
            description="Automate a web browser. "
                        "Operations: open_url (navigate to a URL), "
                        "search (search the web), "
                        "get_page_content (retrieve page text), "
                        "screenshot (capture page screenshot as base64). "
                        "Requires playwright or selenium.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Browser operation to perform.",
                        "enum": ["open_url", "search", "get_page_content", "screenshot"],
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to "
                                       "(for open_url, get_page_content, screenshot).",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for search operation).",
                    },
                },
                "required": ["operation"],
            },
        )

    @property
    def category(self) -> str:
        return "web"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.DANGEROUS

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not HAS_PLAYWRIGHT and not HAS_SELENIUM:
            return {
                "success": False,
                "output": "",
                "error": "No browser automation library available. "
                         "Install playwright (pip install playwright) "
                         "or selenium (pip install selenium).",
            }

        operation: str = str(kwargs.get("operation", ""))
        handlers = {
            "open_url": self._open_url,
            "search": self._search,
            "get_page_content": self._get_page_content,
            "screenshot": self._screenshot,
        }

        handler = handlers.get(operation)
        if handler is None:
            return {
                "success": False,
                "output": "",
                "error": f"Unknown browser operation: {operation}. "
                         f"Must be one of: {', '.join(handlers)}",
            }

        return await handler(kwargs)

    def _backend(self) -> str:
        return "playwright" if HAS_PLAYWRIGHT else "selenium"

    async def _open_url(self, params: dict[str, Any]) -> dict[str, Any]:
        url: str = str(params.get("url", ""))
        if not url:
            return {"success": False, "output": "", "error": "No URL provided."}
        return await asyncio.to_thread(self._open_url_impl, url)

    async def _search(self, params: dict[str, Any]) -> dict[str, Any]:
        query: str = str(params.get("query", ""))
        if not query:
            return {"success": False, "output": "", "error": "No search query provided."}
        return await asyncio.to_thread(self._search_impl, query)

    async def _get_page_content(self, params: dict[str, Any]) -> dict[str, Any]:
        url: str = str(params.get("url", ""))
        if not url:
            return {"success": False, "output": "", "error": "No URL provided."}
        return await asyncio.to_thread(self._get_page_content_impl, url)

    async def _screenshot(self, params: dict[str, Any]) -> dict[str, Any]:
        url: str = str(params.get("url", ""))
        if not url:
            return {"success": False, "output": "", "error": "No URL provided."}
        return await asyncio.to_thread(self._screenshot_impl, url)

    # --- Shared sync implementations ---

    def _open_url_impl(self, url: str) -> dict[str, Any]:
        try:
            if self._backend() == "playwright":
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded")
                    title = page.title()
                    final_url = page.url
                    browser.close()
            else:
                opts = ChromeOptions()
                opts.add_argument("--headless")
                driver = webdriver.Chrome(options=opts)
                driver.get(url)
                title = driver.title
                final_url = driver.current_url
                driver.quit()
            return {
                "success": True,
                "output": f"Navigated to {url}",
                "data": {"title": title, "url": final_url},
            }
        except Exception as exc:
            return {"success": False, "output": "", "error": f"open_url failed: {exc}"}

    def _search_impl(self, query: str) -> dict[str, Any]:
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        try:
            if self._backend() == "playwright":
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(search_url, wait_until="domcontentloaded")
                    title = page.title()
                    final_url = page.url
                    browser.close()
            else:
                opts = ChromeOptions()
                opts.add_argument("--headless")
                driver = webdriver.Chrome(options=opts)
                driver.get(search_url)
                title = driver.title
                final_url = driver.current_url
                driver.quit()
            return {
                "success": True,
                "output": f"Searched for: {query}",
                "data": {"title": title, "url": final_url, "query": query},
            }
        except Exception as exc:
            return {"success": False, "output": "", "error": f"search failed: {exc}"}

    def _get_page_content_impl(self, url: str) -> dict[str, Any]:
        try:
            if self._backend() == "playwright":
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded")
                    title = page.title()
                    content = page.inner_text("body")
                    browser.close()
            else:
                opts = ChromeOptions()
                opts.add_argument("--headless")
                driver = webdriver.Chrome(options=opts)
                driver.get(url)
                title = driver.title
                content = driver.find_element("tag name", "body").text
                driver.quit()
            truncated = len(content) > 10000
            return {
                "success": True,
                "output": content[:10000],
                "data": {"title": title, "url": url, "truncated": truncated},
            }
        except Exception as exc:
            return {
                "success": False,
                "output": "",
                "error": f"get_page_content failed: {exc}",
            }

    def _screenshot_impl(self, url: str) -> dict[str, Any]:
        try:
            if self._backend() == "playwright":
                import base64
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded")
                    data = page.screenshot(full_page=True, type="png")
                    b64 = base64.b64encode(data).decode("utf-8")
                    browser.close()
            else:
                opts = ChromeOptions()
                opts.add_argument("--headless")
                driver = webdriver.Chrome(options=opts)
                driver.get(url)
                b64 = driver.get_screenshot_as_base64()
                driver.quit()
            return {
                "success": True,
                "output": b64,
                "data": {"format": "base64", "mime_type": "image/png", "url": url},
            }
        except Exception as exc:
            return {"success": False, "output": "", "error": f"screenshot failed: {exc}"}
