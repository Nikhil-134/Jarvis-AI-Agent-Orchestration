"""DesktopAgent — desktop automation and intelligence.

Provides mouse control, keyboard control, window management,
clipboard operations, application launching, screenshots, OCR,
and file explorer interaction via the existing tool system.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask

_logger = logging.getLogger(__name__)

try:
    import pyautogui
    _HAS_PYAUTOGUI = True
except ImportError:
    pyautogui = None
    _HAS_PYAUTOGUI = False

try:
    import pyperclip
    _HAS_PYPERCLIP = True
except ImportError:
    pyperclip = None
    _HAS_PYPERCLIP = False

try:
    from PIL import ImageGrab
    _HAS_PIL = True
except ImportError:
    ImageGrab = None
    _HAS_PIL = False


class DesktopAgent(Agent):
    """Desktop automation agent.

    Supports mouse control, keyboard input, window management,
    clipboard operations, screenshot capture, OCR, and app launching.
    """

    def __init__(self) -> None:
        super().__init__(
            name="desktop",
            supported_task_types=(
                "desktop.mouse.click",
                "desktop.mouse.move",
                "desktop.keyboard.type",
                "desktop.keyboard.hotkey",
                "desktop.screenshot",
                "desktop.clipboard.read",
                "desktop.clipboard.write",
                "desktop.window.list",
                "desktop.launch",
                "desktop.ocr",
            ),
        )

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"DesktopAgent cannot handle task type: {task.task_type}",
            )

        task_type = task.task_type
        payload = task.payload

        try:
            if task_type == "desktop.mouse.click":
                return await self._mouse_click(task)
            if task_type == "desktop.mouse.move":
                return await self._mouse_move(task)
            if task_type == "desktop.keyboard.type":
                return await self._keyboard_type(task)
            if task_type == "desktop.keyboard.hotkey":
                return await self._keyboard_hotkey(task)
            if task_type == "desktop.screenshot":
                return await self._screenshot(task)
            if task_type == "desktop.clipboard.read":
                return await self._clipboard_read(task)
            if task_type == "desktop.clipboard.write":
                return await self._clipboard_write(task)
            if task_type == "desktop.launch":
                return await self._launch_app(task)
            if task_type == "desktop.ocr":
                return await self._ocr(task)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Unknown desktop task: {task_type}",
            )
        except Exception as exc:
            _logger.exception("Desktop operation failed")
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=str(exc),
                data={"error": str(exc)},
            )

    async def _mouse_click(self, task: AgentTask) -> AgentResult:
        if not _HAS_PYAUTOGUI:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="pyautogui not installed")

        x = task.payload.get("x")
        y = task.payload.get("y")
        button = task.payload.get("button", "left")
        clicks = int(task.payload.get("clicks", 1))

        if x is not None and y is not None:
            pyautogui.click(int(x), int(y), button=button, clicks=clicks)
        else:
            pyautogui.click(button=button, clicks=clicks)

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Clicked at ({x}, {y})", data={"x": x, "y": y, "button": button})

    async def _mouse_move(self, task: AgentTask) -> AgentResult:
        if not _HAS_PYAUTOGUI:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="pyautogui not installed")

        x = int(task.payload.get("x", 0))
        y = int(task.payload.get("y", 0))
        duration = float(task.payload.get("duration", 0.25))
        pyautogui.moveTo(x, y, duration=duration)

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Moved to ({x}, {y})", data={"x": x, "y": y})

    async def _keyboard_type(self, task: AgentTask) -> AgentResult:
        if not _HAS_PYAUTOGUI:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="pyautogui not installed")

        text = str(task.payload.get("text", ""))
        interval = float(task.payload.get("interval", 0.05))
        pyautogui.write(text, interval=interval)

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Typed {len(text)} characters", data={"length": len(text)})

    async def _keyboard_hotkey(self, task: AgentTask) -> AgentResult:
        if not _HAS_PYAUTOGUI:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="pyautogui not installed")

        keys = task.payload.get("keys", [])
        if isinstance(keys, str):
            keys = [keys]
        pyautogui.hotkey(*keys)

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Pressed hotkey: {'+'.join(keys)}", data={"keys": keys})

    async def _screenshot(self, task: AgentTask) -> AgentResult:
        if not _HAS_PIL:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="PIL not installed")

        save_path = task.payload.get("save_path")
        screenshot = ImageGrab.grab()

        if save_path:
            screenshot.save(str(save_path))
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Screenshot saved to {save_path}", data={"path": str(save_path), "size": screenshot.size})

        import base64, io
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Screenshot captured", data={"format": "png", "base64": b64, "width": screenshot.width, "height": screenshot.height})

    async def _clipboard_read(self, task: AgentTask) -> AgentResult:
        if not _HAS_PYPERCLIP:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="pyperclip not installed")

        text = pyperclip.paste()
        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Clipboard read", data={"text": text, "length": len(text)})

    async def _clipboard_write(self, task: AgentTask) -> AgentResult:
        if not _HAS_PYPERCLIP:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="pyperclip not installed")

        text = str(task.payload.get("text", ""))
        pyperclip.copy(text)
        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Copied {len(text)} characters to clipboard", data={"length": len(text)})

    async def _launch_app(self, task: AgentTask) -> AgentResult:
        import subprocess
        app = str(task.payload.get("app", ""))
        args = task.payload.get("args", "")

        if not app:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="No app specified")

        try:
            if args:
                subprocess.Popen([app] + str(args).split())
            else:
                subprocess.Popen([app])
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Launched {app}", data={"app": app})
        except Exception as exc:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=f"Failed to launch {app}: {exc}", data={"error": str(exc)})

    async def _ocr(self, task: AgentTask) -> AgentResult:
        image_path = task.payload.get("image_path")
        try:
            import pytesseract
            from PIL import Image

            if image_path:
                img = Image.open(str(image_path))
            else:
                img = ImageGrab.grab()

            text = pytesseract.image_to_string(img)
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="OCR completed", data={"text": text.strip(), "length": len(text.strip())})
        except ImportError:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="pytesseract not installed")
        except Exception as exc:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=f"OCR failed: {exc}", data={"error": str(exc)})
