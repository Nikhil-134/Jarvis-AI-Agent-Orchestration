"""Built-in tool implementations for Jarvis.

All built-in tools implement :class:`tools.interfaces.ITool` and can be
registered with a :class:`tools.registry.ToolRegistry` via the
:func:`register_all_builtins` helper.
"""

from __future__ import annotations

from tools.interfaces import IToolRegistry


def register_all_builtins(registry: IToolRegistry) -> None:
    """Discover and register all built-in tools."""
    from tools.builtins.base64_tool import Base64Tool
    from tools.builtins.browser_tool import BrowserTool
    from tools.builtins.calculator import CalculatorTool
    from tools.builtins.clipboard_tool import ClipboardTool
    from tools.builtins.datetime_tool import DateTimeTool
    from tools.builtins.file_system_tool import FileSystemTool
    from tools.builtins.hash_tool import HashTool
    from tools.builtins.json_tool import JsonTool
    from tools.builtins.notification_tool import NotificationTool
    from tools.builtins.screenshot_tool import ScreenshotTool
    from tools.builtins.shell_tool import ShellTool
    from tools.builtins.system_info import SystemInfoTool
    from tools.builtins.text_tool import TextTool
    from tools.builtins.uuid_tool import UuidTool

    tools = [
        Base64Tool(),
        BrowserTool(),
        CalculatorTool(),
        ClipboardTool(),
        DateTimeTool(),
        FileSystemTool(),
        HashTool(),
        JsonTool(),
        NotificationTool(),
        ScreenshotTool(),
        ShellTool(),
        SystemInfoTool(),
        TextTool(),
        UuidTool(),
    ]
    for tool in tools:
        registry.register(tool)
