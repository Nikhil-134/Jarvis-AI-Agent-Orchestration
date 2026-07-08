"""Agent capability definitions for Jarvis specialist agents.

Each agent declares its capabilities so the Prime Orchestrator can
intelligently route sub-tasks to the right specialist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Capability:
    """A discrete capability that an agent provides.

    Examples:
        Capability(name="web_search", description="Search the web for information")
        Capability(name="code_generation", description="Generate source code")
    """

    name: str
    description: str = ""
    required_tools: list[str] = field(default_factory=list)
    required_memory_types: list[str] = field(default_factory=list)


# ── Agent capability sets ────────────────────────────────────────────────

CAPABILITY_RESEARCH = Capability(
    name="research",
    description="Research topics, gather and synthesize information from multiple sources",
    required_tools=["web_search", "web_fetch"],
    required_memory_types=["fact", "conversation"],
)

CAPABILITY_CODE_ENGINEERING = Capability(
    name="code_engineering",
    description="Generate, review, refactor, and analyze source code",
    required_tools=["file_system", "shell", "text"],
    required_memory_types=["fact"],
)

CAPABILITY_VISION = Capability(
    name="vision",
    description="Analyze images, perform OCR, understand screenshots",
    required_tools=["screenshot", "ocr"],
)

CAPABILITY_SECURITY = Capability(
    name="security",
    description="Monitor system security, scan for threats, analyze logs",
    required_tools=["shell", "file_system", "system_info"],
)

CAPABILITY_STRATEGY = Capability(
    name="strategy",
    description="High-level planning, task decomposition, workflow design",
    required_tools=[],
    required_memory_types=["conversation", "summary"],
)

CAPABILITY_ENGINEERING = Capability(
    name="engineering",
    description="Build, compile, and deploy software projects",
    required_tools=["shell", "file_system"],
)

CAPABILITY_TESTING = Capability(
    name="testing",
    description="Write and execute tests, analyze coverage, validate behavior",
    required_tools=["shell", "file_system", "text"],
)

CAPABILITY_KNOWLEDGE = Capability(
    name="knowledge",
    description="Store, query, and manage knowledge base",
    required_tools=["memory"],
    required_memory_types=["fact", "preference"],
)

CAPABILITY_BROWSER = Capability(
    name="browser",
    description="Navigate web pages, scrape content, automate browser interactions",
    required_tools=["browser", "web_fetch"],
)

CAPABILITY_COMPUTATION = Capability(
    name="computation",
    description="Perform heavy computations, data processing, batch operations",
    required_tools=["shell", "calculator", "file_system"],
)

CAPABILITY_USER_EXPERIENCE = Capability(
    name="user_experience",
    description="Manage notifications, display information, handle user interaction",
    required_tools=["voice", "display"],
)

CAPABILITY_STORAGE = Capability(
    name="storage",
    description="Organize, backup, and manage file storage",
    required_tools=["file_system", "shell"],
)

CAPABILITY_DEVOPS = Capability(
    name="devops",
    description="Configure, deploy, and administer systems",
    required_tools=["shell", "file_system", "system_info"],
)
