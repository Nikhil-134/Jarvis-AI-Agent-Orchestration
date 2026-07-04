"""Jarvis application entry point."""

import logging

from agents import MemoryAgent, PlannerAgent, ToolAgent, VoiceAgent
from agents.contracts import AgentTask
from config import configure_logging, load_settings
from llm import build_llm_provider
from orchestrator import Orchestrator


def build_orchestrator() -> Orchestrator:
    """Create an orchestrator with all Jarvis agents registered."""
    settings = load_settings()
    llm_provider = build_llm_provider(settings) if settings.llm_enabled else None
    return Orchestrator(
        agents=(
            PlannerAgent(llm_provider=llm_provider),
            MemoryAgent(),
            ToolAgent(),
            VoiceAgent(),
        )
    )


def main() -> None:
    """Start the Jarvis orchestration system with a smoke task."""
    settings = load_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)
    orchestrator = build_orchestrator()
    result = orchestrator.route(AgentTask(task_type="plan", payload={"goal": "startup"}))
    logger.info("Startup route result: %s", result)


if __name__ == "__main__":
    main()
