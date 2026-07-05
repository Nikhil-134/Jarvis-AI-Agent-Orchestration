"""Jarvis application entry point (async)."""

import asyncio
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


async def main() -> None:
    """Start the Jarvis orchestration system (async REPL)."""
    settings = load_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    orchestrator = build_orchestrator()

    print("=" * 50)
    print("JARVIS AI ASSISTANT")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        goal = input("\nYou: ").strip()

        if goal.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break

        if not goal:
            continue

        result = await orchestrator.route(
            AgentTask(
                task_type="plan",
                payload={"goal": goal},
            )
        )

        print("\nJarvis:\n")
        print(result.data.get("plan", result.message))


if __name__ == "__main__":
    asyncio.run(main())
