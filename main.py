"""Jarvis application entry point (async) with memory system."""

import asyncio
import logging

from agents import MemoryAgent, PlannerAgent, ToolAgent, VoiceAgent
from agents.contracts import AgentTask
from config import configure_logging, load_settings
from llm import build_llm_provider
from memory import MemoryManager, MemoryService
from memory.document_store import SQLiteDocumentStore
from memory.vector_store import ChromaVectorStore
from orchestrator import Orchestrator


async def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Create and initialise an orchestrator with all Jarvis agents.

    Wires the memory system (MemoryManager + MemoryService) into the
    planner and memory agents when memory is enabled.  The MemoryManager
    is initialised here — once, at startup — so that agents never
    encounter an uninitialised backend at runtime.

    Accepts an optional *settings* object; if not provided settings are
    loaded from the environment.
    """
    settings = settings or load_settings()
    llm_provider = build_llm_provider(settings) if settings.llm_enabled else None

    memory_service: MemoryService | None = None
    if settings.memory_enabled:
        vector_store = ChromaVectorStore(path=settings.memory_vector_store_path)
        document_store = SQLiteDocumentStore(path=settings.memory_document_store_path)

        memory_manager = MemoryManager(
            vector_store=vector_store,
            document_store=document_store,
            dedup_threshold=settings.memory_dedup_threshold,
            importance_threshold=settings.memory_importance_threshold,
        )
        await memory_manager.initialize()
        memory_service = MemoryService(memory_manager)

    orchestrator = Orchestrator(
        agents=(
            PlannerAgent(
                llm_provider=llm_provider,
                memory_service=memory_service,
            ),
            MemoryAgent(memory_service=memory_service),
            ToolAgent(),
            VoiceAgent(),
        )
    )
    await orchestrator.initialize()
    await orchestrator.start()
    return orchestrator


async def main() -> None:
    """Start the Jarvis orchestration system (async REPL)."""
    settings = load_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    orchestrator = await build_orchestrator(settings)

    print("=" * 50)
    print("JARVIS AI ASSISTANT")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    try:
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
            print(result.data.get("response", result.message))

            enriched = result.data.get("memory_enriched", False)
            if enriched:
                count = result.data.get("memory_count", 0)
                logger.debug("Response used %d memory item(s)", count)
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
