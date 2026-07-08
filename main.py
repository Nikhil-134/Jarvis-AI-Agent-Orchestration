"""Jarvis application entry point — uses Runtime layer for all processing."""

from __future__ import annotations

import asyncio
import logging
import sys

# Windows terminals default to legacy code pages (e.g. cp1252) that cannot
# encode characters Jarvis routinely emits (×, —, emoji). Without this, a
# single such character raises UnicodeEncodeError and kills the REPL.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from config import configure_logging, load_settings
from config.settings import Settings
from memory import MemoryManager, MemoryService, PersistentMemoryService
from memory.reflection import ReflectionEngine
from memory.document_store import SQLiteDocumentStore
from memory.vector_store import ChromaVectorStore
from orchestrator import Orchestrator
from tools import (
    PermissionManager,
    ToolExecutionEngine,
    ToolManager,
    ToolRegistry,
)
from prompt import InputReader, InputMode
from tools.builtins import register_all_builtins

from agents import (
    AthenaAgent,
    BrowserAgent,
    CalendarAgent,
    DesktopAgent,
    EmailAgent,
    FridayAgent,
    GeckoAgent,
    HerculesAgent,
    HulkAgent,
    JarvisPrimeAgent,
    JeromeAgent,
    MemoryAgent,
    NotesAgent,
    OracleAgent,
    PepperAgent,
    PlannerAgent,
    ReminderAgent,
    ResponseComposer,
    StarkAgent,
    SteveAgent,
    ToolAgent,
    UltronAgent,
    VeronicaAgent,
    VisionAgent,
    VoiceAgent,
)
from llm import build_llm_provider
from runtime import Runtime


async def _build_confirmation_callback(settings: Settings):
    if settings.tool_auto_approve:
        async def _auto_approve(name: str, reason: str) -> bool:
            return True
        return _auto_approve

    async def _confirm(name: str, reason: str) -> bool:
        print(f"\n[Permission Required] Tool '{name}' requires confirmation.")
        print(f"  Reason: {reason}")
        response = input("  Allow execution? (y/N): ").strip().lower()
        return response in ("y", "yes")
    return _confirm


def _build_voice_providers(settings: Settings):
    """Construct local voice providers from settings.

    Returns ``(tts, stt, audio)``. Any component may be ``None`` / unavailable;
    callers must check each provider's ``available`` property. Import failures
    (missing native deps) degrade to ``None`` rather than crashing startup.
    """
    if not settings.voice_enabled:
        return None, None, None

    tts = stt = audio = None
    try:
        from voice import AudioIO, PiperTTSProvider, WhisperSTTProvider
        from voice.edge_tts import EdgeTTSProvider

        audio = AudioIO()
        if settings.voice_tts_provider == "edge":
            # Online fallback — only if the user explicitly opts out of local.
            tts = EdgeTTSProvider()
        else:
            tts = PiperTTSProvider(settings.voice_piper_model)
        stt = WhisperSTTProvider(settings.voice_stt_model)
    except Exception:  # pragma: no cover - defensive; missing optional deps
        logging.getLogger(__name__).exception("Voice provider initialisation failed")

    return tts, stt, audio


def _build_tool_system(settings: Settings):
    registry = ToolRegistry()
    register_all_builtins(registry)

    if settings.tool_enabled:
        from tools.discovery import discover_plugins
        plugin_dirs = [p.strip() for p in settings.tool_plugin_dirs.split(",") if p.strip()]
        discover_plugins(registry, plugin_dirs=plugin_dirs)

    enabled_str = settings.tool_enabled_tools.strip()
    disabled_str = settings.tool_disabled_tools.strip()
    enabled_set = set(t.strip() for t in enabled_str.split(",") if t.strip()) if enabled_str else None
    disabled_set = set(t.strip() for t in disabled_str.split(",") if t.strip()) if disabled_str else set()

    manager = ToolManager(
        registry=registry,
        enabled_tools=enabled_set,
        disabled_tools=disabled_set,
        default_timeout=settings.tool_default_timeout,
    )
    return manager, manager.engine


def _build_internet_service(settings: Settings):
    """Build the shared Internet Knowledge Engine, or None when disabled.

    Fail-safe: any construction error logs and yields None so the app still
    boots and runs local-only.
    """
    if not getattr(settings, "internet_enabled", True):
        return None
    try:
        from knowledge.internet import build_internet_service

        return build_internet_service(
            enabled=True,
            timeout=settings.internet_timeout_seconds,
            overall_timeout=settings.internet_overall_timeout_seconds,
            cache_ttl_seconds=settings.internet_cache_ttl_seconds,
            min_interval_seconds=settings.internet_min_interval_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - never block startup
        logging.getLogger(__name__).warning("Internet service unavailable: %s", exc)
        return None


async def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Create and initialise an orchestrator with all Jarvis agents."""
    settings = settings or load_settings()
    llm_provider = build_llm_provider(settings) if settings.llm_enabled else None

    memory_service: MemoryService | None = None
    persistent_memory: PersistentMemoryService | None = None
    if settings.memory_enabled:
        vector_store = ChromaVectorStore(path=settings.memory_vector_store_path)
        document_store = SQLiteDocumentStore(path=settings.memory_document_store_path)
        memory_manager = MemoryManager(
            vector_store=vector_store, document_store=document_store,
            dedup_threshold=settings.memory_dedup_threshold,
            importance_threshold=settings.memory_importance_threshold,
        )
        await memory_manager.initialize()
        memory_service = MemoryService(memory_manager)
        # Cross-session persistent layer (sessions/projects/profile/reflection)
        # built on the SAME manager — recording a turn here is what the
        # KnowledgeEngine later recalls semantically. Reflection uses the local
        # LLM when available, falling back to the deterministic heuristic.
        if settings.memory_persist_enabled:
            reflection_engine = ReflectionEngine(llm_provider) if llm_provider else ReflectionEngine()
            persistent_memory = PersistentMemoryService(
                memory_manager, reflection_engine=reflection_engine,
            )

    tool_manager: ToolManager | None = None
    tool_engine: ToolExecutionEngine | None = None
    tool_agent: ToolAgent | None = None
    if settings.tool_enabled:
        confirm_cb = await _build_confirmation_callback(settings)
        perm_mgr = PermissionManager(confirmation_callback=confirm_cb)
        mgr, eng = _build_tool_system(settings)
        eng._permission_manager = perm_mgr
        mgr._permission_manager = perm_mgr
        tool_manager = mgr
        tool_engine = eng
        tool_agent = ToolAgent(
            engine=eng, auto_register_builtins=False,
            memory_service=memory_service if settings.memory_enabled else None,
            store_results=settings.tool_store_results,
        )

    planner_agent = PlannerAgent(
        llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine,
        max_context_tokens=settings.prompt_max_context_tokens,
        max_chunk_tokens=settings.prompt_max_chunk_tokens,
        chars_per_token=settings.prompt_chars_per_token,
    )

    response_composer = ResponseComposer(llm_provider=llm_provider)

    tts_provider, stt_provider, audio_io = _build_voice_providers(settings)

    all_agents = (
        planner_agent,
        FridayAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        VeronicaAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        VisionAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        UltronAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        AthenaAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        StarkAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        SteveAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        OracleAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        GeckoAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        HerculesAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        PepperAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        HulkAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        JeromeAgent(llm_provider=llm_provider, memory_service=memory_service, tool_engine=tool_engine),
        MemoryAgent(memory_service=memory_service),
        tool_agent or ToolAgent(auto_register_builtins=True),
        VoiceAgent(tts=tts_provider, stt=stt_provider, audio=audio_io),
        DesktopAgent(), BrowserAgent(),
        NotesAgent(), CalendarAgent(), ReminderAgent(), EmailAgent(),
    )

    orchestrator = Orchestrator(agents=all_agents)
    if tool_manager is not None:
        orchestrator.tool_manager = tool_manager
    # Stash voice providers so the REPL can assemble a VoicePipeline.
    orchestrator.voice_providers = (tts_provider, stt_provider, audio_io)
    # Stash the persistent-memory service so the Runtime and REPL can wire in
    # per-turn recording, boot-time session restore, and reflect-on-exit.
    orchestrator.persistent_memory = persistent_memory
    # Compose the Internet Knowledge Engine once (shared cache + rate limiter).
    # Context-only, last-resort retrieval; None when disabled or unavailable.
    orchestrator.internet_service = _build_internet_service(settings)
    # Stash the tool engine so the Runtime can wire the Planning subsystem's
    # ToolInvoker to the SAME engine (shared registry/permissions). The Planning
    # coordinator itself is composed in the Runtime (it needs the LLMGuard +
    # KnowledgeEngine that live there); this just shares the tool backend.
    orchestrator.tool_engine = tool_engine

    jarvis_agent = JarvisPrimeAgent(
        llm_provider=llm_provider, memory_service=memory_service,
        workflow_engine=orchestrator.workflow_engine,
        planner_agent=planner_agent, response_composer=response_composer,
        max_context_tokens=settings.prompt_max_context_tokens,
        max_chunk_tokens=settings.prompt_max_chunk_tokens,
        chars_per_token=settings.prompt_chars_per_token,
    )
    jarvis_agent.bind_runtime(orchestrator.context, orchestrator.event_bus)
    orchestrator._agents["jarvis"] = jarvis_agent

    await orchestrator.initialize()
    await orchestrator.start()
    return orchestrator


async def main() -> None:
    """Start the Jarvis AI Operating System (async REPL).

    All user input flows through the Runtime layer.
    """
    settings = load_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    orchestrator = await build_orchestrator(settings)
    runtime = Runtime(orchestrator=orchestrator, settings=settings)

    # Restore the previous conversation from durable memory so JARVIS continues
    # exactly where it left off after a restart. Best-effort: a restore failure
    # must not stop the app, but it is logged (no hidden failures).
    persistent_memory = getattr(orchestrator, "persistent_memory", None)
    if persistent_memory is not None:
        try:
            restored = await persistent_memory.restore_session(settings.memory_session_id)
            if restored:
                logger.info("Restored %d turn(s) from previous session '%s'",
                            len(restored), settings.memory_session_id)
        except Exception:
            logger.exception("Failed to restore session '%s'", settings.memory_session_id)

    # Assemble the local voice pipeline (if enabled and providers came up).
    voice_pipeline = None
    tts_provider, stt_provider, audio_io = getattr(
        orchestrator, "voice_providers", (None, None, None)
    )
    if settings.voice_enabled and tts_provider is not None:
        from voice import VoicePipeline
        voice_pipeline = VoicePipeline(
            brain=runtime, stt=stt_provider, tts=tts_provider, audio=audio_io,
            record_seconds=settings.voice_record_seconds,
        )

    def _can_converse() -> bool:
        return (
            voice_pipeline is not None
            and audio_io is not None and audio_io.available
            and stt_provider is not None and getattr(stt_provider, "available", False)
            and tts_provider is not None and getattr(tts_provider, "available", False)
        )

    async def _run_continuous_loop() -> None:
        """Run the hands-free continuous conversation loop until it ends."""
        from voice import ContinuousVoiceLoop, EndpointConfig, LoopConfig

        cfg = LoopConfig(
            wake_mode=settings.voice_wake_mode,
            wake_words=tuple(w.strip() for w in settings.voice_wake_words.split(",") if w.strip()),
            frame_ms=settings.voice_frame_ms,
            inactivity_timeout=settings.voice_inactivity_timeout,
            greeting=settings.voice_greeting,
            endpoint=EndpointConfig(
                trailing_silence=settings.voice_trailing_silence,
                max_utterance=settings.voice_max_utterance,
            ),
        )

        def _on_event(kind: str, text: str) -> None:
            labels = {
                "state": "  [%s]" % text,
                "heard": "\nYou (voice): %s" % text,
                "response": "\nJarvis:\n%s" % text,
            }
            print(labels.get(kind, f"  [{kind}] {text}"))

        loop = ContinuousVoiceLoop(
            brain=runtime, stt=stt_provider, tts=tts_provider, audio=audio_io,
            frame_source=lambda: audio_io.stream_frames(frame_ms=settings.voice_frame_ms),
            config=cfg, capture_rate=audio_io.capture_rate, on_event=_on_event,
        )
        wake_hint = (
            f"say '{cfg.wake_words[0]}' to wake"
            if settings.voice_wake_mode != "none" else "always listening"
        )
        print(f"\n  [continuous voice active — {wake_hint}; say 'goodbye' or Ctrl+C to stop]")
        try:
            await loop.run()
        except KeyboardInterrupt:
            loop.stop()
        print("  [continuous voice stopped]")

    llm_status = f"LLM:{settings.llm_provider}/{settings.llm_model}" if settings.llm_enabled else "LLM:off"
    memory_status = f"Memory:{settings.memory_vector_store_path}" if settings.memory_enabled else "Memory:off"
    tools_status = f"Tools:{settings.tool_enabled_tools or 'all'}" if settings.tool_enabled else "Tools:off"
    if voice_pipeline is not None:
        voice_status = (
            f"Voice:{settings.voice_tts_provider}"
            f"(speak={'on' if voice_pipeline.can_speak else 'off'},"
            f"listen={'on' if voice_pipeline.can_listen else 'off'})"
        )
    else:
        voice_status = "Voice:off"

    print("=" * 50)
    print("JARVIS AI OPERATING SYSTEM")
    print(f"  {llm_status} | {memory_status} | {tools_status}")
    print(f"  {voice_status}")
    print("-" * 50)
    print("  !file <path> | !paste | /tool <name> | !speak <text>")
    print("  !voice (one turn) | !converse (hands-free loop) | exit/quit")
    print("=" * 50)

    input_reader = InputReader()

    # Optionally launch straight into the hands-free loop.
    if settings.voice_autostart and _can_converse():
        await _run_continuous_loop()

    try:
        while True:
            raw, mode = await input_reader.read_interactive()

            if raw.lower() in ("exit", "quit"):
                print("\nGoodbye!")
                break

            if not raw:
                continue

            if mode in (InputMode.FILE, InputMode.PASTE):
                kb = len(raw) / 1024
                est_tokens = len(raw) // 4
                print(f"  [input: {kb:.1f} KB, ~{est_tokens} tokens]")

            if raw.startswith("!speak "):
                text = raw[len("!speak "):].strip()
                if voice_pipeline is None or not voice_pipeline.can_speak:
                    print("  [voice output unavailable]")
                    continue
                spoke = await voice_pipeline.speak(text)
                print(f"  [spoke: {spoke}]")
                continue

            if raw.strip() == "!voice":
                if voice_pipeline is None or not voice_pipeline.can_listen:
                    print("  [voice input unavailable — check mic and faster-whisper]")
                    continue
                print("  [listening...]")
                turn = await voice_pipeline.listen_and_respond()
                if not turn.transcript:
                    print("  [heard nothing]")
                    continue
                print(f"\nYou (voice): {turn.transcript}")
                print(f"\nJarvis:\n{turn.response}")
                continue

            if raw.strip() == "!converse":
                if not _can_converse():
                    print("  [continuous voice unavailable — need mic, faster-whisper and Piper]")
                    continue
                await _run_continuous_loop()
                continue

            if raw.startswith("/tool "):
                parts = raw[len("/tool "):].strip().split()
                if not parts:
                    print("Usage: /tool <name> [key=value ...]")
                    continue
                tool_name = parts[0]
                tool_kwargs: dict[str, str] = {}
                for arg in parts[1:]:
                    if "=" in arg:
                        k, v = arg.split("=", 1)
                        tool_kwargs[k] = v
                result = await runtime.conversation.tool_executor.execute(tool_name, tool_kwargs)
                print(f"\nJarvis:\n{result}")
                continue

            result = await runtime.run(raw)
            print(f"\nJarvis:\n{result}")

    finally:
        # Reflect on the session before shutting down — distil decisions, tasks,
        # and lessons into durable memory so JARVIS gets smarter without
        # retraining. Best-effort and logged; never blocks a clean shutdown.
        if persistent_memory is not None:
            try:
                insights = await persistent_memory.reflect_on_session(settings.memory_session_id)
                if insights:
                    logger.info("Reflection stored %d insight(s) for session '%s'",
                                len(insights), settings.memory_session_id)
            except Exception:
                logger.exception("Reflection failed for session '%s'", settings.memory_session_id)
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
