# PROJECT_BRAIN.md — JARVIS AI Operating System

> **Single source of truth for this repository.**
> New sessions: read THIS FILE ONLY, then read only the files relevant to the
> requested task. Do NOT re-audit the whole repo unless this file is missing,
> corrupted, clearly outdated, or a full review is explicitly requested.
>
> **Last updated:** 2026-07-08 (cycle 9 — Runtime Intelligence: calculator
> reliability + automatic preference learning)
> **Maintainer role:** Principal AI Systems Architect (permanent)
>
> **Status: Cycle 9 shipped two targeted runtime-intelligence gap-closures
> (1034 offline tests green, +70): (1) the deterministic math engine now
> supports `pow()` and companions, keeps large integers exact, refuses
> pathological inputs, and never leaks a raw library error — and math phrasings
> like `pow(2,10)` now route to the calculator instead of the LLM; (2) roadmap
> #11 is DONE — preferences stated in ordinary chat ("call me Boss", "my
> favourite language is Rust") auto-promote to the structured profile with no
> "remember" command. Cycle 8's Planning & Task Execution subsystem remains
> complete. Roadmap #10 (knowledge graph) still NOT started — deliberately
> deferred. NOTE: cycles 6–8 already implemented most of the generic
> "production runtime" wishlist (continuous voice loop, parallel internet
> fan-out, reflection/summarization, persistent memory, pronoun resolution),
> so cycle 9 extended the two genuine remaining gaps rather than rebuilding.**

---

## 1. Vision

Build a **real, local-first AI Operating System** named JARVIS — not a chatbot.
Modular, multi-agent, voice-capable, with memory, tools, and desktop/browser
reach. **Hard constraint: ₹0 running cost. Everything runs locally. No paid
APIs** (no OpenAI/Anthropic/Gemini/Azure/Pinecone-cloud). Target platform:
Windows 11 + Python + VS Code. Intended to be open-sourced, GitHub-quality.

---

## 2. Environment (verified 2026-07-07)

- **Repo root:** `/f/AI_Agent_Orchestration` (Windows drive F:)
- **Python:** 3.14.6, venv at `.venv/` → interpreter `.venv/Scripts/python`
- **LLM runtime:** Ollama running at `http://localhost:11434`
  - Models present: `qwen2.5-coder:latest` (7.6B), `qwen2.5-coder:3b`
  - Active model (`.env`): `qwen2.5-coder:3b`
  - ⚠️ These are *coder* models — competent but not ideal for warm general
    chat. Swapping in a general model (e.g. `qwen2.5:3b`, `llama3.2`) is an
    open improvement, not a blocker.
- **Installed Python libs that matter:** chromadb, httpx, onnxruntime, numpy,
  requests, **piper-tts**, **faster-whisper**, **sounddevice**.
- **NOT installed** (subsystems depending on these are inert): playwright,
  selenium, pyautogui, pyperclip, pytesseract, PIL/Pillow, openwakeword,
  pypdf, psutil, fastapi, pvporcupine, edge_tts.

### Console gotcha (Windows)
Windows terminals default to cp1252 and crash on `×`, `—`, emoji. `main.py`
now reconfigures stdout/stderr to UTF-8 at startup. When running one-off
scripts, set `PYTHONIOENCODING=utf-8` (and `HF_HUB_DISABLE_SYMLINKS_WARNING=1`
to quiet a faster-whisper cache warning).

---

## 3. Architecture (high level)

Clean-ish layered design. Data flows:

```
User (text or voice)
   │
   ├─ voice/ (optional)  wake→STT→ text ─┐
   │                                     ▼
main.py REPL ───────────────► runtime.Runtime.run(text)
                                         │
                          runtime.ConversationRuntime._pipeline
                                         │
   1. PersonalityEngine   (fast canned greetings/small-talk)
   2. ContextManager      (pronoun/follow-up enrichment)
   3. IntentEngine        (regex/keyword classification → IntentResult)
   3b. KnowledgeEngine ◄── knowledge/chat/unknown/low-confidence go DIRECT
                           to LLM (LLMGuard). Bypasses planner + specialist
                           workflow. THIS is where normal conversation
                           intelligence comes from. Now also injects (cycle 6)
                           local memory context AND — only if the freshness
                           router says the query is time-sensitive — a live
                           internet context block (context-only; Qwen reasons).
   3c. PlanningCoordinator ◄─ NEW cycle 8. ONLY for actionable + multi-step
                           (or explicit heavy-planning) goals. Decomposes →
                           executes a dependency task graph (parallel, retries,
                           timeout, cancel) → verifies → returns. On
                           decline/low-confidence it FALLS THROUGH to step 4
                           (never dead-ends). Simple chat / single-tool / current
                           _info all skip this and stay on their proven paths.
   4. RoutingEngine       (tool/vision/browser/plan intents → Orchestrator)
   5. Response synthesis/formatting
   6. ContextManager.update (store turn)
                                         │
                                         ▼
                    orchestrator.Orchestrator.route(AgentTask)
                     → first agent whose can_handle() matches
                       (first-match by registration order)
                                         │
                    JarvisPrimeAgent / specialist agents / ToolAgent
                                         │
                    LLM (llm/ollama_provider) · Memory (chromadb) · Tools
```

**Persistent memory (cross-session) — NEW 2026-07-07:**
`memory/persistent_memory.PersistentMemoryService` is a thin DI layer **on top of**
the existing `MemoryManager` (no new backend). Durability is already provided by
the manager's SQLite document store + ChromaDB vector store; this layer adds
*structure* so JARVIS continues exactly where it left off after a restart:
- **Sessions** — `record_turn(session_id, …)` tags every turn; `restore_session`
  reloads a session chronologically into working memory; `recent_turns` /
  `session_transcript` answer "what did we discuss".
- **Projects** — `remember_project` / `get_project` / `list_projects` /
  `update_project_status` (deterministic `project:<id>` doc id = upsert).
- **User profile** — `set_preference` / `get_preference` / `get_profile`
  (`pref:<key>` = upsert; latest wins). Answers "favourite language" etc.
- **Reflection** — `reflect_on_session` runs `ReflectionEngine` (LLM→strict JSON,
  deterministic heuristic fallback, never fabricated) and stores
  summary/decisions/tasks/lessons as typed memories.
- **Classification + gate** — `classify()` + `is_meaningful()` so only meaningful
  info is stored (not everything). Reuses the proven store-worthiness gate.
- **Injection-safe recall** — `build_context()` wraps retrieved memories in a
  delimited "untrusted reference data" block (memory-poisoning mitigation).
- **Restart verified** with real ChromaDB + SQLite + ONNX embedding: turns,
  project, preference, and reflection all survived a fresh manager on the same
  data dir; semantic recall returned the right memory.
- **AUTO-WIRED into the runtime (cycle 5, 2026-07-07):** `build_orchestrator`
  constructs one `PersistentMemoryService` (LLM-backed `ReflectionEngine`) and
  stashes it on `orchestrator.persistent_memory`. `Runtime` extracts it and
  passes it + `settings.memory_session_id` into `ConversationRuntime`, which
  calls `record_turn` after BOTH return paths (knowledge fast-path and routing)
  via a DRY `_persist_turn` helper (best-effort, logged, never breaks the
  reply). `main()` calls `restore_session` on boot and `reflect_on_session` in
  the `finally` on exit. Gated by `MEMORY_PERSIST_ENABLED` (default true).
  **End-to-end restart proof (real Runtime, two boots):** a live turn was
  auto-persisted in boot 1 and restored from disk by a fresh orchestrator in
  boot 2.

**Internet Knowledge Engine (roadmap #9) — NEW cycle 6, 2026-07-07:**
`knowledge/internet/` is a lightweight, **context-only, last-resort** retrieval
layer. It supplies fresh public facts (weather/news/current events/recent
releases/office-holders) to the local model as *temporary context* — it never
reasons, never stores durably, and only activates when local sources can't
answer. **Reasoning stays 100% inside Qwen.**
- **Priority ladder (enforced):** 1 persistent memory → 2 current conversation
  → 3 local KB → 4 local docs → **5 internet (only if required)** → 6 Qwen
  reasoning. The `router.needs_internet()` gate returns True *only* for
  time-sensitive queries; personal/memory queries (`is_memory_query`) and
  timeless "what is/explain" questions never leave the machine.
- **Providers (plugin, DI):** `DuckDuckGoProvider` (Instant Answer API) +
  `WikipediaProvider` (action API `extracts`). Public, **key-free JSON only —
  no scraping, no Selenium/Playwright, no HTML parsing.**
- **Security boundary (`SafeHttpClient`):** HTTPS-only, **domain whitelist**
  (`api.duckduckgo.com`, `*.wikipedia.org`) checked before any socket → **no
  SSRF / no arbitrary fetch**; **no redirect following** (SSRF pivot blocked);
  bounded timeout + retries; **512 KiB streamed size cap**; localhost/127.0.0.1
  rejected. Fail-safe: any error/timeout → `[]` → local-only answer.
- **Privacy:** only the *current question* is sent outward — never history,
  memory, secrets, or internal prompts.
- **Performance:** async parallel provider fan-out under one overall timeout,
  short-TTL cache (default 300 s), per-service rate limit (min interval),
  cancellation on timeout. Runtime never blocks.
- **Injection-safe:** retrieved snippets wrapped in a delimited "untrusted
  external data" block (mirrors the memory layer's mitigation).
- **Verified live** (Wi-Fi): "Who is the PM of India?" → Narendra Modi (fetched);
  "What is Python?" → local, no fetch; "What do I like?" → offline, memory recall.

**Planning & Task Execution subsystem (roadmap #12 / cycle 8) — NEW 2026-07-08:**
`planning/` is an enterprise-grade layer that decomposes complex, *actionable*
goals into a dependency-aware task graph, executes it concurrently, and verifies
the answer before it reaches the user. It **orchestrates the existing
subsystems** (Memory, InternetKnowledgeService, KnowledgeEngine, ToolEngine,
LLMGuard) — it does **not** reimplement any of them.
- **Activation is deliberately narrow** (`ConversationRuntime._should_plan`):
  only goals that are `is_actionable` AND (multi-step OR explicitly
  heavy-planning). Plain chat, knowledge questions, greetings, single-tool
  requests, `current_info`, and vision all keep their existing fast paths. This
  preserves the cycle-7 stabilized memory-first / local-first behaviour exactly.
- **Confidence-based routing (supersedes regex *for these goals only*):** the
  planner assigns each task a `required_tool` capability; the `CapabilityCatalog`
  resolves it to a **real** backend and reports live availability; the
  `ToolInvoker` delegates. Regex routing (step 4) stays as the fallback and for
  everything else.
- **Fail-safe / backward compatible:** the coordinator, executor, invoker,
  planner, and verifier **never raise** — every failure is a typed result. On
  empty/low-confidence/all-failed plans the `PlanningOutcome.accepted=False` and
  the runtime falls through to regex routing. Disabled (`PLANNING_ENABLED=false`)
  → the pipeline is byte-for-byte the pre-cycle-8 behaviour.
- **Memory-first / local-first preserved:** the coordinator accepts *pre-fetched*
  memory context (it reads nothing itself — single-reader discipline via
  `KnowledgeEngine._retrieve_memory_context`); reasoning/synthesis use the local
  model; the internet stays `needs_internet`-gated inside the invoker.
- **WorkingMemory:** a per-run `ReasoningScratchpad` (TTL, injectable clock)
  holds intermediate step results for multi-step reasoning and is **never
  persisted** — only the final verified answer flows to the existing
  `_persist_turn` path. Distinct from `memory.WorkingMemory` (the durable-item
  LRU inside `MemoryManager`).
- **Structured telemetry (local, ₹0):** `planning/telemetry.py` emits one JSON
  line per planner decision / task execution / retry / latency / fallback reason
  to the existing rotating log (null sink when disabled → zero overhead; never
  raises). See the cycle-8 change-log entry for the event schema.
- **Security preserved:** DANGEROUS-permissioned tools are treated as
  unavailable to the autonomous planner unless `TOOL_AUTO_APPROVE=true` (no
  stdin hang, no unapproved side effects); `python` capability is intentionally
  backendless (no arbitrary code exec); the verifier reuses `core.response_guards`
  so no internal machinery leaks.

**Key architectural facts:**
- Routing is **regex/keyword-based** for simple/single-tool intents (IntentEngine
  + RoutingEngine `_TOOL_PATTERNS` + `tools/intent_detector.py`); **cycle 8**
  adds **confidence-based** routing via the `PlanningCoordinator` for actionable,
  multi-step goals (regex remains the fallback).
- Orchestrator agent selection = **first `can_handle()` match** in
  registration order (`orchestrator/core.py:_find_agent`).
- `LLMGuard` (retry/timeout/graceful-fallback) wraps LLM calls; used by
  KnowledgeEngine.
- Agents share a clean base (`agents/base.py`, `IAgent`), lifecycle
  (initialize/start/stop/health_check), `bind_runtime(context, event_bus)`.

---

## 4. Folder / file responsibilities

| Path | Responsibility | Status |
|---|---|---|
| `main.py` | Entry point, builds orchestrator+runtime, REPL, voice commands | REAL |
| `config/settings.py` | `Settings` dataclass + `load_settings()` from `.env` | REAL |
| `config/logging_config.py` | Logging setup | REAL |
| `runtime/runtime.py` | `Runtime` — single entry; builds guard, intent, memory; **c8** `_build_planning_coordinator` (reuses same guard+KnowledgeEngine+tool engine) → `set_planning_coordinator` | REAL |
| `runtime/conversation_runtime.py` | The pipeline (the heart); records each turn to persistent memory (`_persist_turn`); **c8** step 3c `_should_plan`/`_run_planning` (planning path, memory-first pre-fetch, fall-through on decline) | REAL |
| `runtime/knowledge_engine.py` | Direct-LLM chat/knowledge path; **now injects memory + (gated) live internet context** (cycle 6) | REAL ✅ |
| `knowledge/internet/interfaces.py` | **NEW** `IRetrievalProvider` protocol + `RetrievalResult` | REAL ✅ |
| `knowledge/internet/http_client.py` | **NEW** `SafeHttpClient` — whitelist/HTTPS/no-redirect/size-cap/retries (SSRF-safe egress) | REAL ✅ |
| `knowledge/internet/providers.py` | **NEW** `DuckDuckGoProvider` + `WikipediaProvider` (JSON APIs only) | REAL ✅ |
| `knowledge/internet/cache.py` | **NEW** async-safe bounded TTL cache | REAL ✅ |
| `knowledge/internet/router.py` | **NEW** `needs_internet()` freshness gate (priority ladder) | REAL ✅ |
| `knowledge/internet/service.py` | **NEW** `InternetKnowledgeService` — parallel fan-out, cache, rate-limit, fail-safe, injection-safe context | REAL ✅ |
| `planning/__init__.py` | **NEW c8** `build_planning_subsystem()` DI factory (→ `PlanningCoordinator` or `None`) + public exports | REAL ✅ |
| `planning/models.py` | **NEW c8** pure domain types: `TaskNode` (id/deps/priority/status/retry/cost/tool/confidence), `Plan`, `RetryPolicy`, `NodeResult`, `ExecutionMetrics`, `VerificationResult`, `PlanningOutcome`, `TaskStatus` | REAL ✅ |
| `planning/interfaces.py` | **NEW c8** DIP seams: `ICapabilityCatalog`/`ITaskPlanner`/`IToolInvoker`/`ITaskExecutor`/`IResponseVerifier` | REAL ✅ |
| `planning/planner.py` | **NEW c8** `TaskPlanner` (the PlannerAgent role) — LLM→strict-JSON decomposition + deterministic heuristic fallback; never raises | REAL ✅ |
| `planning/task_graph.py` | **NEW c8** `TaskGraph` — Kahn validation (unknown-dep + cycle), ready-set scheduling, skip-unreachable, progress | REAL ✅ |
| `planning/executor.py` | **NEW c8** `TaskExecutor` — continuous semaphore scheduling, per-task timeout, retries+backoff, cooperative cancel, progress+telemetry; never raises | REAL ✅ |
| `planning/capabilities.py` | **NEW c8** `CapabilityCatalog` — the confidence-based routing table; live availability from real tool registry; DANGEROUS-gated; `python` backendless | REAL ✅ |
| `planning/tool_invoker.py` | **NEW c8** `ToolInvoker` — delegates one task to the REAL backend (memory/internet/reasoning/tool); reimplements nothing; never raises | REAL ✅ |
| `planning/verifier.py` | **NEW c8** `ResponseVerifier` — empty/leak/tool-json/traceback/fabricated-success/grounding/hallucination/confidence checks + graceful salvage | REAL ✅ |
| `planning/scratchpad.py` | **NEW c8** `ReasoningScratchpad` — ephemeral per-run WorkingMemory (TTL, injectable clock); never persisted | REAL ✅ |
| `planning/coordinator.py` | **NEW c8** `PlanningCoordinator` — the single façade the runtime calls (plan→execute→synthesize→verify); never raises | REAL ✅ |
| `planning/telemetry.py` | **NEW c8** structured telemetry: `ITelemetrySink`/`Null`/`Logging`/`InMemory` + `PlanningTelemetry` facade (JSON lines → local log; never raises) | REAL ✅ |
| `planning/exceptions.py` | **NEW c8** `GraphValidationError`/`UnknownDependencyError`/`GraphCycleError` (raised only by `TaskGraph.validate`, caught by coordinator) | REAL ✅ |
| `core/response_guards.py` | **NEW c8** shared user-safe markers + `is_user_safe`/`looks_like_tool_json`/`has_leaked_scaffolding` (one list shared by runtime composer + verifier — no drift) | REAL ✅ |
| `runtime/intent_engine.py` | Multi-intent classification (word-boundary matching) | REAL |
| `runtime/routing_engine.py` | Intent→agent dispatch, `_TOOL_PATTERNS` | REAL |
| `runtime/personality_engine.py` | Canned greeting/small-talk fast path | REAL |
| `runtime/context_manager.py` | Session turns, pronoun/follow-up enrichment | REAL |
| `runtime/llm_guard.py` | Retry/timeout/fallback wrapper for LLM | REAL |
| `runtime/tool_executor.py` | Safe tool execution facade | REAL |
| `runtime/response_*.py` | synthesizer / formatter / composer | REAL |
| `runtime/fallback_engine.py` | Graceful degradation | REAL |
| `orchestrator/core.py` | Agent registry + routing + lifecycle | REAL |
| `orchestrator/{context,message_bus,task_queue,workflow,middleware}.py` | Infra | REAL |
| `agents/base.py`, `contracts.py`, `interfaces.py`, `capabilities.py` | Agent foundation | REAL |
| `agents/jarvis_agent.py` | JarvisPrimeAgent — central request processor | REAL (see debt) |
| `agents/planner.py` | Rule-based plan + LLM responder | REAL (bypassed for chat) |
| `agents/response_composer.py` | Multi-agent merge (LLM) | REAL |
| `agents/conversation_manager.py`, `context_manager.py` | Session/context (agent-side) | REAL (duplicates runtime/) |
| `agents/voice_agent.py` | **REWRITTEN** real speak/transcribe | REAL ✅ |
| `agents/*_agent.py` (13 "specialist" Marvel-named) | friday/veronica/vision/ultron/athena/stark/steve/oracle/gecko/hercules/hulk/jerome/pepper | REAL/HONEST — no fake success (2026-07-07) |
| `agents/{memory,tool}_agent.py` | Memory CRUD / tool exec agents | REAL |
| `agents/{notes,calendar,reminder,email}_agent.py` | SQLite-backed | REAL but not routed from chat |
| `agents/{desktop,browser}_agent.py` | Stubs (deps absent) | STUB |
| `memory/memory_service.py` | RAG facade; storage gate | REAL ✅ |
| `memory/persistent_memory.py` | **NEW** cross-session layer: sessions, projects, user profile, reflection, classification, restart-restore; **c9** `learn_preferences()` auto-promotes stated prefs to structured profile | REAL ✅ |
| `memory/preference_extractor.py` | **NEW c9** deterministic (regex, no LLM) extractor: name/language/location/occupation/coding-style/favorite/likes → `(key,value)`; precision-first (rejects questions & ambiguous forms) | REAL ✅ |
| `memory/reflection.py` | **NEW** ReflectionEngine — LLM+heuristic distil (summary/decisions/tasks/lessons) | REAL ✅ |
| `memory/validation.py` | **NEW** security guards: id/text sanitisation, item validation, injection-safe context block | REAL ✅ |
| `memory/memory_manager.py`, `vector_store.py`, `document_store.py`, `embedding_provider.py` | ChromaDB + MiniLM ONNX + SQLite; **NEW** `recent()` recency query | REAL, local |
| `memory/models.py` | MemoryItem + **extended** MemoryType taxonomy (project/task/decision/idea/reflection/user_profile/meeting_notes) | REAL |
| `llm/ollama_provider.py` | Local Ollama `/api/chat` | REAL ✅ |
| `llm/openai_provider.py` | Optional; only if user sets key | REAL (unused, off) |
| `llm/{base,factory,registry,chat_session,prompt_manager,interfaces}.py` | LLM plumbing | REAL |
| `tools/` + `tools/builtins/` | 14 builtin tools, registry, engine, permissions | REAL (some need absent deps) |
| `voice/interfaces.py` | ISTTProvider/ITTSProvider/IWakeWordDetector | REAL |
| `voice/piper_tts.py` | **NEW** Piper local TTS (CLI subprocess) | REAL ✅ |
| `voice/whisper_stt.py` | **FIXED** faster-whisper STT | REAL ✅ |
| `voice/audio_io.py` | **NEW** mic capture + speaker playback (sounddevice) | REAL ✅ |
| `voice/pipeline.py` | single-turn (push-to-talk) wake→STT→brain→TTS→playback | REAL ✅ |
| `voice/vad.py` | **NEW** EnergyVAD (adaptive noise floor) + async Endpointer | REAL ✅ |
| `voice/wake.py` | **NEW** wake strategies: TranscriptWakeWord (default), AlwaysAwake | REAL ✅ |
| `voice/continuous_loop.py` | **NEW** hands-free ASLEEP↔ACTIVE conversation loop | REAL ✅ |
| `voice/audio_io.py` | mic capture + playback + **NEW** `stream_frames()` streaming | REAL ✅ |
| `voice/wake_word.py` | openwakeword acoustic detector (dep absent → inert; optional) | OPTIONAL |
| `voice/edge_tts.py` | ⚠️ Online MS service — non-default fallback only | AVOID (not local) |
| `prompt/` | Chunking, token budget, input reader, CLI compose | REAL |
| `plugins/` | Plugin interfaces only | INTERFACE ONLY |
| `voice_models/` | Downloaded Piper `.onnx` (gitignored) | present: en_US-lessac-medium |

### Repo hygiene debt
Stray junk dirs duplicating real root files — safe-to-delete candidates:
`ini/`, `md/`, `txt/`, `typed/`, `no_extension/`, `example/`. (Not yet removed.)

---

## 5. Tools (14 builtins, `tools/builtins/`)

base64, browser*, calculator, clipboard*, datetime, file_system*, hash, json,
notification, screenshot*, shell*, system_info, text, uuid.
`*` = needs an absent native dep or is DANGEROUS-permissioned. calculator has
a hardened AST expression-safety layer (`tools/expression_safety.py`).
**c9:** the engine now supports `pow`, `exp`, `log2`, `hypot`, `atan2` (in
addition to the prior set), keeps **large integers exact** (`factorial(100)`,
`2**5000` no longer overflow/lose precision), **guards** against pathological
inputs (huge exponent/factorial → clean "too large", never a hang), and maps
every math error to **user-facing text** (no `math domain error` leak). The
routing fragment in `tools/intent_detector.py` is kept in sync, so `pow(2,10)`
now reaches the calculator — the LLM is never asked to do maths.

---

## 6. Voice pipeline (COMPLETE & VERIFIED, 2026-07-07)

- **TTS:** Piper (local ONNX). Invoked via `python -m piper` **CLI subprocess**
  (NOT `import piper`) to keep the GPL-3.0 engine at arm's length →
  project stays permissively licensed. Synthesizes to a temp WAV file
  (piper 1.4.2 stdout mode crashes on Windows).
- **STT:** faster-whisper (MIT), model `base`. Fixed the original bug — it
  now decodes WAV→float32 mono array (old code passed raw bytes and never
  worked). Runs in a worker thread.
- **Audio I/O:** sounddevice, 16-bit mono. Graceful no-op without hardware.
- **Pipeline:** `VoicePipeline(brain, stt, tts, audio)`; provider-agnostic
  (brain = anything with `async run(str)->str`, i.e. `Runtime`).
- **Verified working:** Piper↔Whisper round-trip transcribes verbatim; full
  brain loop "capital of France?" → LLM → "Paris" → spoken aloud.
  `can_speak: True`, `can_listen: True` on the dev machine.

### Continuous hands-free loop (COMPLETE & VERIFIED, 2026-07-07 cycle 2)
`voice/continuous_loop.py` — production always-on conversation:

```
mic stream (open once) ─► EnergyVAD endpointer ─► Whisper STT
   ─► [wake gate] ─► Runtime.run(text) ─► Piper TTS ─► playback ─► repeat
```

- **State machine:** ASLEEP ↔ ACTIVE. In `transcript` wake mode, an utterance
  activates only if it contains a wake word ("jarvis"/"computer"); the command
  after the wake word runs immediately ("Jarvis, what's the time?").
- **Endpointing:** energy VAD with adaptive noise floor + pre-roll; an utterance
  ends on ~0.8 s trailing silence (configurable), not a fixed timer.
- **Exit phrases** end the session ("goodbye", "shutdown", "exit", ...);
  **sleep phrases** re-arm the wake word ("go to sleep", "stop listening");
  **inactivity timeout** (default 30 s) returns ACTIVE→ASLEEP.
- **Mic stays open** across turns (`AudioIO.stream_frames`, RawInputStream →
  asyncio.Queue) for low latency. Playback runs off the event loop.
- **Fully injectable** (brain/STT/TTS/audio/frame-source) → unit-testable
  without a mic. A finite frame source ends via a `STREAM_ENDED` sentinel
  (distinct from inactivity `None`) so the loop terminates cleanly in tests.
- **Verified:** real Piper→Whisper loop — "Jarvis, what is two plus two?"
  woke + answered, "Goodbye" exited (wakes=1, turns=1, stopped=exit_phrase).

**Wake word = transcript-based (default), zero extra deps.** Chosen over
openwakeword to stay dependency-light and testable; openwakeword (heavy:
scikit-learn+scipy) remains an optional acoustic upgrade behind
`VOICE_WAKE_MODE`, not required.

### How the user activates voice
1. `.env` has `VOICE_ENABLED=true` (+ VOICE_* keys). **Currently ENABLED.**
2. Run `.venv/Scripts/python main.py`. Banner shows
   `Voice:piper(speak=on,listen=on)`.
3. REPL commands:
   - `!speak <text>` — say text aloud.
   - `!voice` — single push-to-talk turn (record ~5 s → answer → speak).
   - `!converse` — **hands-free continuous loop** (say "jarvis ..." ; "goodbye"
     or Ctrl+C to stop).
   - `VOICE_AUTOSTART=true` launches straight into `!converse` on boot.
   - Plain text still works as normal chat.
4. First transcription is slow (~10–20 s) — Whisper loads once, then fast.
5. Wake modes (`VOICE_WAKE_MODE`): `transcript` (say wake word) | `none`
   (always listening, every utterance is a command).

### Voice setup for a fresh machine
`pip install -r requirements-voice.txt` then
`python -m piper.download_voices en_US-lessac-medium --data-dir voice_models`.

---

## 7. Key design decisions (do not silently reverse)

1. **Knowledge/chat bypasses the planner+workflow** and calls the LLM directly
   via KnowledgeEngine. Reason: routing normal questions through the rule-based
   planner + fake-specialist workflow + 14 injected tool-defs made a 3B model
   emit empty/tool-JSON, and returned the greeting fallback. Specialist/tool
   intents still go through RoutingEngine.
2. **Memory storage gate** (`MemoryService._is_storeworthy`): never store
   empty/failed/boilerplate/tool-JSON turns. Reason: a poisoning feedback loop
   — failures were retrieved as "context" and dragged answers into fallback.
3. **Piper via subprocess** (license isolation, see §6).
4. **Word-boundary intent matching** (`IntentEngine._has_keyword`): `"photo"`
   must not match `"photosynthesis"`; multi-word phrases matched as substrings.
5. **Prompt templates use `Template.substitute` with empty-fill** for missing
   placeholders (was `safe_substitute`, which leaked literal `$tool_results`).
6. **Voice OFF by default in code defaults**; enabled per-machine via `.env`.
   Text mode must remain byte-for-byte unchanged when voice is off.
7. **Continuous voice = energy VAD endpointing + transcript wake word**, no
   heavy deps (no webrtcvad/silero/openwakeword required). Endpointer returns
   `bytes` | `None` (inactivity) | `STREAM_ENDED` (source exhausted). The loop
   is dependency-injected for testability; the mic is opened once per session.
8. **Planner orchestrates, never reimplements (cycle 8).** The `PlanningCoordinator`
   and `ToolInvoker` delegate to the EXISTING Memory / InternetKnowledgeService /
   KnowledgeEngine / ToolEngine / LLMGuard. No parallel Memory, no parallel LLM
   path. Reason: a duplicate system would violate the single-source-of-truth
   rule and drift from the stabilized behaviour. If you're tempted to add logic
   here that already exists elsewhere, wire to the existing subsystem instead.
9. **Planning activates ONLY for actionable + multi-step goals** (`_should_plan`).
   Reason: the cycle-7 memory-first / local-first fast paths (chat, knowledge,
   `current_info` → internet, single-tool) are proven and must not regress;
   routing simple turns through a task graph would reintroduce the 3B-model
   empty/tool-JSON failure mode cycle 1 fixed. Everything else falls through.
10. **Planning is fail-safe & non-breaking.** Every planning component returns a
    typed result and never raises; `PlanningOutcome.accepted=False` falls through
    to regex routing; `PLANNING_ENABLED=false` restores byte-for-byte pre-c8
    behaviour. New constructor params are keyword-only with no-op defaults
    (existing positional call sites unaffected).
11. **`TaskExecutor` uses continuous semaphore scheduling, NOT the wave-based
    `orchestrator.workflow.WorkflowEngine`.** Reason: the workflow engine blocks
    on the slowest node per wave (head-of-line blocking between a fast local tool
    and a slow model call); the planner needs a node to start the instant its
    deps finish. These are deliberately two schedulers for two different jobs
    (workflows vs. ad-hoc goal plans) — not duplication.
12. **Telemetry is a separate concern from logging and the progress callback.**
    `_logger.*` = human log lines; `progress(event,node,detail)` = live UI
    strings; `PlanningTelemetry` = machine-readable JSON records for the five
    required signals. A null sink by default keeps it zero-overhead; a raising
    sink can never break planning. Reused the existing rotating file logger
    rather than the `AgentMessage` MessageBus (wrong shape, adds latency).

---

## 8. Progress scores (updated 2026-07-08)

| Dimension | Score | Notes |
|---|---:|---|
| Architecture | 9/10 | Clean layering; **cycle 8** adds a SOLID/DI planning subsystem that orchestrates (never duplicates) existing services; residual duplication (2 composers, 2 context mgrs) still open |
| Agent honesty | 8/10 | Specialists no longer fake success; `success` reflects real work (2026-07-07). Consolidation of 13→few still open |
| Planner | 8/10 | **cycle 8**: real `TaskPlanner` (LLM→JSON + heuristic) + dependency `TaskGraph` + concurrent `TaskExecutor` + `ResponseVerifier`, wired for actionable/multi-step goals. Legacy `agents/planner.py` still the conversational responder; quality bounded by the local 3B model |
| Runtime | 9/10 | Solid pipeline; KnowledgeEngine fixed the core failure; **c8** planning path integrates cleanly with narrow activation + fall-through |
| Memory | 9/10 | Real local RAG + persistent cross-session layer auto-wired into the runtime, restart-proven; **c8** adds ephemeral per-run WorkingMemory (scratchpad) for multi-step reasoning; **c9** auto-promotes stated preferences to the structured profile (roadmap #11 done) → exact identity recall; ranking still basic; knowledge graph + scheduled decay open |
| Voice | 9/10 | Continuous hands-free loop + wake word + VAD verified; acoustic wake (openwakeword) + barge-in still open |
| Internet Retrieval | 8/10 | Context-only DuckDuckGo+Wikipedia, SSRF-safe, cached, router-gated, live-verified; **c8** planner reuses the same gated service; ranking + more providers open |
| Planning / Task Execution | 8/10 | **NEW c8** decompose→graph→parallel execute (retries/timeout/cancel)→verify; confidence-based routing; fail-safe fall-through; structured telemetry; 115 offline tests. Bounded by 3B decomposition quality; `python` capability intentionally backendless |
| Browser | 1/10 | Playwright/selenium absent; tool is a guarded stub (distinct from Internet Retrieval above) |
| Vision | 1/10 | PIL/deps absent; stub |
| OCR | 0/10 | pytesseract absent; not implemented |
| Desktop Automation | 1/10 | pyautogui/pyperclip absent; stub |
| Reasoning | 7/10 | **c8**: real multi-step decomposition + dependency-ordered execution + step-result fusion for actionable goals; single-shot LLM still the path for plain chat |
| Tool Calling | 8/10 | Framework real; **c8** confidence-based capability routing + per-tool arg extraction; **c9** deterministic math engine hardened (`pow`/large-ints/clean-errors) and math phrasings route to the calculator, never the LLM; small model still unreliable free-form |
| Testing | 9/10 | **1034** offline tests (+70 in cycle 9: calculator pow/big-int/error/guard, pow routing, preference extractor, auto-learn wiring, synthesizer big-int), all offline fakes |
| Production Readiness | 8/10 | Chat + hands-free voice + live cross-session memory + gated internet retrieval + **c8 planning** + **c9 reliable maths & auto-learned profile** real & verified; contracts strictly enforced (no None/leak/crash); browser/vision/OCR/desktop inert |
| **Overall Completion** | **~73%** | Conversation + continuous voice + persistent memory (now auto-learning preferences) + context-only internet retrieval + hardened contracts + enterprise planning + **a reliable deterministic calculator**, all wired into the live runtime & verified |

Prior "455 tests / production-ready" claim (see `PAT_REPORT.md`) was
misleading — tests were green but flagship behavior (answering a question) was
broken. Now fixed and verified.

---

## 9. Test suite

- Full run (minus slow memory + opt-in integration): **1034 passed** as of
  cycle 9 (was 964). **+70 cycle-9 tests, all offline fakes:**
  `test_expression_safety.py` (pow/companions, exact large integers, clean
  error messages, reliability guards), `test_intent_detector.py` (pow/exp/log2
  route to the calculator, not the LLM), `test_preference_extractor.py` (NEW —
  positive extraction + precision guardrails/no-false-fire + value hygiene),
  `test_runtime_persistence.py` (auto-learn wired / gated off / failure-safe /
  minimal-fake-safe), `test_runtime_pipeline.py` (synthesizer keeps big
  integers exact). No Ollama/network/ChromaDB.
- Prior: **964 passed** as of cycle 8 (was 845). **+115 planning/task-execution
  tests, all offline fakes:**
  `test_planning_task_graph.py` (validation/cycles/skip), `test_planning_scratchpad.py`
  (TTL/results), `test_planning_planner.py` (LLM-JSON + heuristic + defensive parse),
  `test_planning_verifier.py` (empty/leak/fabricated-success/grounding/confidence),
  `test_planning_executor.py` (parallelism/deps/timeout/retry/cancel/scratchpad),
  `test_planning_tool_invoker.py` (backend delegation/unavailability/arg-extraction),
  `test_planning_coordinator.py` (accept/decline/never-raises + factory),
  `test_conversation_runtime_planning.py` (activation gate + fall-through),
  `test_planning_telemetry.py` (**25** — sinks, executor/coordinator emission, JSON
  validity, never-raises, factory on/off). No Ollama/network/ChromaDB.
- Prior: `tests/test_production_stabilization.py`
  (**28**, all offline fakes — None-content contract, memory-first recall, internet
  routing, no-internal-leak; each reproduces a specific observed bug).
- Prior new: `tests/test_internet_knowledge.py`
  + `tests/test_knowledge_engine_internet.py` (**51**, all offline — httpx
  `MockTransport` + injected clocks; no live network in CI).
  `tests/test_persistent_memory.py` (42, fast fakes + real-SQLite restart tests).
  Slow `tests/test_memory_manager.py` (42) still green with real ChromaDB.
- Slow tests (load ChromaDB ONNX / whisper): `test_integration.py`,
  `test_memory_manager.py`. Skip during focused work.
- **`integration` marker** (registered in `pytest.ini`): real-audio tests
  (Piper/Whisper). Run with `-m integration`; excluded by default focused runs
  via `-m "not integration"`.
- New this project: `tests/test_knowledge_engine.py` (15),
  `tests/test_voice_pipeline.py` (11), `tests/test_voice_continuous.py`
  (13 fast + 1 integration).
- Run focused: `.venv/Scripts/python -m pytest tests/test_X.py -q -m "not integration"`.
- ⚠️ No `pytest-timeout` plugin installed — wrap long runs in an external
  `timeout` when there's any risk of a loop hang.

---

## 10. Roadmap (priority order)

1. ~~**Hands-free wake-word loop**~~ ✅ DONE (2026-07-07 cycle 2) — see §6.
   Remaining voice polish (optional): acoustic wake via openwakeword; barge-in
   (interrupt TTS by speaking); streaming partial transcripts; wake chime.
2. **Consolidate the 13 specialist agents** — ~~several return fake success~~
   ✅ fake success REMOVED (2026-07-07): every specialist now reports honest
   `success` (real work, real failure, or an explicit "unavailable"). Remaining:
   collapse the 13 into a few genuinely distinct LLM-backed agents (dedup/naming).
3. **Remove duplication** — merge `agents/response_composer.py` ↔
   `runtime/response_composer.py`, and `agents/context_manager.py` ↔
   `runtime/context_manager.py`. Clear ownership.
4. **Repo hygiene** — delete stray `ini/ md/ txt/ typed/ no_extension/ example/`.
5. ~~**Real browser/web tools** (DuckDuckGo/Wikipedia HTTP)~~ ✅ DONE as the
   **Internet Knowledge Engine** (cycle 6, roadmap #9). `current_info` questions
   now get real live facts instead of hallucination. Remaining (optional):
   more providers, result re-ranking, optional Playwright for JS-heavy pages.
6. **General chat model** — consider a non-coder Ollama model for warmth.
7. Vision/OCR/Desktop — only if user wants them; each needs native deps.
8. ~~**Wire persistent memory into the runtime**~~ ✅ DONE (cycle 5, 2026-07-07)
   — `record_turn` per turn, `restore_session` on boot, `reflect_on_session` on
   exit are all live (gated by `MEMORY_PERSIST_ENABLED`). Restart-proven with
   the real Runtime. Remaining (optional): per-turn `build_context` enrichment
   is already covered by KnowledgeEngine's semantic recall over the same store;
   scheduled `cleanup`/decay still open.
9. ~~**Internet Knowledge Engine**~~ ✅ DONE (cycle 6, 2026-07-07) — context-only
   DuckDuckGo+Wikipedia retrieval, router-gated, SSRF-safe, cached, live-verified.
10. **Knowledge graph** memory layer (future-ready; entities/relations over the
   existing typed memories). **STILL NOT STARTED** — deliberately deferred again
   in cycle 8 (planning subsystem took priority per the milestone).
11. ~~**Natural-language preference extraction**~~ ✅ DONE (cycle 9, 2026-07-08).
   Preferences stated in ordinary conversation ("call me Boss", "my favourite
   language is Rust", "I live in Bangalore", "I'm a backend developer") now
   auto-promote to structured `set_preference` entries via a deterministic,
   local `PreferenceExtractor` wired into the existing `_persist_turn` path — no
   "remember" command required. Precision-first (questions/ambiguous forms are
   rejected); gated by `MEMORY_AUTO_LEARN_ENABLED`. The `KnowledgeEngine` already
   injects `get_profile()` for identity questions, so recall is now exact.
12. ~~**Planning & Task Execution subsystem**~~ ✅ DONE (cycle 8, 2026-07-08) —
   `planning/` decomposes actionable/multi-step goals into a dependency task
   graph, executes concurrently (retries/timeout/cancel), routes by capability
   confidence, and verifies the answer; orchestrates existing subsystems, never
   duplicates them; structured telemetry; 115 offline tests. Remaining (optional):
   a safe sandboxed `python` capability; relevance-scored capability selection;
   a telemetry metrics exporter/dashboard; consolidating the two schedulers'
   shared graph math if `WorkflowEngine` is ever retired.

---

## 11. Known limitations / technical debt

- **Planning (cycle 8):** decomposition quality is bounded by the local 3B model
  — the heuristic fallback splits only on coordinating conjunctions (compound
  goals), so an unusual phrasing may under-decompose (falls back to a single
  reasoning step, never errors). The `python` capability is intentionally
  backendless (no safe local executor) → any task the planner assigns to it
  reports honest unavailability. Capability selection is confidence/keyword-based,
  not relevance-scored. Telemetry is emitted to the local log but there is no
  aggregation/exporter yet (an `InMemoryTelemetrySink` exists for in-process
  reads/tests). `TaskGraph` and `orchestrator.workflow.WorkflowEngine` share
  conceptual DAG math but are deliberately separate (continuous vs. wave-based);
  unifying them is optional future work, not required.
- Continuous voice (`!converse`) is verified with synthetic/real-audio feeds
  but the LIVE microphone path (`AudioIO.stream_frames` from a physical mic)
  is not automatically tested — needs a human with a mic. Import-safe & guarded.
- Voice has no barge-in yet: JARVIS finishes speaking before it listens again.
- Wake word is transcript-based (Whisper on every VAD-detected utterance) —
  higher CPU than an acoustic detector; fine for a desktop, not a low-power SBC.
- ~~Specialist agents partly theatrical (fake success returns).~~ FIXED 2026-07-07
  — success now reflects reality across all specialists.
- Duplicate ResponseComposer + ContextManager (agent-side vs runtime-side).
- Persistent memory is now **auto-wired into `conversation_runtime`** (cycle 5):
  per-turn `record_turn`, boot `restore_session`, exit `reflect_on_session`.
  Session/project filtering scans recent-by-type then
  filters in Python (fine at local scale; add a `session_id` SQL column if the
  store grows large). Semantic recall across restart needs the persistent Chroma
  dir (already the default); fast tests use in-memory fakes so only SQLite
  durability is asserted there (real-backend restart verified manually + slow suite).
- Reflection quality is bounded by the local 3B model; heuristic fallback is
  coarse (keyword-based). Memory decay/`cleanup` exists on the manager but is not
  scheduled.
- Browser/Vision/OCR/Desktop non-functional (deps not installed). NOTE: the
  Internet Knowledge Engine (cycle 6) is a *retrieval* layer, NOT a browser — it
  does not render pages or run JS; the browser *tool/agent* remain stubs.
- Internet retrieval routing is keyword/regex-based (`router.needs_internet`) —
  robust for the listed cases but not exhaustive; an odd phrasing of a
  time-sensitive question could miss the gate (falls back to local answer, never
  errors). DuckDuckGo Instant Answer is sparse for some queries (Wikipedia
  usually compensates). Results are ranked by fixed provider score, not
  relevance-scored. No result is ever persisted (by design).
- "remember I like X" stores a conversation turn (semantically recalled), not a
  structured preference yet — see roadmap #11.
- Piper 22.05kHz vs Whisper 16kHz — no resampling (works, not unified).
- `edge_tts.py` present but online — must never be the default.
- Coder LLM (`qwen2.5-coder:3b`) weak at general chat & tool selection.
- ChromaDB emits a Py3.14 `asyncio.iscoroutinefunction` DeprecationWarning
  (upstream, harmless).

---

## 12. Change log

### 2026-07-08 — Cycle 9: Runtime Intelligence — calculator reliability + auto preference learning
**Context / method:** the brief was a broad "make the runtime production-ready"
wishlist (15 problems). An empirical audit showed cycles 6–8 had **already**
implemented the large majority (continuous `!converse` loop, parallel internet
fan-out with timeout/cache/dedup, reflection/summarization, persistent
cross-session memory, pronoun resolution, confidence routing, VAD/Whisper STT).
Per the "do NOT rewrite / stop on duplicate logic / extend the existing
subsystem" rule, cycle 9 **did not rebuild** any of those. It closed the **two
genuine, verifiable gaps** found by probing each subsystem — each an extension
of an existing component, fully offline-testable, backward-compatible.

**Gap 1 — the deterministic math engine (complaints #5/#6/#13).** `pow()` was a
required function but missing from the whitelist, so `pow(2,10)` and "what is
pow(2,10)" classified as `unknown`/`knowledge_question` and were answered by the
**LLM** — a direct violation of *"the LLM must NEVER calculate mathematics."*
Errors also leaked raw library text ("expected a nonnegative input"), and
`2**5000`/`factorial(100)` failed or lost precision under a blanket
`float(result)` cast.
- `tools/expression_safety.py`: added `pow` (built-in, exact big-int),
  `exp`, `log2`, `hypot`, `atan2`; `evaluate()` now **preserves `int`** for
  integral results (exact large integers) and maps every exception to clean,
  user-facing text (`_friendly_error`); **reliability guards** `_safe_pow` /
  `_safe_factorial` refuse pathological magnitudes (huge exponent/factorial)
  cleanly instead of hanging.
- `tools/intent_detector.py`: `_MATH_FUNC_NAMES` kept in sync (longest-first
  alternation) so the new functions route to the calculator, not the LLM.
- `runtime/response_synthesizer.py`: `_format_calculator` no longer routes a
  big-integer string through `float()` (which overflowed to `inf` then raised
  on `int(inf)`), keeping the "answer" formatting for exact large integers.
- `tools/builtins/calculator.py`: spec/description updated.

**Gap 2 — automatic preference learning (complaints #2/#12, roadmap #11).**
`set_preference` existed but was **never called from the runtime** (grep-proven):
"I like Python" / "call me Boss" were stored only as raw turns, so `get_profile()`
stayed empty and identity recall relied on fuzzy semantic search.
- `memory/preference_extractor.py` (NEW): deterministic, local (regex, **no
  LLM**) `PreferenceExtractor` mapping name/preferred_language/location/
  occupation/coding_style/favorite_*/likes to `(key,value)`. **Precision-first**:
  questions ("what is my name?"), ambiguous statements ("I am tired"), and
  pronoun/filler values ("I like it") are rejected; values are clause-trimmed,
  length-capped, and single-long-token garbage is dropped.
- `memory/persistent_memory.py`: new `learn_preferences(user_text)` uses the
  extractor + the **existing** `set_preference` (upsert) — no new storage,
  single-writer discipline. Extractor is DI'd (default-constructed).
- `runtime/conversation_runtime.py`: `_persist_turn` now also calls
  `learn_preferences` best-effort (getattr-guarded so minimal fakes are safe;
  runs independently of the turn store-worthiness gate).
- Config: `MEMORY_AUTO_LEARN_ENABLED` (default true) threaded
  `settings → Runtime → ConversationRuntime`; `.env.example` updated.
- Loop closed: `KnowledgeEngine._retrieve_memory_context` already injects
  `get_profile()` first for personal/identity questions, so learned facts now
  surface via **exact structured recall**.

**Preserved / backward-compatible:** every new constructor param is keyword-only
with a no-op default; `MEMORY_AUTO_LEARN_ENABLED=false` restores pre-c9 recording
behaviour; math results still compare equal to the old floats (`4 == 4.0`); no
existing public API changed; no subsystem duplicated (both changes extend
`tools/expression_safety` and `memory/persistent_memory`).

**Tests:** focused suite **1034 passed** (was 964; **+70**), 0 regressions
(slow memory/integration excluded as usual). New/extended:
`test_expression_safety.py`, `test_intent_detector.py`,
`test_preference_extractor.py` (NEW), `test_runtime_persistence.py`,
`test_runtime_pipeline.py`.

**Manual verification (offline, deterministic, real code paths):**
- Preference loop through the **real** `ConversationRuntime` + real
  `PersistentMemoryService`: "call me Boss" / "my favourite language is Rust" /
  "I live in Bangalore" / "I'm a backend developer" → structured profile
  `{name: Boss, preferred_language: Rust, location: Bangalore, occupation:
  backend developer}`; `KnowledgeEngine` then injects all four for "what do you
  know about me?".
- Calculator through the real routing→tool→synthesizer chain: `pow(2,10)`→1024,
  `pow(2,100)`/`factorial(100)`/`2**5000` exact, `17%5`→2, `sqrt(-1)`/`1/0` →
  clean messages, "explain why 2+2=4" correctly stays on the LLM path. Every
  math phrasing routed to the calculator (never the LLM).

**Remaining debt (unchanged / new):** preference extraction is regex-based
(precision over recall — unusual phrasings are missed, never mis-stored);
`favorite_*`/`likes` values are latest-wins single values (no list history);
calculator "unit conversions" from the wishlist remain out of scope (degrees/
radians aside). Roadmap #10 (knowledge graph) still deferred.
- Overall completion ~72% → ~73%. Tool Calling 7→8.

### 2026-07-08 — Cycle 8: Planning & Task Execution subsystem (roadmap #12)
**Objective:** an enterprise-grade layer that decomposes complex, *actionable*
goals into a dependency-aware task graph, executes it concurrently
(retries/timeout/cancel), routes each task to a REAL backend by capability
confidence, and verifies the final answer — while **orchestrating the existing
subsystems, never reimplementing them**, and preserving every cycle-7 guarantee
(memory-first, local-first, internet-gated, no None/leak/crash, plugin arch).

**New package `planning/` (10 modules) + `core/response_guards.py`:**
- `models.py` — pure domain types. `TaskNode` carries the exact required fields:
  `id`, `dependencies`, `priority`, `status`, `retry_policy`, `estimated_cost`,
  `required_tool`, `confidence` (+ `args`, `critical`, `result`). `Plan`,
  `RetryPolicy`, `NodeResult`, `ExecutionMetrics`/`TaskMetrics`,
  `VerificationResult`, `PlanningOutcome`, `TaskStatus`.
- `interfaces.py` — `ICapabilityCatalog` / `ITaskPlanner` / `IToolInvoker` /
  `ITaskExecutor` / `IResponseVerifier` (DIP seams; everything faked in tests).
- `planner.py` — `TaskPlanner` (the **PlannerAgent** role, req #1): local model
  → strict JSON decomposition, defensive parse, deterministic **heuristic
  fallback** (mirrors `ReflectionEngine`). Never raises; writes nothing.
- `task_graph.py` — `TaskGraph` (req #2): Kahn validation (unknown-dep + cycle),
  priority-ordered ready-set, `skip_unreachable` on critical failure, progress.
- `executor.py` — `TaskExecutor` (req #3): **continuous semaphore scheduling**
  (a node starts the instant its deps finish — no wave head-of-line blocking),
  per-task `wait_for` timeout, `RetryPolicy` backoff, cooperative `asyncio.Event`
  cancel, failure isolation, progress + telemetry. Never raises.
- `capabilities.py` — `CapabilityCatalog` (req #4): the **confidence-based
  routing table**. Live availability from the real `ToolRegistry`;
  DANGEROUS tools gated behind auto-approve; `python` intentionally backendless.
- `tool_invoker.py` — `ToolInvoker`: delegates ONE task to the existing backend
  (Memory / InternetKnowledgeService / KnowledgeEngine|LLMGuard / ToolEngine).
  Reimplements nothing; per-tool arg extraction (e.g. calculator expression).
- `verifier.py` — `ResponseVerifier` (req #5): empty / leaked-exception /
  tool-JSON / traceback / fabricated-success / tool-result grounding /
  numeric-URL hallucination / confidence-threshold, + graceful salvage/fallback.
- `scratchpad.py` — `ReasoningScratchpad` (req #6): ephemeral per-run
  **WorkingMemory** (TTL, injectable clock), never persisted.
- `coordinator.py` — `PlanningCoordinator` (req #7): the single façade the
  runtime calls; plan → execute → synthesize → verify. Never raises.
- `telemetry.py` — structured telemetry (`ITelemetrySink`, Null/Logging/InMemory,
  `PlanningTelemetry` facade): one JSON line per event to the local rotating log.
- `__init__.py` — `build_planning_subsystem()` DI factory (→ coordinator or
  `None` when disabled).

**Planner workflow (req #10):**
```
goal ─► CapabilityCatalog (live availability)
     ─► TaskPlanner.decompose ──► Plan            (LLM strict-JSON,
     │      • empty?      → decline → regex fallback   else heuristic split;
     │      • conf < min? → decline → regex fallback   snap unknown tool→reasoning)
     ─► fresh ReasoningScratchpad (per-run WorkingMemory)
     ─► TaskExecutor.execute(plan) ──► ExecutionMetrics + NodeResults
     ─► synthesize (single step → its output; multi → local-model fuse)
     ─► ResponseVerifier.verify ──► VerificationResult
     ─► PlanningOutcome(accepted, response, plan, metrics, verification, reason)
```

**Runtime integration sequence (req #10) — step 3c, non-breaking:**
```
User ─► ConversationRuntime.process
          │  IntentEngine.classify
          ▼
     _should_plan(intent)?  (actionable AND multi-step/heavy-planning,
          │  no          not current_info / conversation / vision)
          │────────────────► existing paths (KnowledgeEngine / RoutingEngine) [UNCHANGED]
          │ yes
          ▼
     _gather_memory_context (memory-first pre-fetch via KnowledgeEngine)
          ▼
     PlanningCoordinator.run(goal, memory_context)
          │  accepted? ── yes ─► format ─► context.update ─► _persist_turn ─► reply
          │  no (decline/low-conf/all-failed)
          ▼
     fall through to Step 4 RoutingEngine (regex)  [NEVER dead-ends]
```

**Task lifecycle (req #10) — `TaskStatus` state machine:**
```
PENDING ─(deps satisfied)─► READY ─(worker slot)─► RUNNING ─┬─► SUCCEEDED
   │                                                        ├─► FAILED     (backend error, retries exhausted)
   │                                                        ├─► TIMED_OUT  (per-task wait_for, retries exhausted)
   │                                                        └─► CANCELLED  (cooperative cancel)
   └─(a critical dependency reached FAILED/TIMED_OUT/CANCELLED)─► SKIPPED
```
Retries loop within RUNNING per `RetryPolicy` (`backoff * multiplier**attempt`);
a `task_retry` telemetry event fires on each. Terminal states are recorded once
in `_finish_node` (the single choke point → exactly one `task_completed` event).

**Execution metrics + telemetry (req #10):** `ExecutionMetrics` aggregates
total / succeeded / failed / skipped / cancelled / timed_out / total_attempts /
`wall_time_ms` + `per_task` `TaskMetrics`. **Structured telemetry** emits the
five required signals as JSON lines (local log, ₹0, never raises):
| signal | event(s) | key fields |
|---|---|---|
| planner decision | `plan_decided` | strategy, node_count, confidence |
| tool execution | `task_started` + `task_completed` | node_id, tool, status, attempts |
| retries | `task_retry` | node_id, attempt, reason |
| latency | `task_completed.duration_ms`, `run_completed.wall_time_ms` | ms |
| fallback reason | `plan_declined`, `run_completed` (accepted=false) | reason |
Emission verified end-to-end on a live 2-step run (see manual verification).

**Confidence-based routing supersedes regex — but only for actionable/multi-step
goals** (`_should_plan`). Simple chat, knowledge questions, greetings,
single-tool requests, `current_info` (→ internet), and vision all keep their
proven fast paths. Regex routing (Step 4) is the fallback and handles everything
the planner declines.

**Preserved (req #8) — verified:** memory-first (coordinator takes pre-fetched
context; reads nothing itself), local-first (synthesis via the local model),
internet only when required (`needs_internet`-gated inside the invoker), security
(DANGEROUS-gated, `python` backendless, verifier reuses `core.response_guards`),
plugin architecture (capabilities resolve through the existing `ToolRegistry`),
and full backward compatibility (`PLANNING_ENABLED=false` → byte-for-byte pre-c8;
every new constructor param is keyword-only with a no-op default).

**Settings / env:** `PLANNING_ENABLED` (default true), `PLANNING_MAX_PARALLEL=4`,
`PLANNING_TASK_TIMEOUT_SECONDS=30`, `PLANNING_MAX_RETRIES=1`,
`PLANNING_CONFIDENCE_THRESHOLD=0.55`, `PLANNING_MIN_GOAL_CONFIDENCE=0.5`,
`PLANNING_TELEMETRY_ENABLED` (default true).

**Files:** NEW `planning/` (11 files incl. telemetry + exceptions),
`core/response_guards.py`, `tests/test_planning_*.py` (9 files, **115 tests**).
MODIFIED (wiring only): `runtime/runtime.py` (`_build_planning_coordinator`),
`runtime/conversation_runtime.py` (step 3c `_should_plan`/`_run_planning`/
`_gather_memory_context`), `config/settings.py` + `.env.example` (7 settings),
`runtime/response_composer.py` (shares `core.response_guards`). No existing
public API changed; no logic duplicated (the coordinator/invoker delegate).

**Tests:** focused suite **964 passed** (was 845; +115, incl. 25 telemetry),
0 regressions (slow memory/integration excluded as usual).

**Manual verification (live, offline fakes for determinism):** a 2-step goal
("Find the capital of France and then summarise it") decomposed → executed s1→s2
in dependency order → synthesized → verified (accepted, conf 0.64); the telemetry
stream emitted `plan_decided` → `task_started/task_completed` ×2 →
`run_completed` (wall_time_ms, succeeded=2/total=2), each a valid JSON line.
Decline paths (empty/low-confidence/all-failed/coordinator-error) each emit a
`run_completed` with the fallback reason and fall through to regex routing.

**Discovery note:** cycle 8's implementation was found **already present and
green** at session start (uncommitted `planning/` + wiring + 90 tests). Per the
"stop on duplicate logic" rule, the parallel build was NOT redone; this cycle
**completed** the two genuine gaps — structured telemetry (+25 tests) and this
PROJECT_BRAIN.md documentation. Roadmap #10 (knowledge graph) remains deferred.
- Overall completion ~69% → ~72%. Planner 5→8, Architecture 8→9, Reasoning 5→7.

### 2026-07-08 — Cycle 7: Production Stabilization (mandatory, pre-roadmap-#10)
**Objective:** make the *existing* architecture production-stable before adding
any new capability. No new features, no architecture changes — enforce strict
contracts so the runtime never crashes, never leaks internals, and routes
memory/internet queries correctly. Quality over features.

- **Root cause of the flagship crash** (`'NoneType' object has no attribute
  'strip'/'split'` on "remember my name is Nikhil" / "Call me boss"): a provider
  could hand back `None` content, which then flowed into `.strip()`/`.split()`
  in the planner/decomposer. **Fixed at the single source of truth:**
  `llm.interfaces.LLMResponse.__post_init__` now coerces `content` → `str` (None
  → `""`), `tool_calls` → `tuple`, `metadata` → `dict`. `BaseLLMProvider.
  generate_text()` is guaranteed to return a `str` (`response.content or ""`).
  Planner/Jarvis additionally fall back to a real reply on empty content rather
  than propagating it. This closes the *entire class* of None-string bugs, not
  just the two reported phrases.
- **Contract enforcement at every subsystem boundary** (issue 6/7): memory ids /
  session ids / preference keys sanitised (`memory/validation.py`), every
  `MemoryItem` validated before store, `_is_storeworthy` guards empty/junk turns,
  empty collections returned instead of `None` throughout.
- **Memory-first recall (issue 2):** identity/preference questions ("Who am I?",
  "What is my name?", "What do you know about me?") consult persistent memory +
  semantic recall *before* the LLM and **never leave the machine**
  (`is_memory_query` short-circuits the internet gate). Verified: durable facts
  are injected into the prompt ahead of generation.
- **Preference durability (issue 3):** "Remember I like Python" / "Call me boss"
  store successfully and are recalled **after a restart** (semantic recall over
  the same durable ChromaDB+SQLite store). Structured auto-promotion to
  `set_preference` remains roadmap #11 (not required for stability).
- **Internet routing (issue 4):** weather / news / "latest" / current
  office-holders trigger the KnowledgeEngine → router-gated
  `InternetKnowledgeService` **only**; timeless "what is/explain" questions stay
  100% local (`needs_internet` returns False). Fail-safe: any fetch error/timeout
  → local-only answer, never a crash.
- **No internal leaks (issue 5):** `RuntimeResponseComposer` never surfaces
  internal failure messages ("memory_id is required", "…not installed",
  "cannot handle task type", stack traces) — these are logged internally and
  replaced with a graceful user-facing response; execution continues.
- **Regression tests (issue 8):** `tests/test_production_stabilization.py` — **28
  tests**, one per observed bug, written to fail pre-fix and pass post-fix; all
  offline fakes (no Ollama/network/ChromaDB).
- **Full suite (issue 9):** **845 passed** (was 817; +28), 0 regressions (slow
  memory/integration excluded as usual).
- **Manual verification (issue 10), live Ollama + real restart:** Who am I? /
  Call me boss / Remember I like Python / restart / What do I like? (recalled
  "Python") / Weather in Bangalore / Latest AI news / What is recursion? / Solve
  23*(18+7) → **575** / Tell me a joke — **all return clean strings; no crash, no
  leaked internals**. (Weather/news answer via the gated internet path; the free
  DDG/Wikipedia providers are sparse for live weather — routing is correct and
  fails safe, a known provider limitation, not a runtime bug.)
- **Files touched (hardening only, no architecture change):** `llm/interfaces.py`,
  `llm/base.py`, `agents/planner.py`, `agents/jarvis_agent.py`,
  `runtime/response_composer.py`, `memory/{validation,memory_service,exceptions,
  models,memory_manager}.py`, `orchestrator/core.py` + infra. New:
  `tests/test_production_stabilization.py`.
- **Roadmap #10 (knowledge graph) deliberately NOT started** — the runtime is
  now stable; capability expansion resumes in the next cycle.
- Overall completion ~68% → ~69%. Production Readiness 7→8.

### 2026-07-07 — Cycle 6: Internet Knowledge Engine (roadmap #9)
**Objective:** a lightweight, context-only, last-resort retrieval layer so
time-sensitive questions (weather/news/current events/recent releases/office-
holders) get real public facts, while reasoning stays 100% local in Qwen and the
internet stays subordinate to memory + the local model.
- **New package `knowledge/internet/`:** `interfaces.py` (`IRetrievalProvider`
  protocol + `RetrievalResult`), `http_client.py` (`SafeHttpClient` — the SSRF-
  safe egress boundary), `providers.py` (`DuckDuckGoProvider` + `WikipediaProvider`,
  JSON APIs only), `cache.py` (async TTL cache), `router.py` (`needs_internet()`
  freshness gate encoding the priority ladder), `service.py`
  (`InternetKnowledgeService` — parallel fan-out, cache, rate-limit, fail-safe,
  injection-safe context), plus `__init__.py` + `build_internet_service()` factory.
- **Modified (wiring only, no refactors):** `runtime/knowledge_engine.py` (inject
  internet service; consult AFTER memory and ONLY when `needs_internet`; add live
  context block to the prompt; send only the current question), `runtime/
  conversation_runtime.py` + `runtime/runtime.py` (thread the service through),
  `main.py` (compose once, stash on `orchestrator.internet_service`),
  `config/settings.py` + `.env.example` (6 `INTERNET_*` settings; default on).
- **New tests:** `tests/test_internet_knowledge.py` (router, SafeHttpClient
  security matrix, providers vs mocked JSON, service fan-out/dedupe/timeout/cache/
  fail-safe — via httpx `MockTransport`, no live network) +
  `tests/test_knowledge_engine_internet.py` (engine gating + privacy + fail-safe).
  **51 new tests.**
- **Priority order enforced (per user refinement):** 1 memory → 2 conversation →
  3 local KB → 4 local docs → 5 internet (only if required) → 6 Qwen. Memory/
  personal queries and timeless "what is/explain" questions never leave the box.
- **Security review:** HTTPS-only + host **whitelist** checked before any socket
  (no SSRF, no arbitrary fetch); **no redirect following** (blocks SSRF pivot);
  bounded timeout + retries; **512 KiB** streamed size cap (DoS-resistant);
  localhost/127.0.0.1 rejected; **JSON only, no HTML parsing** (no parser attack
  surface); per-service **rate limit**; **only the question is sent outward**;
  retrieved text wrapped in an **injection-safe** untrusted-data block; **nothing
  persisted**. All fail-safe → local-only answer on any error/timeout.
- **Performance:** async parallel provider fan-out under one overall timeout;
  short-TTL cache avoids duplicate calls; cancellation on timeout salvages
  completed providers; the runtime never blocks. Local-only queries pay **zero**
  network cost (router short-circuits before any fetch).
- **Trade-offs:** router is keyword/regex (simple, testable, occasionally
  conservative — misses fall back to local, never error); results ranked by fixed
  provider score, not relevance; DDG Instant Answer is sparse for some queries
  (Wikipedia compensates). All deliberate for a lightweight ₹0 layer.
- **Tests:** focused suite **817 passed** (was 766; +51), 0 regressions (slow
  memory/integration excluded as usual). **Live manual verification (Wi-Fi):**
  "What is Python?" → local, no fetch; "Who is the PM of India?" → **Narendra
  Modi** (fetched + injected); "Remember I like dark mode" → recorded; restart →
  "What do I like?" stayed **offline** (memory-owned) and recalled **dark mode**.
- **Remaining debt:** natural-language "remember X" → structured preference not
  auto-promoted yet (roadmap #11); result re-ranking + more providers open.
- Overall completion ~65% → ~68%.

### 2026-07-07 — Cycle 5: wire persistent memory into the runtime (roadmap #8)
**Objective:** make the restart-verified persistent-memory layer actually *live*
— JARVIS records every turn, restores the session on boot, and reflects on exit,
so it truly continues where it left off after a restart.
- **Manual runtime verification FIRST (per policy):** booted the real
  orchestrator+Runtime and drove live turns — "capital of France" → "Paris",
  "17×23" → "391", photosynthesis → correct. No runtime bugs found; the earlier
  "not fully initialized" was only an under-constructed ad-hoc test (no
  orchestrator), not a product bug. Runtime confirmed stable before adding wiring.
- **Modified:** `config/settings.py` + `.env.example` (`MEMORY_PERSIST_ENABLED`,
  `MEMORY_SESSION_ID`), `main.py` (build one `PersistentMemoryService` with an
  LLM-backed `ReflectionEngine`; stash on `orchestrator.persistent_memory`;
  `restore_session` on boot; `reflect_on_session` in `finally`), `runtime/runtime.py`
  (`_extract_persistent_memory`, thread service + session_id), `runtime/
  conversation_runtime.py` (accept `persistent_memory`; new DRY `_persist_turn`
  called after BOTH the knowledge fast-path and the routing path).
- **New:** `tests/test_runtime_persistence.py` (5: turn recorded, greeting NOT
  recorded, storage failure doesn't break reply, no-layer no-op, empty input).
- **Design:** single rolling session (`default`) so each boot resumes one
  continuous thread. `record_turn` is **awaited** (not fire-and-forget) so
  errors stay honest and chronology is correct; the write is small + off-thread.
  Recording feeds the SAME manager the KnowledgeEngine already reads → past
  conversations become semantically recallable for free. Text mode unchanged
  when `MEMORY_PERSIST_ENABLED=false`.
- **Honest failure:** a `record_turn`/restore/reflect error is logged
  (`_logger.exception`) and never breaks the reply or shutdown.
- **Tests:** focused suite **766 passed** (was 761; +5), 0 regressions (slow
  memory/integration excluded as usual). **End-to-end restart proof with the
  real Runtime across two separate boots:** boot 1 auto-persisted a live turn;
  boot 2 (fresh orchestrator) restored it from disk. Library-level proof also
  confirmed turns + preference + semantic recall survive a fresh process.
- Overall completion ~62% → ~65%. Memory 8→9.

### 2026-07-07 — Cycle 4: persistent (cross-session) memory
**Objective:** JARVIS remembers conversations, projects, and preferences across
restarts, and reflects to get smarter without retraining.
- **New files:** `memory/persistent_memory.py` (`PersistentMemoryService` —
  sessions/projects/user-profile/reflection/classification/restart-restore),
  `memory/reflection.py` (`ReflectionEngine` + `Reflection`, LLM→JSON with
  deterministic heuristic fallback), `memory/validation.py` (security guards),
  `tests/test_persistent_memory.py` (42).
- **Modified:** `memory/models.py` (extended `MemoryType`: project/task/decision/
  idea/reflection/user_profile/meeting_notes — additive, back-compatible),
  `memory/memory_manager.py` (+`recent()` recency-ordered durable query),
  `memory/exceptions.py` + `core/exceptions.py` (+`MemoryValidationError`),
  `memory/__init__.py` (exports).
- **Architecture:** thin DI layer over the existing `MemoryManager`; no new
  backend (reuses SQLite doc store + ChromaDB). Deterministic ids give upsert
  for projects/preferences. SOLID/DRY (reuses the store-worthiness gate).
- **Security:** id sanitisation (rejects `..`/path separators/invalid chars →
  no path traversal), text sanitisation (control-char strip, NFC, size caps →
  overflow/DoS), `validate_memory_item` (non-empty, importance∈[0,1], metadata
  cap → corruption), injection-safe `to_safe_context_block` (memory-poisoning
  mitigation), JSON-only (no pickle → no unsafe deserialisation), dedup reused
  (duplicate prevention). Reflection never fabricates — falls back to heuristic.
- **Performance:** reads use recency-ordered document-store scans (no
  re-embedding on preference/project reads); embeddings only on write; async
  throughout (SQLite off-thread); working-memory restore for low-latency replay.
- **Tests:** 42 new (fast fakes + real-SQLite restart). Focused suite **761
  passed** (was 719); slow `test_memory_manager.py` **42 passed** (real Chroma);
  end-to-end restart **verified with real ChromaDB+SQLite+ONNX** (turns, project,
  preference, reflection survived; semantic recall correct). 0 regressions.
- **Not yet done (debt/roadmap #8):** auto-wiring into `conversation_runtime`;
  knowledge-graph layer; scheduled decay.
- Overall completion ~60% → ~62%.

### 2026-07-07 — Cycle 3: specialist-agent honesty (no fake success)
**Objective:** eliminate simulated success across the 13 specialist agents so
`AgentResult.success` always reflects real work (policy: no fake implementations).
- **Principle applied:** operations that need a backend (shell / file_system /
  tool engine) return `success=False` when it is absent or when the backend
  actually failed; pure no-op stubs return `success=False` with a clear message.
- **Modified agents:**
  - `agents/stark_agent.py` — `build.compile` & `build.deploy` now propagate the
    real shell result (deploy requires a `command`; no more `deployed:True`
    fabrication); `project.setup` requires a tool engine and reports per-item
    mkdir/touch failures.
  - `agents/steve_agent.py` — `test.run` & `coverage.report` propagate the real
    pytest exit result (a failing run is no longer reported as success).
  - `agents/gecko_agent.py` — `browser.navigate` & `web.automate` no longer fake
    success; return `unavailable` and point callers at the working `web.fetch`.
  - `agents/hercules_agent.py` — `compute.process` fails honestly with no engine;
    `batch.execute` now genuinely runs each op via the tool engine (real
    per-op success), or fails if no engine / no ops.
  - `agents/pepper_agent.py` — `ux.speak` no longer fakes TTS; defers to the
    voice subsystem (VoiceAgent).
  - `agents/ultron_agent.py` — **security fix**: `security.scan` reimplemented in
    pure Python (removes a shell **command-injection** vector from unsanitised
    `target`/`pattern`), runs off-thread via `asyncio.to_thread`, bounded by
    file-count / file-size / match caps (DoS-resistant); `system.monitor` fails
    honestly with no engine.
- **Unchanged (already honest):** friday, veronica, oracle, hulk, jerome, athena,
  vision (`_screenshot` already returned `success=False`).
- **Tests:** `tests/test_specialist_agents.py` updated + expanded (mock/failing
  tool engines; no-engine and failure-propagation cases; real Ultron scan).
  Focused suite **719 passed** (was 706), 0 regressions. Blast radius confined to
  this test file (workflow tests use task-type strings only).
- Overall completion ~58% → ~60%.

### 2026-07-07 — Session: audit + conversation fix + voice pipeline
**Conversation fix**
- Added `runtime/knowledge_engine.py` (direct-LLM chat path).
- `memory/memory_service.py`: added `_is_storeworthy` storage gate.
- `llm/prompt_manager.py`: `substitute` + empty-fill (fixed `$tool_results` leak).
- `runtime/intent_engine.py`: word-boundary matching (`_has_keyword`); removed
  bare `"scan"` from vision keywords.
- `runtime/conversation_runtime.py` + `runtime/runtime.py`: route knowledge
  intents to KnowledgeEngine; thread `memory_service` through.
- `main.py`: UTF-8 stdio reconfigure.
- Purged poisoned live vector/doc store (backed up under `memory_data/_backup_*`).
- Result: "explain photosynthesis" / "capital of France" / "17×23" all now
  answer correctly (were empty / greeting-fallback / crash before).
- Tests: `tests/test_knowledge_engine.py` (15) added.

**Voice pipeline**
- New: `voice/piper_tts.py`, `voice/audio_io.py`, `voice/pipeline.py`,
  `requirements-voice.txt`, `tests/test_voice_pipeline.py` (11).
- Rewrote `agents/voice_agent.py` (stub → real), `voice/__init__.py`.
- Fixed `voice/whisper_stt.py` (WAV decode; threaded).
- `config/settings.py` + `.env.example`: 6 `VOICE_*` settings.
- `main.py`: voice provider factory, `!speak`/`!voice` commands, banner + voice
  status; VoiceAgent now built with providers.
- `.gitignore`: ignore `voice_models/`, `*.onnx`.
- Installed: piper-tts, faster-whisper, sounddevice. Downloaded
  `en_US-lessac-medium` voice.
- Verified: full local TTS↔STT round-trip + brain loop spoken aloud.
- Updated `tests/test_agents.py` voice test to match real agent.
- Full suite: 693 passed (minus slow memory tests).

### 2026-07-07 — Session: PROJECT_BRAIN established
- Created `docs/PROJECT_BRAIN.md` as permanent knowledge base + session policy.
- `.env`: enabled voice (`VOICE_ENABLED=true` + VOICE_* keys) at user request.

### 2026-07-07 — Cycle 2: continuous hands-free voice loop
**Objective:** turn the single-turn voice proof into a production continuous
conversation loop (wake → listen → STT → brain → TTS → repeat).
- **New files:** `voice/vad.py` (EnergyVAD + Endpointer + STREAM_ENDED),
  `voice/wake.py` (WakeStrategy / TranscriptWakeWord / AlwaysAwake / factory),
  `voice/continuous_loop.py` (ContinuousVoiceLoop state machine),
  `tests/test_voice_continuous.py` (13 fast + 1 real-audio integration).
- **Modified:** `voice/audio_io.py` (+`stream_frames` async mic streaming),
  `voice/__init__.py` (exports), `config/settings.py` + `.env` + `.env.example`
  (7 new VOICE_* settings: wake_mode, frame_ms, trailing_silence, max_utterance,
  inactivity_timeout, greeting, autostart), `main.py` (`!converse` command,
  autostart, continuous-loop factory, banner/help), `pytest.ini` (integration
  marker).
- **Architecture:** Strategy (wake), DI throughout (loop injectable), event-
  driven state machine, single long-lived mic stream. No new heavy deps —
  reuses numpy + existing Whisper/Piper. Fully local, ₹0.
- **Tests:** 706 passed (was 693; +13 fast). Real-audio integration test passes
  (Piper speech → VAD endpointer → Whisper transcript). End-to-end loop verified
  with real Piper+Whisper: wake + command + exit phrase all correct.
- **Runtime change:** new `!converse` REPL command + `VOICE_AUTOSTART`; text and
  `!voice`/`!speak` behaviour unchanged.
- **Debt:** live-mic path not auto-tested (needs hardware); no barge-in;
  transcript wake is CPU-heavier than acoustic. Voice score 8→9.
- Overall completion ~55% → ~58%.
