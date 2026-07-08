# Architecture Refactoring: JarvisPrimeAgent as Single Entry Point

## 1. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER (CLI REPL)                                  │
│                    main.py  ←→  compose_response()                         │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      Orchestrator.route()                                 │
│           AgentTask("jarvis.process", payload)                             │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     JarvisPrimeAgent.handle()                              │
│                                                                           │
│  ┌─────────────────┐  ┌───────────────────┐  ┌─────────────────────────┐  │
│  │ ConversationMgr │  │  MemoryService    │  │  ToolManager           │  │
│  │  - session state │  │  - RAG enrich     │  │  - execute tools       │  │
│  │  - turn tracking  │  │  - persist        │  │  - intent detection    │  │
│  └─────────────────┘  └───────────────────┘  └─────────────────────────┘  │
│                                                                           │
│                        ┌──────────────────┐                              │
│                        │  _decompose_goal  │                              │
│                        │  LLM or rule-based │                             │
│                        └────────┬─────────┘                              │
│                                 │                                          │
│          ┌──────────────────────┴──────────────────────┐                  │
│          ▼                                             ▼                  │
│  ┌──────────────────┐                        ┌──────────────────┐         │
│  │  Zero task types  │                        │  One+ task types │         │
│  │  (conversational) │                        │  (multi-agent)   │         │
│  └────────┬─────────┘                        └────────┬─────────┘         │
│           ▼                                           ▼                   │
│  ┌──────────────────┐                        ┌──────────────────┐         │
│  │  PlannerAgent    │                        │  WorkflowEngine  │         │
│  │  - plan + respond │                        │  - build plan     │         │
│  │  - LLM response   │                        │  - execute steps  │         │
│  └────────┬─────────┘                        └────────┬─────────┘         │
│           │                                           │                   │
│           │              ┌────────────────────────────┘                   │
│           │              ▼                                                │
│           │     ┌──────────────────────┐                                 │
│           │     │  Specialist Agents   │                                 │
│           │     │  (Friday, Veronica,  │                                 │
│           │     │   Stark, Steve, ...) │                                 │
│           │     └──────────┬───────────┘                                 │
│           │                ▼                                              │
│           │     ┌──────────────────────┐                                 │
│           └────►│  ResponseComposer    │                                  │
│                 │  - merge results     │                                  │
│                 │  - LLM fusion        │                                  │
│                 │  - fallback          │                                  │
│                 └──────────┬───────────┘                                  │
│                            ▼                                              │
│                 ┌──────────────────────┐                                 │
│                 │  Memory persist      │                                  │
│                 └──────────────────────┘                                 │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
                    compose_response()
                    (CLI output)
```

## 2. Execution Flow

### A. Normal conversational query (simple greeting / chat)

```
main.py REPL
  └─ orchestrator.route(AgentTask("jarvis.process", {"goal": "Hello"}))
       └─ JarvisPrimeAgent.handle(task)
            ├─ ConversationManager.get_or_create_session()
            ├─ MemoryService.enrich_prompt(goal) → enriched goal
            ├─ _decompose_goal(enriched) → [] (empty — no task types detected)
            ├─ PlannerAgent.handle(AgentTask("plan", {"goal": goal}))
            │    ├─ _plan() → structured internal plan
            │    └─ _respond() → natural language response
            ├─ ConversationManager.add_turn(goal, response)
            ├─ MemoryService.store(goal + response)
            └─ Return AgentResult(response)
```

### B. Multi-agent workflow query (e.g., "research AI and write code")

```
main.py REPL
  └─ orchestrator.route(AgentTask("jarvis.process", {"goal": "Research AI agents and implement one"}))
       └─ JarvisPrimeAgent.handle(task)
            ├─ ConversationManager.get_or_create_session()
            ├─ MemoryService.enrich_prompt(goal) → enriched
            ├─ _decompose_goal(enriched) → ["research", "code"]
            ├─ _execute_and_merge(goal, ["research", "code"])
            │    ├─ WorkflowPlan(goal, steps=[
            │    │    WorkflowStep("research", target="friday"),
            │    │    WorkflowStep("code", target="veronica", depends_on=["research"])
            │    │  ])
            │    ├─ WorkflowEngine.execute(plan)
            │    │    ├─ Step "research" → FridayAgent (research results)
            │    │    └─ Step "code" → VeronicaAgent (code implementation)
            │    └─ ResponseComposer.merge(goal, [friday_result, veronica_result])
            │         ├─ LLM merger (if LLM available) → fused response
            │         └─ OR simple concatenation (fallback)
            ├─ ConversationManager.add_turn(goal, merged_response)
            ├─ MemoryService.store(goal + merged_response)
            └─ Return AgentResult(merged_response)
```

### C. Tool execution query (e.g., "what time is it" → /tool datetime)

```
main.py REPL
  └─ orchestrator.route(AgentTask("jarvis.process", {"goal": "what time is it", "task_type": "tool.execute"}))
       └─ JarvisPrimeAgent.handle(task)
            └─ _handle_tool_execution(task)
                 ├─ ToolManager.execute("datetime", {})
                 └─ Return AgentResult(response="Current time is ...")
```

## 3. File Tree (Excluding `__pycache__`, `.git`, `.egg-info`, `.venv`, `node_modules`, `build`, `dist`, `logs`, `memory_data`)

```
F:\AI_Agent_Orchestration\
│
├── .env                          # Environment variables (local)
├── .env.example                  # Template for .env
├── README.md                     # Project overview
├── main.py                       # Application entry point (async REPL)
├── py.typed                      # PEP 561 marker
├── pytest.ini                    # Pytest config (asyncio_mode = auto)
├── requirements.txt              # Runtime dependencies
│
├── agents\                       # Agent implementations
│   ├── __init__.py               # Package exports
│   ├── base.py                   # Abstract Agent base class
│   ├── contracts.py              # Data contracts: AgentTask, AgentResult, AgentMessage
│   ├── interfaces.py             # IAgent interface
│   ├── planner.py                # PlannerAgent (internal planning + LLM response)
│   ├── response_composer.py      # ResponseComposer (result merging) ★ NEW
│   ├── conversation_manager.py   # ConversationManager (multi-turn state) ★ ENHANCED
│   ├── jarvis_agent.py           # JarvisPrimeAgent (central entry point) ★ REWRITTEN
│   ├── capabilities.py           # Capability enums
│   ├── friday_agent.py           # FridayAgent (Research & Information)
│   ├── veronica_agent.py         # VeronicaAgent (Code Engineering)
│   ├── vision_agent.py           # VisionAgent (Image/OCR)
│   ├── ultron_agent.py           # UltronAgent (Security)
│   ├── athena_agent.py           # AthenaAgent (Strategy)
│   ├── stark_agent.py            # StarkAgent (Build/Deploy)
│   ├── steve_agent.py            # SteveAgent (Testing)
│   ├── oracle_agent.py           # OracleAgent (Knowledge)
│   ├── gecko_agent.py            # GeckoAgent (Browser/Web)
│   ├── hercules_agent.py         # HerculesAgent (Compute/Data)
│   ├── hulk_agent.py             # HulkAgent (Storage)
│   ├── jerome_agent.py           # JeromeAgent (DevOps)
│   ├── pepper_agent.py           # PepperAgent (UX/Notifications)
│   ├── memory_agent.py           # MemoryAgent (Memory CRUD)
│   ├── tool_agent.py             # ToolAgent (Tool execution)
│   ├── voice_agent.py            # VoiceAgent (Voice I/O)
│   ├── desktop_agent.py          # DesktopAgent
│   ├── browser_agent.py          # BrowserAgent
│   ├── notes_agent.py            # NotesAgent
│   ├── calendar_agent.py         # CalendarAgent
│   ├── reminder_agent.py         # ReminderAgent
│   └── email_agent.py            # EmailAgent
│
├── llm\                          # LLM provider abstraction layer
│   ├── __init__.py
│   ├── base.py                   # BaseLLMProvider, LLMConfig
│   ├── interfaces.py             # ILLMProvider, LLMResponse, ToolCall, ToolDefinition
│   ├── chat_session.py           # ChatSession (history management)
│   ├── factory.py                # build_llm_provider
│   ├── registry.py               # ProviderRegistry
│   ├── prompt_manager.py         # PromptManager (templates: responder, decomposer, merger)
│   ├── openai_provider.py        # OpenAI API provider
│   ├── ollama_provider.py        # Ollama local provider
│   └── errors.py                 # LLMError, LLMProviderError, LLMTimeoutError
│
├── orchestrator\                 # Coordination and routing
│   ├── __init__.py
│   ├── interfaces.py             # ISharedContext, IEventBus, ITaskQueue
│   ├── context.py                # SharedContext (thread-safe key-value)
│   ├── message_bus.py            # MessageBus (async pub/sub)
│   ├── task_queue.py             # TaskQueue (async background tasks)
│   ├── middleware.py             # MiddlewarePipeline (hooks)
│   ├── workflow.py               # WorkflowEngine, WorkflowPlan, WorkflowStep
│   ├── core.py                   # Orchestrator (registry, routing, lifecycle)
│   └── exceptions.py             # Orchestration exceptions
│
├── memory\                       # Memory and persistence
│   ├── __init__.py
│   ├── interfaces.py             # IMemoryStore, IVectorStore, etc.
│   ├── models.py                 # MemoryItem, MemoryType, calculate_importance
│   ├── memory_manager.py         # MemoryManager (orchestrates stores)
│   ├── memory_service.py         # MemoryService (agent-facing RAG pipeline)
│   ├── vector_store.py           # ChromaVectorStore
│   ├── document_store.py         # SQLiteDocumentStore
│   ├── embedding_provider.py     # ChromaEmbeddingProvider
│   └── exceptions.py             # Memory exceptions
│
├── tools\                        # Tool integration system
│   ├── __init__.py
│   ├── interfaces.py             # ITool, IToolRegistry, ToolSpec
│   ├── registry.py               # ToolRegistry
│   ├── engine.py                 # ToolExecutionEngine, ToolResult
│   ├── manager.py                # ToolManager (unified facade)
│   ├── permissions.py            # PermissionManager
│   ├── context.py                # ToolContext
│   ├── discovery.py              # Plugin discovery
│   ├── expression_safety.py      # Expression safety validator
│   ├── intent_detector.py        # IntentDetector
│   ├── exceptions.py             # Tool exceptions
│   ├── builtins\                 # Built-in tool implementations
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   ├── datetime_tool.py
│   │   ├── uuid_tool.py
│   │   ├── base64_tool.py
│   │   ├── hash_tool.py
│   │   ├── json_tool.py
│   │   ├── text_tool.py
│   │   ├── shell_tool.py
│   │   ├── system_info.py
│   │   ├── file_system_tool.py
│   │   ├── clipboard_tool.py
│   │   ├── notification_tool.py
│   │   ├── screenshot_tool.py
│   │   └── browser_tool.py
│   └── mcp\                      # MCP support
│       ├── __init__.py
│       └── interfaces.py
│
├── voice\                        # Voice I/O integrations
│   ├── __init__.py
│   ├── interfaces.py
│   ├── edge_tts.py
│   ├── whisper_stt.py
│   └── wake_word.py
│
├── prompt\                       # Prompt processing utilities
│   ├── __init__.py
│   ├── budget.py                 # TokenBudget
│   ├── chunker.py                # PromptChunker
│   ├── processor.py              # ChunkProcessor
│   ├── composer.py               # compose() CLI formatting
│   ├── input_reader.py           # InputReader
│   └── progress.py               # ChunkProgress
│
├── config\                       # Configuration
│   ├── __init__.py
│   ├── settings.py               # Settings + load_settings()
│   └── logging_config.py         # configure_logging()
│
├── plugins\                      # Plugin system
│   ├── __init__.py
│   └── interfaces.py
│
├── tests\                        # Test suite
│   ├── __init__.py
│   ├── test_jarvis_agent.py      # ★ NEW (14 tests)
│   ├── test_response_composer.py # ★ NEW (11 tests)
│   ├── test_agents.py
│   ├── test_chat_session.py
│   ├── test_context.py
│   ├── test_expression_safety.py
│   ├── test_integration.py
│   ├── test_intent_detector.py
│   ├── test_llm_factory.py
│   ├── test_llm_providers.py
│   ├── test_long_prompt.py
│   ├── test_memory_manager.py
│   ├── test_message_bus.py
│   ├── test_orchestrator.py
│   ├── test_prompt_manager.py
│   ├── test_settings.py
│   ├── test_specialist_agents.py
│   ├── test_task_queue.py
│   ├── test_tool_framework.py
│   └── test_workflow.py
│
└── docs\                         # Documentation
    ├── architecture_refactor.md  # This file
    ├── architecture.md           # Original architecture doc
    └── references\               # Reference images/videos
```

## 4. Key Files and Their Roles

### Core files (changed / new)
| File | Lines | Role |
|------|-------|------|
| `agents/jarvis_agent.py` | 394 | **★ REWRITTEN** — Single entry point: receive → enrich → decompose → execute → merge → persist |
| `agents/response_composer.py` | 133 | **★ NEW** — Merges multi-agent outputs into coherent response with LLM fusion fallback |
| `agents/conversation_manager.py` | 229 | **★ ENHANCED** — Added `get_context()` and `summarize()` for persistent state |
| `agents/planner.py` | 405 | **★ FIXED** — Moved `ChunkProcessor`/`TokenBudget` imports inside methods to fix circular import hang |
| `agents/__init__.py` | 65 | **★ UPDATED** — Exports `ResponseComposer` |
| `main.py` | 316 | **★ UPDATED** — All requests routed through `orchestrator.route(AgentTask("jarvis.process", ...))` |

### Test files (new)
| File | Lines | Tests | Role |
|------|-------|-------|------|
| `tests/test_response_composer.py` | 188 | 11 | Empty results, single/multi agent merge, LLM fusion, fallback |
| `tests/test_jarvis_agent.py` | 183 | 14 | Entry point, conversation turns, workflow engine, rule decomposition, health check, tool delegation |

### Supporting files (unchanged)
| File | Lines | Role |
|------|-------|------|
| `orchestrator/workflow.py` | 258 | `WorkflowEngine` — dependency-graph multi-agent execution |
| `orchestrator/core.py` | ~350 | `Orchestrator` — agent registry, routing, lifecycle |
| `memory/memory_service.py` | ~200 | RAG pipeline for memory enrichment |
| `memory/memory_manager.py` | 387 | Orchestrates vector + document stores |
| `tools/manager.py` | 189 | `ToolManager` — unified tool execution facade |
| `tools/intent_detector.py` | ~150 | NL → tool matching |
| `config/settings.py` | 129 | Configuration dataclass from `.env` |
| `prompt/composer.py` | ~100 | `compose_response()` CLI formatter |
| Specialist agents (22 files) | 50-300 each | Individual skill agents (Friday, Veronica, Stark, Steve, etc.) |

## 5. Design Decisions

### Decision 1: Every request goes through `"jarvis.process"`
**What:** Even `/tool` commands and intent-detected tool calls are wrapped as a `"jarvis.process"` task with `task_type: "tool.execute"` in the payload.
**Why:** Ensures JarvisPrimeAgent is always the brain — it can enrich with memory, track conversation state, and decide whether a simple tool call suffices or multi-agent orchestration is needed.
**Trade-off:** Slight overhead for simple tool invocations, but eliminates the bypass path where tool calls skipped JPA entirely.

### Decision 2: ResponseComposer as a standalone service
**What:** Extracted `_merge_responses()` and `_simple_merge()` from JarvisPrimeAgent into `agents/response_composer.py` as a stateless, independently testable class.
**Why:**
- Single responsibility — JPA orchestrates, ResponseComposer merges
- Reusable by other agents or future entry points (e.g., web API)
- Testable without instantiating the full JPA (11 dedicated tests)

### Decision 3: Rule-based decomposition as default (no LLM dependency)
**What:** `_decompose_goal()` first tries LLM-based decomposition via the `decomposer` prompt template. If no LLM provider is available, it falls back to `_rule_based_decompose()` which uses keyword matching.
**Why:** The system must work without any LLM. Rule-based decomposition supports the most common task types (research, code, test, deploy, etc.) and returns empty list for conversational queries (→ falls back to PlannerAgent).
**Coverage:** Handles: research, write, code, generate, create, implement, develop, build, test, deploy, security, strategy, analyze, browser, web, scrape, compute, data, process, storage, save, file, memory, remind, notify, tool, execute, run, command (29 keywords).

### Decision 4: Empty decomposition → PlannerAgent; Non-empty → WorkflowEngine
**What:** When `_decompose_goal()` returns 0 task types, JPA delegates to PlannerAgent for a single conversational response. When ≥1 task type, it builds a `WorkflowPlan` and executes through `WorkflowEngine`, then merges via `ResponseComposer`.
**Why:** Simple queries (greetings, chit-chat) don't need workflow overhead. Multi-step tasks benefit from dependency-graph execution, parallelization, and retry logic.

### Decision 5: PlannerAgent is ONLY used for simple/conversational queries
**What:** PlannerAgent is now a fallback for single-agent conversational scenarios, not the primary routing mechanism. It is not called for multi-agent workflows or tool executions.
**Why:** Clean separation — PlannerAgent focuses on plan + respond (no routing logic). JPA owns all routing decisions.

### Decision 6: ConversationManager provides `get_context()` and `summarize()`
**What:** Added two methods to `ConversationManager` to support persistent conversation state beyond a single CLI session.
**Why:** Enables future features like conversation save/load, session resume after restart, and context window management across sessions.

### Decision 7: `_handle_tool_execution` creates appropriate task for PlannerAgent or WorkflowEngine
**What:** When JPA receives a `tool.execute` task, it calls `_handle_tool_execution()` which creates either a "plan" task for PlannerAgent fallback or builds a `WorkflowPlan` with a "tool.execute" step for the workflow engine path.
**Why:** Tool execution is legacy-compatible — both PlannerAgent (via LLM tool calls) and WorkflowEngine (via direct ToolManager) are supported, with JPA making the routing decision.

### Decision 8: No duplicate PlannerAgent logic
**What:** JPA never replicates PlannerAgent's internal planning logic (phase 1 plan, memory retrieval, response generation). It delegates entirely.
**Why:** Avoids code drift and maintenance burden.

### Decision 9: Circular import fix — local imports in planner.py
**What:** `from prompt import ChunkProcessor, TokenBudget` was moved from module level into method bodies (`_respond()` and `_plan()` respectively).
**Why:** The import chain `prompt.__init__ → prompt/composer.py → agents → planner → prompt` caused a circular import that hung at collection time. Local imports resolve the cycle because they execute at call time, not module load time.

### Decision 10: Test strategy — mock providers for isolation
**What:** Tests use `StaticProvider` (ResponseComposer tests) and `EchoPlannerProvider` (JPA tests) with `unittest.mock` to avoid LLM/network dependencies.
**Why:** Fast, deterministic tests that validate logic without infrastructure.

## 6. Test Results

```
Total: 454 tests passed (429 existing + 25 new)
- test_response_composer.py: 11/11 passed
- test_jarvis_agent.py: 14/14 passed
- All existing tests: 429/429 passed (no regressions)
```
