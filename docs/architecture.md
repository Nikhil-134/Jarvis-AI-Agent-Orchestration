# JARVIS AI Agent Orchestration — Architecture

## Overview

JARVIS is a local-first, modular AI agent orchestration system with tool execution, memory management, and LLM integration. Everything runs locally on Windows with Ollama — no paid APIs or cloud services.

---

## Directory Structure

```
F:\AI_Agent_Orchestration
├── main.py              # CLI entry point
├── agents/              # Agent implementations
│   ├── base.py          # Abstract base Agent class
│   ├── contracts.py     # AgentTask, AgentResult data contracts
│   ├── interfaces.py    # IAgent interface
│   ├── planner.py       # PlannerAgent — plan + respond
│   ├── tool_agent.py    # ToolAgent — execute tools
│   ├── memory_agent.py  # MemoryAgent — manage memory
│   └── voice_agent.py   # VoiceAgent — voice interface stub
├── config/
│   ├── settings.py      # Pydantic settings (env/file)
│   └── logging_config.py
├── llm/
│   ├── base.py          # BaseLLMProvider, LLMConfig
│   ├── chat_session.py  # ChatSession with history
│   ├── interfaces.py    # LLMResponse, ToolDefinition, ToolCall
│   ├── ollama_provider.py
│   ├── openai_provider.py
│   ├── factory.py       # build_llm_provider()
│   ├── prompt_manager.py
│   └── registry.py
├── memory/
│   ├── memory_manager.py   # Core memory CRUD + RAG
│   ├── memory_service.py   # High-level memory facade
│   ├── models.py           # MemoryItem, MemoryType
│   ├── vector_store.py     # ChromaDB vector store
│   ├── document_store.py   # SQLite document store
│   └── embedding_provider.py
├── orchestrator/
│   ├── core.py          # Orchestrator — agent routing
│   ├── context.py       # SharedContext
│   ├── middleware.py     # MiddlewarePipeline
│   ├── message_bus.py   # Event bus
│   ├── task_queue.py     # Async task queue
│   └── interfaces.py    # IEventBus, ISharedContext, ITaskQueue
├── plugins/             # Plugin system
├── prompt/
│   ├── budget.py        # TokenBudget, estimate_tokens
│   ├── chunker.py       # PromptChunker
│   ├── composer.py      # ResponseComposer (natural language output)
│   ├── input_reader.py  # InputReader, InputMode
│   ├── processor.py     # ChunkProcessor
│   └── progress.py      # ChunkProgress
├── tools/
│   ├── builtins/        # Tool implementations
│   │   ├── calculator.py
│   │   ├── base64_tool.py
│   │   ├── datetime_tool.py
│   │   ├── file_system_tool.py   # Consolidated: read/write/list/search
│   │   ├── hash_tool.py
│   │   ├── json_tool.py
│   │   ├── shell_tool.py
│   │   ├── system_info.py
│   │   ├── text_tool.py
│   │   └── uuid_tool.py
│   ├── mcp/             # MCP tool interfaces
│   ├── engine.py        # ToolExecutionEngine
│   ├── manager.py       # ToolManager (facade)
│   ├── registry.py      # ToolRegistry
│   ├── permissions.py   # PermissionManager
│   ├── interfaces.py    # ITool, IToolRegistry, PermissionLevel, ToolSpec
│   ├── expression_safety.py  # AST-based math validator
│   ├── intent_detector.py    # NL → tool routing
│   └── context.py       # ToolContext
├── tests/               # 310 tests across 16 files
└── voice/               # Voice interface interfaces
```

---

## Execution Pipeline

```
User Input
    │
    ▼
InputReader ────► IntentDetector ────► ToolManager ────► ToolResult
    │                    │                                    │
    │                    ▼ (no match)                          │
    │                    │                                    │
    │                    ▼                                    │
    │              Orchestrator                               │
    │                    │                                    │
    │                    ▼                                    │
    │              PlannerAgent                               │
    │              ├── _retrieve_memories()                   │
    │              ├── _plan()  (rule-based, no LLM)          │
    │              ├── TokenBudget check                      │
    │              ├── _respond_single()  or                  │
    │              │   _respond_chunked()                     │
    │              └──→ LLM or fallback                       │
    │                    │                                    │
    │                    ▼                                    │
    │              ResponseComposer ───────────► CLI Output
    │                           (wraps in natural language)
    │
    └── /tool <name> ──► ToolManager ──► ToolResult ──► ResponseComposer
```

### Flow details

1. **InputReader** reads input from CLI (single-line, multi-line paste, or file).
2. **IntentDetector** checks if input matches a tool intent (calculator, datetime, uuid, base64, hash, JSON, system info, text, shell) via regex + AST analysis. If matched, routes directly to ToolManager.
3. **Orchestrator** routes plan tasks to PlannerAgent.
4. **PlannerAgent**:
   - Phase 1: Retrieves memories via RAG, builds rule-based internal plan.
   - Phase 2: Checks TokenBudget. If goal fits context window, sends single-shot LLM prompt. If not, chunks the goal and processes sequentially with progress indication.
   - LLM can invoke tools via tool definitions. Results are fed back for final response.
5. **ResponseComposer** wraps raw AgentResult data in natural conversational text — never exposes JSON, internal plans, or raw tool payloads.

---

## Component Relationships

```
main.py
  │
  ├──► InputReader        (prompt/input_reader.py)
  ├──► IntentDetector     (tools/intent_detector.py)
  │     └──► ExpressionSafety  (tools/expression_safety.py)
  ├──► Orchestrator       (orchestrator/core.py)
  │     ├──► PlannerAgent (agents/planner.py)
  │     │     ├──► ChatSession  (llm/chat_session.py)
  │     │     │     └──► BaseLLMProvider  (llm/base.py)
  │     │     │           ├──► OllamaProvider
  │     │     │           └──► OpenAIProvider
  │     │     ├──► MemoryService  (memory/memory_service.py)
  │     │     │     └──► MemoryManager  (memory/memory_manager.py)
  │     │     │           ├──► VectorStore (ChromaDB)
  │     │     │           └──► DocumentStore (SQLite)
  │     │     ├──► ToolExecutionEngine (tools/engine.py)
  │     │     │     ├──► ToolRegistry
  │     │     │     ├──► PermissionManager
  │     │     │     └──► ITool implementations
  │     │     └──► PromptManager  (llm/prompt_manager.py)
  │     ├──► ToolAgent    (agents/tool_agent.py)
  │     │     └──► ToolExecutionEngine
  │     ├──► MemoryAgent  (agents/memory_agent.py)
  │     │     └──► MemoryService
  │     └──► VoiceAgent   (agents/voice_agent.py)
  ├──► ToolManager        (tools/manager.py)
  │     └──► ToolExecutionEngine
  └──► ResponseComposer   (prompt/composer.py)
```

---

## Issues Found & Fixed (Architecture Review)

### 1. PermissionLevel Duplication
- **Location**: `tools/interfaces.py:16` and `tools/permissions.py:12`
- **Problem**: `PermissionLevel` IntEnum was defined in two files identically.
- **Fix**: Removed from `permissions.py`, imported from `interfaces.py`.

### 2. Duplicate File Tools
- **Location**: `tools/builtins/file_reader.py`, `file_writer.py`, `folder_listing.py`, `search_files.py`
- **Problem**: Four separate tools duplicated functionality of `file_system_tool.py`.
- **Fix**: Deleted the 4 files and their registrations. Only `FileSystemTool` remains.

### 3. Private Attribute Access
- **Location**: `agents/planner.py:_respond_chunked`, `_handle_tool_calls`
- **Problem**: Accessed `_chat_session._history`, `_chat_session._prompt_cache` directly.
- **Fix**: Added `ChatSession.append_message()` and `ChatSession.replace_last_assistant()` public methods.

### 4. No Response Composer
- **Location**: `main.py`
- **Problem**: `result.data.get("response", result.message)` leaked internal fallback "Planning completed." Tool results were printed raw without wrapping.
- **Fix**: Created `ResponseComposer` in `prompt/composer.py` — wraps all results in natural language, never exposes internals.

### 5. `.gitignore` Missing Runtime Data
- **Location**: `.gitignore`
- **Problem**: `memory_data/` directory (ChromaDB vectors + SQLite DB) not excluded.
- **Fix**: Added `memory_data/` to `.gitignore`.

### 6. Unused Imports
- **Location**: `agents/planner.py`
- **Problem**: `ChatMessage`, `ChunkProgress` imported but unused.
- **Fix**: Removed unused imports.

---

## Remaining Recommendations

### Medium Priority
1. **Add filesystem intent detection** — `IntentDetector` doesn't route "read file", "list directory" etc. to `file_system` tool.
2. **Add structured error types** — consolidate `ToolError`, `LLMError`, `MemoryError` under a common base.
3. **Add `hasattr(orch, "tool_manager")` → explicit property** — currently uses duck-typing check.
4. **Add `ToolsMiddleware`** — hooks into orchestrator middleware for logging, metrics.

### Low Priority
1. **Add `py.typed` marker** — for PEP 561 compliance.
2. **Add `pydantic` validation** for tool schemas (currently manual dict-based).
3. **Add `tzdata` to requirements** — optionally, for Windows timezone support.
4. **Consolidate `MemoryService`/`MemoryManager`** — the service/manager split adds indirection with thin delegation.

---

## Test Coverage

- **310 tests total** across 16 test files
- **All pass** (0 failures, 0 errors)
- Coverage: IntentDetector (43 tests), ExpressionSafety (26 tests), LongPrompts (43 tests), Tools framework (128 tests), Memory (35 tests), Integration (7 tests), Agents (6 tests), Chat (4 tests), Orchestrator (6 tests), etc.

### Test files
| File | Tests |
|------|-------|
| test_expression_safety.py | 26 |
| test_intent_detector.py | 43 |
| test_long_prompt.py | 43 |
| test_tool_framework.py | 128 |
| test_agents.py | 5 |
| test_memory_manager.py | 35 |
| test_integration.py | 7 |
| test_chat_session.py | 4 |
| test_orchestrator.py | 6 |
| test_message_bus.py | 5 |
| test_task_queue.py | 4 |
| test_context.py | 3 |
| test_settings.py | 5 |
| test_llm_factory.py | 3 |
| test_llm_providers.py | 4 |
| test_prompt_manager.py | 3 |

---

## Dependencies (zero-cost, local-first)

| Dependency | Purpose |
|------------|---------|
| `chromadb` | Vector storage for RAG |
| `ollama` | Local LLM inference |
| `openai` | OpenAI-compatible API |
| `pydantic-settings` | Configuration management |
| `aiofiles` | Async file I/O |
| `colorama` | CLI colors |
| `pyyaml` | YAML config |
| `rich` | Rich CLI output |

Zero cloud dependencies. Zero paid APIs.
