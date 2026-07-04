# Jarvis

Jarvis is a Python 3.12+ multi-agent AI orchestration system scaffold.

## Status

Phase 3 implements the core runtime architecture for a multi-agent orchestration system.
It includes agent contracts, placeholder agents, routing, inter-agent messaging,
shared context, asynchronous task queuing, lifecycle management, configurable LLM
providers, prompt templates, chat history, logging, and tests. Real agent
specialization beyond planner LLM integration is intentionally not implemented yet.

## Project Structure

```text
agents/        Agent interfaces and future agent implementations
orchestrator/  Coordination and routing layer
memory/        Memory abstractions and persistence adapters
tools/         Tool integrations and adapters
voice/         Voice input/output integrations
llm/           LLM provider abstraction, prompts, and chat sessions
config/        Configuration loading and settings
tests/         Automated tests
logs/          Runtime logs, ignored by git
```

## Phase 1 Modules

- `agents/contracts.py` defines the typed task and result objects shared by agents and the orchestrator.
- `agents/base.py` defines the abstract `Agent` interface and default lifecycle methods.
- `agents/planner.py` defines `PlannerAgent`, a no-LLM planning placeholder.
- `agents/memory_agent.py` defines `MemoryAgent`, a no-LLM memory placeholder.
- `agents/tool_agent.py` defines `ToolAgent`, a no-LLM tool execution placeholder.
- `agents/voice_agent.py` defines `VoiceAgent`, a no-LLM voice placeholder.
- `orchestrator/context.py` defines `SharedContext`, a thread-safe key-value store shared by agents.
- `orchestrator/message_bus.py` defines `MessageBus`, an async publish/subscribe bus for inter-agent communication.
- `orchestrator/task_queue.py` defines `TaskQueue`, an async queue for background task routing.
- `orchestrator/core.py` defines the `Orchestrator` registry, routing logic, lifecycle management, and queue integration.
- `orchestrator/exceptions.py` defines orchestration-specific exceptions.
- `llm/base.py` defines `BaseLLMProvider` and shared provider configuration.
- `llm/openai_provider.py` implements the OpenAI chat completions provider.
- `llm/ollama_provider.py` implements the local Ollama chat provider.
- `llm/factory.py` creates providers based on `.env` configuration.
- `llm/prompt_manager.py` stores and renders prompt templates.
- `llm/chat_session.py` manages conversation history and streaming sessions.
- `llm/errors.py` defines LLM-specific exceptions.
- `config/settings.py` loads `.env` and process environment configuration.
- `config/logging_config.py` defines centralized console and rotating file logging.

## Runtime Capabilities

- Register agents dynamically with `orchestrator.register(agent)`.
- Remove agents dynamically with `orchestrator.unregister("agent_name")`.
- Route tasks synchronously with `orchestrator.route(task)`.
- Queue tasks asynchronously with `await orchestrator.enqueue(task)`.
- Share state through `orchestrator.context`.
- Publish inter-agent messages through `orchestrator.message_bus`.
- Manage lifecycle with `initialize()`, `start()`, `stop()`, and `health_check()`.
- Switch LLM providers with `LLM_PROVIDER=openai` or `LLM_PROVIDER=ollama`.
- Enable planner LLM integration with `LLM_ENABLED=true`.

## LLM Configuration

Copy `.env.example` to `.env` and adjust values:

```text
LLM_ENABLED=false
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
LLM_BASE_URL=http://localhost:11434
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2
LLM_RETRY_BACKOFF_SECONDS=0.25
OPENAI_API_KEY=
```

When `LLM_ENABLED=false`, `PlannerAgent` keeps the deterministic placeholder
behavior used by earlier phases. When enabled, `PlannerAgent` renders the
planner prompt template and sends it through the configured provider.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

## Test

```powershell
pytest
```
