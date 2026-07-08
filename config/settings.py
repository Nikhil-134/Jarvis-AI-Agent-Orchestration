"""Application settings loaded from environment and .env files."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for Jarvis.

    All values are sourced from ``.env`` and/or process environment
    variables, with sensible defaults for local-first development.
    """

    # General
    environment: str = "development"
    log_level: str = "INFO"

    # LLM
    llm_enabled: bool = False
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 0.25
    llm_max_tokens: int | None = 4096
    openai_api_key: str | None = None

    # Tools
    tool_enabled: bool = True
    tool_auto_approve: bool = False
    tool_plugin_dirs: str = "plugins/tools"
    tool_enabled_tools: str = ""
    tool_disabled_tools: str = ""
    tool_default_timeout: float = 30.0
    tool_store_results: bool = True

    # Memory
    memory_enabled: bool = True
    memory_vector_store_path: str = "./memory_data/vectors"
    memory_document_store_path: str = "./memory_data/documents.db"
    memory_embedding_model: str = "all-MiniLM-L6-v2"
    memory_search_top_k: int = 5
    memory_importance_threshold: float = 0.3
    memory_dedup_threshold: float = 0.95
    memory_cleanup_max_age_days: int = 30
    memory_working_memory_size: int = 50
    memory_working_memory_ttl_seconds: int = 3600
    # Cross-session persistent memory (record turns, restore session on boot,
    # reflect on exit). Off → text mode behaves exactly as before.
    memory_persist_enabled: bool = True
    memory_session_id: str = "default"
    # Auto-promote preferences stated in conversation ("call me Boss", "my
    # favourite language is Rust") to structured profile entries — no explicit
    # "remember" command needed (roadmap #11). Off → turns are still recorded,
    # just not distilled into structured preferences.
    memory_auto_learn_enabled: bool = True

    # Internet Knowledge Engine (roadmap #9) — context-only, last-resort
    # retrieval from public JSON APIs (DuckDuckGo + Wikipedia). Reasoning stays
    # local. Off → behaviour is byte-for-byte the local-only path.
    internet_enabled: bool = True
    internet_timeout_seconds: float = 6.0        # per-request (SafeHttpClient)
    internet_overall_timeout_seconds: float = 8.0  # whole parallel fan-out
    internet_cache_ttl_seconds: float = 300.0    # short TTL for volatile facts
    internet_min_interval_seconds: float = 1.0   # rate limit between live fetches
    internet_max_results: int = 5                # snippets injected as context

    # Planning & Task Execution subsystem (cycle 8) — decomposes complex,
    # actionable goals into a dependency task graph and executes them
    # concurrently with retries/timeouts/cancellation, verifying the final
    # answer. Confidence-based routing supersedes regex for actionable goals;
    # regex routing remains the fallback. Off → the runtime routes exactly as
    # before (regex path only).
    planning_enabled: bool = True
    planning_max_parallel: int = 4               # concurrent task workers
    planning_task_timeout_seconds: float = 30.0  # per-task timeout (authoritative)
    planning_max_retries: int = 1                # retries per task on failure/timeout
    planning_confidence_threshold: float = 0.55  # verifier min confidence
    planning_min_goal_confidence: float = 0.5    # decline plans below this (→ fallback)
    planning_telemetry_enabled: bool = True      # structured JSON telemetry (local log)

    # Prompt / context budget
    prompt_max_context_tokens: int = 4096
    prompt_max_chunk_tokens: int = 2048
    prompt_chars_per_token: float = 3.5
    prompt_max_history_tokens: int = 2048
    prompt_memory_context_chars: int = 5000
    prompt_per_memory_chars: int = 2000

    # Voice (100% local; disabled by default so text mode is unaffected)
    voice_enabled: bool = False
    voice_tts_provider: str = "piper"  # piper (local) | edge (online, non-default)
    voice_piper_model: str = "./voice_models/en_US-lessac-medium.onnx"
    voice_stt_model: str = "base"  # faster-whisper model size
    voice_wake_words: str = "jarvis,computer"
    voice_record_seconds: float = 5.0  # push-to-talk (!voice) capture length

    # Continuous voice loop (hands-free)
    voice_wake_mode: str = "transcript"  # transcript (wake word) | none (always on)
    voice_frame_ms: int = 30             # VAD frame size
    voice_trailing_silence: float = 0.8  # silence (s) that ends an utterance
    voice_max_utterance: float = 15.0    # hard cap (s) on one utterance
    voice_inactivity_timeout: float = 30.0  # active→asleep after this quiet (s)
    voice_greeting: str = "Yes?"         # spoken on wake
    voice_autostart: bool = False        # launch straight into continuous loop


def load_settings(env_file: str | Path = ".env") -> Settings:
    """Load settings from a .env file and process environment variables."""
    values = _read_env_file(Path(env_file))
    merged = {**values, **os.environ}

    return Settings(
        environment=merged.get("ENVIRONMENT", "development"),
        log_level=merged.get("LOG_LEVEL", "INFO"),
        llm_enabled=_to_bool(merged.get("LLM_ENABLED", "false")),
        llm_provider=merged.get("LLM_PROVIDER", "ollama").lower(),
        llm_model=merged.get("LLM_MODEL", "llama3.1"),
        llm_base_url=merged.get("LLM_BASE_URL") or None,
        llm_timeout_seconds=float(merged.get("LLM_TIMEOUT_SECONDS", "30")),
        llm_max_retries=int(merged.get("LLM_MAX_RETRIES", "2")),
        llm_retry_backoff_seconds=float(merged.get("LLM_RETRY_BACKOFF_SECONDS", "0.25")),
        llm_max_tokens=_to_int_or_none(merged.get("LLM_MAX_TOKENS", "4096")),
        openai_api_key=merged.get("OPENAI_API_KEY") or None,
        tool_enabled=_to_bool(merged.get("TOOL_ENABLED", "true")),
        tool_auto_approve=_to_bool(merged.get("TOOL_AUTO_APPROVE", "false")),
        tool_plugin_dirs=merged.get("TOOL_PLUGIN_DIRS", "plugins/tools"),
        tool_enabled_tools=merged.get("TOOL_ENABLED_TOOLS", ""),
        tool_disabled_tools=merged.get("TOOL_DISABLED_TOOLS", ""),
        tool_default_timeout=float(merged.get("TOOL_DEFAULT_TIMEOUT", "30")),
        tool_store_results=_to_bool(merged.get("TOOL_STORE_RESULTS", "true")),
        memory_enabled=_to_bool(merged.get("MEMORY_ENABLED", "true")),
        memory_vector_store_path=merged.get("MEMORY_VECTOR_STORE_PATH", "./memory_data/vectors"),
        memory_document_store_path=merged.get("MEMORY_DOCUMENT_STORE_PATH", "./memory_data/documents.db"),
        memory_embedding_model=merged.get("MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        memory_search_top_k=int(merged.get("MEMORY_SEARCH_TOP_K", "5")),
        memory_importance_threshold=float(merged.get("MEMORY_IMPORTANCE_THRESHOLD", "0.3")),
        memory_dedup_threshold=float(merged.get("MEMORY_DEDUP_THRESHOLD", "0.95")),
        memory_cleanup_max_age_days=int(merged.get("MEMORY_CLEANUP_MAX_AGE_DAYS", "30")),
        memory_working_memory_size=int(merged.get("MEMORY_WORKING_MEMORY_SIZE", "50")),
        memory_working_memory_ttl_seconds=int(merged.get("MEMORY_WORKING_MEMORY_TTL_SECONDS", "3600")),
        memory_persist_enabled=_to_bool(merged.get("MEMORY_PERSIST_ENABLED", "true")),
        memory_session_id=merged.get("MEMORY_SESSION_ID", "default"),
        memory_auto_learn_enabled=_to_bool(merged.get("MEMORY_AUTO_LEARN_ENABLED", "true")),
        internet_enabled=_to_bool(merged.get("INTERNET_ENABLED", "true")),
        internet_timeout_seconds=float(merged.get("INTERNET_TIMEOUT_SECONDS", "6")),
        internet_overall_timeout_seconds=float(merged.get("INTERNET_OVERALL_TIMEOUT_SECONDS", "8")),
        internet_cache_ttl_seconds=float(merged.get("INTERNET_CACHE_TTL_SECONDS", "300")),
        internet_min_interval_seconds=float(merged.get("INTERNET_MIN_INTERVAL_SECONDS", "1")),
        internet_max_results=int(merged.get("INTERNET_MAX_RESULTS", "5")),
        planning_enabled=_to_bool(merged.get("PLANNING_ENABLED", "true")),
        planning_max_parallel=int(merged.get("PLANNING_MAX_PARALLEL", "4")),
        planning_task_timeout_seconds=float(merged.get("PLANNING_TASK_TIMEOUT_SECONDS", "30")),
        planning_max_retries=int(merged.get("PLANNING_MAX_RETRIES", "1")),
        planning_confidence_threshold=float(merged.get("PLANNING_CONFIDENCE_THRESHOLD", "0.55")),
        planning_min_goal_confidence=float(merged.get("PLANNING_MIN_GOAL_CONFIDENCE", "0.5")),
        planning_telemetry_enabled=_to_bool(merged.get("PLANNING_TELEMETRY_ENABLED", "true")),
        prompt_max_context_tokens=int(merged.get("PROMPT_MAX_CONTEXT_TOKENS", "4096")),
        prompt_max_chunk_tokens=int(merged.get("PROMPT_MAX_CHUNK_TOKENS", "2048")),
        prompt_chars_per_token=float(merged.get("PROMPT_CHARS_PER_TOKEN", "3.5")),
        prompt_max_history_tokens=int(merged.get("PROMPT_MAX_HISTORY_TOKENS", "2048")),
        prompt_memory_context_chars=int(merged.get("PROMPT_MEMORY_CONTEXT_CHARS", "5000")),
        prompt_per_memory_chars=int(merged.get("PROMPT_PER_MEMORY_CHARS", "2000")),
        voice_enabled=_to_bool(merged.get("VOICE_ENABLED", "false")),
        voice_tts_provider=merged.get("VOICE_TTS_PROVIDER", "piper").lower(),
        voice_piper_model=merged.get("VOICE_PIPER_MODEL", "./voice_models/en_US-lessac-medium.onnx"),
        voice_stt_model=merged.get("VOICE_STT_MODEL", "base"),
        voice_wake_words=merged.get("VOICE_WAKE_WORDS", "jarvis,computer"),
        voice_record_seconds=float(merged.get("VOICE_RECORD_SECONDS", "5.0")),
        voice_wake_mode=merged.get("VOICE_WAKE_MODE", "transcript").lower(),
        voice_frame_ms=int(merged.get("VOICE_FRAME_MS", "30")),
        voice_trailing_silence=float(merged.get("VOICE_TRAILING_SILENCE", "0.8")),
        voice_max_utterance=float(merged.get("VOICE_MAX_UTTERANCE", "15.0")),
        voice_inactivity_timeout=float(merged.get("VOICE_INACTIVITY_TIMEOUT", "30.0")),
        voice_greeting=merged.get("VOICE_GREETING", "Yes?"),
        voice_autostart=_to_bool(merged.get("VOICE_AUTOSTART", "false")),
    )


def _read_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE pairs from a .env file."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _to_bool(value: str) -> bool:
    """Parse common truthy environment values."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int_or_none(value: str) -> int | None:
    """Parse an integer from *value*, returning None for empty/zero."""
    stripped = value.strip()
    if not stripped or stripped == "0":
        return None
    return int(stripped)
