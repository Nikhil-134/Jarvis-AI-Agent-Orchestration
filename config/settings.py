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

    environment: str = "development"
    log_level: str = "INFO"
    llm_enabled: bool = False
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 0.25
    openai_api_key: str | None = None


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
        openai_api_key=merged.get("OPENAI_API_KEY") or None,
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
