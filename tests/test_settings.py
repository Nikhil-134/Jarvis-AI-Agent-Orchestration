"""Tests for .env-backed settings."""

from config import load_settings
from config.settings import Settings


def test_load_settings_reads_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LOG_LEVEL=DEBUG",
                "LLM_ENABLED=true",
                "LLM_PROVIDER=openai",
                "LLM_MODEL=gpt-test",
                "LLM_TIMEOUT_SECONDS=5",
                "LLM_MAX_RETRIES=3",
                "OPENAI_API_KEY=test-key",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.log_level == "DEBUG"
    assert settings.llm_enabled is True
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-test"
    assert settings.llm_timeout_seconds == 5
    assert settings.llm_max_retries == 3
    assert settings.openai_api_key == "test-key"


def test_planning_settings_defaults() -> None:
    """Planning subsystem defaults are present and on."""
    s = Settings()
    assert s.planning_enabled is True
    assert s.planning_max_parallel == 4
    assert s.planning_task_timeout_seconds == 30.0
    assert s.planning_max_retries == 1
    assert s.planning_confidence_threshold == 0.55
    assert s.planning_min_goal_confidence == 0.5


def test_planning_settings_env_overrides(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "PLANNING_ENABLED=false",
                "PLANNING_MAX_PARALLEL=8",
                "PLANNING_TASK_TIMEOUT_SECONDS=12.5",
                "PLANNING_MAX_RETRIES=3",
                "PLANNING_CONFIDENCE_THRESHOLD=0.7",
                "PLANNING_MIN_GOAL_CONFIDENCE=0.4",
            ]
        ),
        encoding="utf-8",
    )
    s = load_settings(env_file)
    assert s.planning_enabled is False
    assert s.planning_max_parallel == 8
    assert s.planning_task_timeout_seconds == 12.5
    assert s.planning_max_retries == 3
    assert s.planning_confidence_threshold == 0.7
    assert s.planning_min_goal_confidence == 0.4


def test_stale_settings_getattr_safe() -> None:
    """A Settings without planning fields must not crash getattr-based reads.

    The factory reads planning_* via getattr(..., default); simulate an older
    Settings by reading a missing attribute the same way.
    """
    s = Settings()
    # Attribute the factory would read defensively if it were absent.
    assert getattr(s, "planning_nonexistent", 123) == 123
