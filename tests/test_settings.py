"""Tests for .env-backed settings."""

from config import load_settings


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
