"""Configuration package exports."""

from typing import Any

__all__ = ["Settings", "configure_logging", "load_settings"]


def __getattr__(name: str) -> Any:
    """Load configuration exports lazily so modules remain runnable with python -m."""
    if name == "configure_logging":
        from config.logging_config import configure_logging

        return configure_logging
    if name in {"Settings", "load_settings"}:
        from config.settings import Settings, load_settings

        return {"Settings": Settings, "load_settings": load_settings}[name]
    raise AttributeError(f"module 'config' has no attribute {name!r}")
