"""Plugin-based tool discovery for Jarvis.

Provides automatic discovery and loading of external tool plugins from
designated directories, using a simple entry-point convention.

Plugin loading architecture
---------------------------
1. **Directory scanning** — walks ``plugin_dirs`` for Python files or
   packages that expose a ``register_tools(registry)`` function.

2. **Registration** — each plugin receives an :class:`IToolRegistry`
   instance and registers its tools via the standard API.

3. **Enable / disable** — plugins can be enabled or disabled via a
   simple list of names; disabled plugins are skipped during discovery.

This architecture is intentionally simple.  A future phase may replace
it with standard Python `entry_points` or a full plugin engine.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from tools.interfaces import IToolRegistry

_logger = logging.getLogger(__name__)


def discover_plugins(
    registry: IToolRegistry,
    plugin_dirs: list[str | Path] | None = None,
    enabled_plugins: list[str] | None = None,
    disabled_plugins: list[str] | None = None,
) -> list[str]:
    """Scan *plugin_dirs* for tool plugins and register their tools.

    Each directory is scanned for ``.py`` files (excluding ``__init__``).
    Files named ``*_plugin.py`` or ``plugin_*.py`` are treated as plugin
    modules and imported.  If a module exposes a top-level
    ``register_tools(registry)`` function it is called with the registry.

    Parameters
    ----------
    registry:
        Tool registry to register plugins into.
    plugin_dirs:
        Directories to scan.  Defaults to ``["plugins/tools"]``.
    enabled_plugins:
        If set, only these plugin names (module stem) are loaded.
    disabled_plugins:
        If set, these plugin names (module stem) are skipped.

    Returns
    -------
    List of successfully loaded plugin module names.
    """
    if plugin_dirs is None:
        plugin_dirs = [Path("plugins/tools")]

    enabled = set(enabled_plugins or [])
    disabled = set(disabled_plugins or [])
    loaded: list[str] = []

    for directory in plugin_dirs:
        plugin_path = Path(directory).resolve()
        if not plugin_path.is_dir():
            _logger.debug("Plugin directory '%s' does not exist, skipping", directory)
            continue

        for entry in sorted(plugin_path.iterdir()):
            if entry.suffix != ".py" or entry.stem == "__init__":
                continue
            if not entry.stem.endswith("_plugin") and not entry.stem.startswith("plugin_"):
                continue

            name = entry.stem
            if enabled and name not in enabled:
                _logger.debug("Plugin '%s' not in enabled list, skipping", name)
                continue
            if name in disabled:
                _logger.debug("Plugin '%s' is disabled, skipping", name)
                continue

            try:
                _load_plugin(entry, registry)
                loaded.append(name)
                _logger.info("Loaded tool plugin '%s' from %s", name, entry)
            except Exception:
                _logger.exception("Failed to load plugin '%s' from %s", name, entry)

    return loaded


def _load_plugin(entry: Path, registry: IToolRegistry) -> None:
    """Import a single plugin file and call ``register_tools(registry)``."""
    spec = importlib.util.spec_from_file_location(entry.stem, entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {entry}")

    module = importlib.util.module_from_spec(spec)
    # Ensure the module's parent package is recognised
    sys.modules[entry.stem] = module
    spec.loader.exec_module(module)

    register_fn = getattr(module, "register_tools", None)
    if register_fn is None:
        _logger.warning(
            "Plugin '%s' has no register_tools(registry) function", entry.stem
        )
        return

    result = register_fn(registry)
    if result is not None:
        _logger.debug("Plugin '%s' register_tools returned: %s", entry.stem, result)
