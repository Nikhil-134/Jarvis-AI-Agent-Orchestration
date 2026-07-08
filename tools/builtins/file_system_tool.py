"""File System tool — consolidated file operations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec


class FileSystemTool(ITool):
    """Consolidated file and directory operations.

    Supports: read, write, create_folder, list_directory, move, copy,
    delete, search.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_system",
            description="Perform file system operations. Operations: read, write, create_folder, list_directory, move, copy, delete, search.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform: read, write, create_folder, list_directory, move, copy, delete, search.",
                        "enum": ["read", "write", "create_folder", "list_directory", "move", "copy", "delete", "search"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path for move/copy operations.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write operation).",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern for list_directory or search (e.g. '*.py', '**/*.txt').",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list recursively (default: false).",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results for search (default: 50).",
                    },
                },
                "required": ["operation", "path"],
            },
        )

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.DANGEROUS

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        operation = str(kwargs.get("operation", ""))
        path_str = str(kwargs.get("path", ""))

        if not operation:
            return {"success": False, "output": "", "error": "No operation specified."}
        if not path_str:
            return {"success": False, "output": "", "error": "No path provided."}

        path = Path(path_str).resolve()

        dispatch = {
            "read": self._read,
            "write": self._write,
            "create_folder": self._create_folder,
            "list_directory": self._list_directory,
            "move": self._move,
            "copy": self._copy,
            "delete": self._delete,
            "search": self._search,
        }

        handler = dispatch.get(operation)
        if handler is None:
            return {"success": False, "output": "", "error": f"Unknown operation: {operation}"}

        try:
            return await handler(path, kwargs)
        except Exception as exc:
            return {"success": False, "output": "", "error": f"{operation} failed: {exc}"}

    async def _read(self, path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return {"success": False, "output": "", "error": f"File not found: {path}"}
        if not path.is_file():
            return {"success": False, "output": "", "error": f"Not a file: {path}"}
        encoding = str(kwargs.get("encoding", "utf-8"))
        content = path.read_text(encoding=encoding)
        return {
            "success": True,
            "output": content,
            "data": {"path": str(path), "size_bytes": path.stat().st_size, "encoding": encoding},
        }

    async def _write(self, path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        content = str(kwargs.get("content", ""))
        encoding = str(kwargs.get("encoding", "utf-8"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return {
            "success": True,
            "output": f"Written {len(content.encode(encoding))} bytes to {path}",
            "data": {"path": str(path), "bytes_written": len(content.encode(encoding)), "encoding": encoding},
        }

    async def _create_folder(self, path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        path.mkdir(parents=True, exist_ok=True)
        return {"success": True, "output": f"Directory created: {path}", "data": {"path": str(path)}}

    async def _list_directory(self, path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return {"success": False, "output": "", "error": f"Path not found: {path}"}
        if not path.is_dir():
            return {"success": False, "output": "", "error": f"Not a directory: {path}"}
        pattern = str(kwargs.get("pattern", "*"))
        recursive = bool(kwargs.get("recursive", False))
        if recursive:
            entries = sorted(path.rglob(pattern))
        else:
            entries = sorted(path.glob(pattern))
        items = []
        for entry in entries:
            rel = entry.relative_to(path) if entry != path else entry
            items.append({
                "name": entry.name,
                "path": str(rel),
                "type": "directory" if entry.is_dir() else "file",
                "size_bytes": entry.stat().st_size if entry.is_file() else 0,
            })
        lines = [f"{'DIR' if i['type'] == 'directory' else '   '}  {i['path']}" for i in items]
        return {
            "success": True,
            "output": "\n".join(lines) if lines else "(empty directory)",
            "data": {"path": str(path), "count": len(items), "items": items},
        }

    async def _move(self, path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        dest = str(kwargs.get("destination", ""))
        if not dest:
            return {"success": False, "output": "", "error": "No destination provided for move."}
        if not path.exists():
            return {"success": False, "output": "", "error": f"Source not found: {path}"}
        dest_path = Path(dest).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest_path))
        return {"success": True, "output": f"Moved {path} to {dest_path}", "data": {"source": str(path), "destination": str(dest_path)}}

    async def _copy(self, path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        dest = str(kwargs.get("destination", ""))
        if not dest:
            return {"success": False, "output": "", "error": "No destination provided for copy."}
        if not path.exists():
            return {"success": False, "output": "", "error": f"Source not found: {path}"}
        dest_path = Path(dest).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(str(path), str(dest_path))
        return {"success": True, "output": f"Copied {path} to {dest_path}", "data": {"source": str(path), "destination": str(dest_path)}}

    async def _delete(self, path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return {"success": False, "output": "", "error": f"Path not found: {path}"}
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return {"success": True, "output": f"Deleted: {path}", "data": {"path": str(path)}}

    async def _search(self, path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return {"success": False, "output": "", "error": f"Path not found: {path}"}
        if not path.is_dir():
            return {"success": False, "output": "", "error": f"Not a directory: {path}"}
        pattern = str(kwargs.get("pattern", ""))
        if not pattern:
            return {"success": False, "output": "", "error": "No pattern provided for search."}
        max_results = int(kwargs.get("max_results", 50))
        matches = sorted(path.rglob(pattern))[:max_results]
        items = []
        for m in matches:
            rel = m.relative_to(path)
            items.append({
                "name": m.name,
                "path": str(rel),
                "type": "directory" if m.is_dir() else "file",
                "size_bytes": m.stat().st_size if m.is_file() else 0,
            })
        total = len(items)
        summary = f"Found {total} result(s)"
        if total >= max_results:
            summary += f" (limited to {max_results})"
        lines = [f"{'DIR' if i['type'] == 'directory' else '   '}  {i['path']}" for i in items]
        return {
            "success": True,
            "output": f"{summary}\n" + ("\n".join(lines) if lines else "(no matches)"),
            "data": {"root": str(path), "pattern": pattern, "count": total, "limited": total >= max_results, "items": items},
        }
