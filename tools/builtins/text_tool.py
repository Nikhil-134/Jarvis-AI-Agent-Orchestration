"""Text tool — text analysis and manipulation operations."""

from __future__ import annotations

import re
from typing import Any

from tools.interfaces import ITool, PermissionLevel, ToolSpec


class TextTool(ITool):
    """Text processing operations: summarize, word_count, char_count, regex_search."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="text",
            description="Text processing operations: summarize, word_count, char_count, regex_search.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation: summarize, word_count, char_count, regex_search.",
                        "enum": ["summarize", "word_count", "char_count", "regex_search"],
                    },
                    "text": {
                        "type": "string",
                        "description": "The input text to process.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern (for regex_search operation).",
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum summary length in characters (default: 200).",
                    },
                },
                "required": ["operation", "text"],
            },
        )

    @property
    def category(self) -> str:
        return "utility"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        operation = str(kwargs.get("operation", ""))
        text = str(kwargs.get("text", ""))

        if not operation:
            return {"success": False, "output": "", "error": "No operation specified."}
        if not text and operation not in ("word_count", "char_count"):
            return {"success": False, "output": "", "error": "No text provided."}

        if operation == "word_count":
            words = text.split()
            return {
                "success": True,
                "output": str(len(words)),
                "data": {"word_count": len(words), "character_count": len(text)},
            }

        if operation == "char_count":
            char_count = len(text)
            char_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", ""))
            return {
                "success": True,
                "output": str(char_count),
                "data": {"character_count": char_count, "character_count_no_spaces": char_no_spaces},
            }

        if operation == "regex_search":
            pattern = str(kwargs.get("pattern", ""))
            if not pattern:
                return {"success": False, "output": "", "error": "No regex pattern provided."}
            try:
                matches = re.findall(pattern, text)
                count = len(matches)
                preview = matches[:20]
                return {
                    "success": True,
                    "output": f"Found {count} match(es): {', '.join(str(m) for m in preview)}",
                    "data": {"pattern": pattern, "count": count, "matches": preview, "truncated": count > 20},
                }
            except re.error as exc:
                return {"success": False, "output": "", "error": f"Invalid regex pattern: {exc}"}

        if operation == "summarize":
            max_length = int(kwargs.get("max_length", 200))
            summary = self._summarize(text, max_length)
            return {
                "success": True,
                "output": summary,
                "data": {"original_length": len(text), "summary_length": len(summary)},
            }

        return {"success": False, "output": "", "error": f"Unknown operation: {operation}"}

    @staticmethod
    def _summarize(text: str, max_length: int = 200) -> str:
        """Simple extractive summarization: returns first N characters or sentences."""
        if len(text) <= max_length:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)
        result: list[str] = []
        for sent in sentences:
            candidate = " ".join(result + [sent])
            if len(candidate) > max_length and result:
                break
            result.append(sent)

        summary = " ".join(result)
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(" ", 1)[0] + "..."

        return summary if summary else text[:max_length] + "..."
