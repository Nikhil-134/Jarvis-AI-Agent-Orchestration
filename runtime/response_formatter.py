"""Response Formatter — formats responses as clean markdown with proper structure.

Ensures all responses are:
- Properly formatted markdown
- Free of raw JSON, tracebacks, internal prompts, or function calls
- Consistent in style (headings, lists, code blocks, tables)
"""

from __future__ import annotations

import logging
import re

_logger = logging.getLogger(__name__)

_JSON_PATTERN = re.compile(r'^[\s\[\{]*("(?:[^"\\]|\\.)*"|\d+|true|false|null)\s*[\]\}]?\s*$')

_INTERNAL_PROMPT_PATTERNS = [
    re.compile(r"\b(You are (Jarvis|an AI|a helpful|a planning|a task decomposition))\b", re.IGNORECASE),
    re.compile(r"\b(instruction|system prompt|user prompt):.*", re.IGNORECASE),
    re.compile(r"^Query is required$", re.IGNORECASE),
]

_TOOL_CALL_PATTERN = re.compile(
    r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\}',
    re.DOTALL,
)

_TRACEBACK_PATTERN = re.compile(
    r"Traceback \(most recent call last\):\n.*?\n\w+(?:Error|Exception):",
    re.DOTALL,
)


class ResponseFormatter:
    """Formats response strings as clean, well-structured markdown.

    Handles code blocks, lists, tables, error messages, and tool summaries.
    Strips any internal artifacts (JSON, tracebacks, prompts) that leak through.
    """

    def format(self, text: str) -> str:
        """Format *text* as clean markdown, stripping internal artifacts."""
        if not text:
            return ""

        cleaned = self._strip_internal_artifacts(text)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._format_code_blocks(cleaned)
        return cleaned.strip()

    def format_error(self, error_message: str) -> str:
        """Format an error message as a user-friendly notice."""
        return f"⚠️ {error_message}"

    def format_list(self, items: list[str], ordered: bool = False) -> str:
        """Format a list of items as markdown."""
        if not items:
            return ""
        lines: list[str] = []
        for i, item in enumerate(items, 1):
            prefix = f"{i}." if ordered else "-"
            lines.append(f"{prefix} {item}")
        return "\n".join(lines)

    def format_code_block(self, code: str, language: str = "") -> str:
        """Format code as a fenced code block."""
        lang = language if language else ""
        return f"```{lang}\n{code}\n```"

    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Format tabular data as a markdown table."""
        if not headers or not rows:
            return ""
        sep = "|" + "|".join("---" for _ in headers) + "|"
        header_line = "|" + "|".join(headers) + "|"
        body = "\n".join(
            "|" + "|".join(row) + "|" for row in rows
        )
        return f"{header_line}\n{sep}\n{body}"

    def format_tool_summary(self, tool_name: str, output: str) -> str:
        """Format a single tool's output as a compact summary."""
        cleaned = self._strip_internal_artifacts(output)
        preview = cleaned[:300] if len(cleaned) > 300 else cleaned
        return f"> **{tool_name}**: {preview}"

    def _strip_internal_artifacts(self, text: str) -> str:
        """Remove any internal artifacts that leaked into the response."""
        if not text:
            return text

        stripped = text

        for pattern in _INTERNAL_PROMPT_PATTERNS:
            stripped = pattern.sub("", stripped)

        stripped = _TOOL_CALL_PATTERN.sub("", stripped)
        stripped = _TRACEBACK_PATTERN.sub("", stripped)

        stripped = re.sub(
            r"data\s*=\s*\{.*?['\"]response['\"]:\s*", "", stripped, count=1, flags=re.DOTALL,
        )

        stripped = re.sub(r"AgentResult\(.*?\)", "", stripped)

        stripped = re.sub(
            r"['\"]memory_enriched['\"]:\s*(True|False),\s*['\"]memory_count['\"]:\s*\d+",
            "", stripped,
        )

        stripped = stripped.replace("\\n", "\n")

        return stripped.strip()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace: collapse multiple blank lines, trim."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +\n", "\n", text)
        return text.strip()

    @staticmethod
    def _format_code_blocks(text: str) -> str:
        """Ensure code blocks are properly fenced."""
        if "```" not in text:
            lines = text.split("\n")
            in_code = False
            formatted: list[str] = []
            for line in lines:
                if line.startswith("    ") and not in_code:
                    in_code = True
                    formatted.append("```")
                    formatted.append(line[4:])
                elif in_code and not line.startswith("    "):
                    in_code = False
                    formatted.append("```")
                    formatted.append(line)
                elif in_code:
                    formatted.append(line[4:] if line.startswith("    ") else line)
                else:
                    formatted.append(line)
            if in_code:
                formatted.append("```")
            return "\n".join(formatted)
        return text

    @staticmethod
    def is_response_clean(text: str) -> bool:
        """Check if a response is free of internal artifacts.

        Tool call JSON patterns are silently removed by format() — they
        are not considered a cleanliness failure unless they remain after
        stripping.
        """
        if _TRACEBACK_PATTERN.search(text):
            return False
        if any(p.search(text) for p in _INTERNAL_PROMPT_PATTERNS):
            return False
        if "AgentResult(" in text:
            return False
        if "Tool execution completed" in text:
            return False
        return True
