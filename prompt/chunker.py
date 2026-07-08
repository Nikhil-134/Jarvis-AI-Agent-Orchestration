"""Semantic text chunking for long prompts.

Splits text into chunks at natural boundaries (paragraphs, lines, sentence
boundaries) so that each chunk is below a given token budget.
"""

from __future__ import annotations

import re
from typing import Any

from prompt.budget import estimate_tokens


class PromptChunker:
    """Splits long text into semantic chunks respecting a token budget.

    Chunking strategy (in order of preference):
    1. Split on double-newline (paragraph) boundaries
    2. Split on single-newline boundaries
    3. Split on sentence boundaries (``. ! ?``)
    4. Split at character count as a last resort

    Usage::

        chunker = PromptChunker(max_chunk_tokens=2048)
        chunks = chunker.chunk(very_long_text)
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}/{len(chunks)}: {len(chunk)} chars")
    """

    def __init__(
        self,
        max_chunk_tokens: int = 2048,
        chars_per_token: float = 3.5,
        overlap_sentences: int = 1,
    ) -> None:
        self._max_tokens = max(1, max_chunk_tokens)
        self._max_chars = int(self._max_tokens * chars_per_token)
        self._overlap = max(0, overlap_sentences)
        self._chunk_count = 0

    @property
    def max_chunk_tokens(self) -> int:
        return self._max_tokens

    @property
    def chunk_count(self) -> int:
        """Number of chunks from the most recent :meth:`chunk` call, or 0."""
        return self._chunk_count

    def chunk(self, text: str) -> list[str]:
        """Split *text* into a list of chunk strings.

        Returns ``[text]`` (unchunked) when the entire text fits within
        the token budget.
        """
        if not text:
            return []

        if estimate_tokens(text, 3.5) <= self._max_tokens:
            self._chunk_count = 1
            return [text]

        # Try paragraph boundaries first, fall back through granularity levels.
        for level in ("paragraph", "line", "sentence", "char"):
            method = getattr(self, f"_split_by_{level}s")
            result = method(text)
            if result and self._all_fit(result):
                self._chunk_count = len(result)
                return result

        self._chunk_count = 1
        return [text]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_fit(self, segments: list[str]) -> bool:
        """Check if all *segments* fit within the max chunk size."""
        for seg in segments:
            if estimate_tokens(seg, 3.5) > self._max_tokens:
                return False
        return True

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split on one or more blank lines, then merge to fit budget."""
        paragraphs = re.split(r"\n\s*\n", text)
        return self._merge_to_budget(paragraphs, sep="\n\n")

    def _split_by_lines(self, text: str) -> list[str]:
        """Split on single newlines, then merge to fit budget."""
        lines = text.split("\n")
        return self._merge_to_budget(lines, sep="\n")

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split on sentence boundaries, then merge to fit budget."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return self._merge_to_budget(sentences, sep=" ")

    def _split_by_chars(self, text: str) -> list[str]:
        """Hard-split at character boundaries (last resort)."""
        if not self._max_chars:
            return [text]
        return [text[i : i + self._max_chars] for i in range(0, len(text), self._max_chars)]

    def _merge_to_budget(
        self, segments: list[str], sep: str = "\n\n",
    ) -> list[str]:
        """Merge *segments* into chunks that each fit within the max char limit.

        Segments that individually exceed the limit are kept as separate
        oversized chunks (they will be caught by *all_fit* and trigger
        the next granularity level).
        """
        result: list[str] = []
        current: list[str] = []
        current_size = 0

        for seg in segments:
            if not seg.strip():
                continue

            seg_size = len(seg)

            # Check if this segment alone exceeds the budget
            if seg_size > self._max_chars:
                # Flush existing buffer
                if current:
                    result.append(sep.join(current))
                    current = []
                    current_size = 0
                result.append(seg)  # keep as oversized
                continue

            # Check if adding would exceed budget
            joined = current_size + (len(sep) if current else 0) + seg_size
            if joined > self._max_chars:
                if current:
                    result.append(sep.join(current))
                current = [seg]
                current_size = seg_size
            else:
                current.append(seg)
                current_size = joined

        if current:
            result.append(sep.join(current))

        return result or [text for text in segments if text.strip()]
