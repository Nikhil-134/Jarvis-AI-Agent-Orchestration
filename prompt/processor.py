"""Chunk processor — processes long prompts in chunks with progress.

When a user prompt exceeds the LLM context window, the chunk processor:
1. Splits the prompt into semantic chunks
2. Processes each chunk through the LLM individually
3. Passes accumulated context between chunks
4. Preserves formatting and important details
5. Displays progress for multi-chunk operations
"""

from __future__ import annotations

import logging
from typing import Any

from prompt.budget import TokenBudget
from prompt.chunker import PromptChunker
from prompt.progress import ChunkProgress

_logger = logging.getLogger(__name__)


class ChunkProcessor:
    """Processes long prompts as a sequence of chunks with context passing.

    Usage::

        processor = ChunkProcessor(
            max_chunk_tokens=2048,
            llm_provider=provider,
        )
        result = await processor.process(
            goal="very long text...",
            deadline=lambda: f"Process this part: {chunk}",
            show_progress=True,
        )
    """

    def __init__(
        self,
        max_chunk_tokens: int = 2048,
        chars_per_token: float = 3.5,
        overlap_sentences: int = 1,
    ) -> None:
        self._max_chunk_tokens = max(1, max_chunk_tokens)
        self._chunker = PromptChunker(
            max_chunk_tokens=max_chunk_tokens,
            chars_per_token=chars_per_token,
            overlap_sentences=overlap_sentences,
        )
        self._cpt = chars_per_token

    async def process_chunked(
        self,
        goal: str,
        *,
        chunk_callback,
        show_progress: bool = True,
    ) -> str:
        """Process *goal* in chunks, calling *chunk_callback* for each.

        *chunk_callback* receives ``(chunk_text, accumulated_context)``
        and must return a ``str`` with the chunk's result/insight.

        Returns the concatenated result from all chunks.
        """
        chunks = self._chunker.chunk(goal)
        if not chunks:
            return ""
        if len(chunks) == 1:
            return await chunk_callback(chunks[0], "")

        progress = ChunkProgress(len(chunks)) if show_progress else None

        if progress:
            progress.start()

        accumulated = ""
        results: list[str] = []

        for i, chunk in enumerate(chunks):
            _logger.debug("Processing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))

            if progress:
                progress.advance(len(chunk))

            summary = await chunk_callback(chunk, accumulated)
            results.append(summary)

            # Update accumulated context
            accumulated = self._compress_context(accumulated, summary)
            _logger.debug("Accumulated context now %d chars", len(accumulated))

        if progress:
            progress.finish()

        return self._combine_results(results)

    @staticmethod
    def _compress_context(
        existing: str, new_summary: str, max_chars: int = 2000
    ) -> str:
        """Combine existing context with new summary, truncating if needed."""
        combined = existing + "\n\n[Continued]\n" + new_summary
        if len(combined) > max_chars:
            # Keep the most recent context
            combined = combined[-max_chars:]
        return combined

    @staticmethod
    def _combine_results(results: list[str]) -> str:
        """Combine per-chunk results into a single coherent response."""
        if len(results) == 1:
            return results[0]
        return "\n\n".join(
            f"[Part {i + 1}]\n{r}" for i, r in enumerate(results)
        )
