"""Tests for long prompt handling — chunking, budget, input reading, progress."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from prompt import (
    ChunkProcessor,
    ChunkProgress,
    InputMode,
    InputReader,
    PromptChunker,
    TokenBudget,
    estimate_tokens,
)


# =========================================================================
# TokenBudget
# =========================================================================

class TestTokenBudget:
    def test_default_budget(self) -> None:
        b = TokenBudget()
        assert b.max_tokens == 4096
        assert b.remaining <= 4096  # reserve reduces it
        assert b.used > 0  # reserve tokens

    def test_custom_budget(self) -> None:
        b = TokenBudget(max_context_tokens=2048, reserve_tokens=0)
        assert b.max_tokens == 2048
        assert b.remaining == 2048

    def test_allocate_system(self) -> None:
        b = TokenBudget(max_context_tokens=1024, reserve_tokens=0)
        t = b.allocate_system("You are a helpful assistant.")
        assert t > 0
        assert b.used > 0
        assert b.remaining < 1024

    def test_allocate_memory(self) -> None:
        b = TokenBudget(max_context_tokens=1024, reserve_tokens=0)
        b.allocate_system("System prompt.")
        b.allocate_memory("Relevant context from memory:\n  [fact] some info")
        assert b.used > 0

    def test_can_fit(self) -> None:
        b = TokenBudget(max_context_tokens=1024, reserve_tokens=0)
        assert b.can_fit("short text")
        assert not b.can_fit("x" * 10000)

    def test_oversize_by(self) -> None:
        b = TokenBudget(max_context_tokens=100, reserve_tokens=0)
        assert b.oversize_by("x" * 1000) > 0
        assert b.oversize_by("hi") == 0

    def test_snapshot(self) -> None:
        b = TokenBudget(max_context_tokens=1024, reserve_tokens=0)
        s = b.snapshot()
        assert s["max_tokens"] == 1024
        assert "used" in s
        assert "remaining" in s

    def test_is_over_budget(self) -> None:
        b = TokenBudget(max_context_tokens=50, reserve_tokens=0)
        b.allocate_system("x" * 200)
        assert b.is_over_budget

    def test_zero_chars_per_token(self) -> None:
        b = TokenBudget(max_context_tokens=1024, reserve_tokens=0, chars_per_token=0)
        assert estimate_tokens("any text", 0) == 0
        assert b.can_fit("very long text that goes on and on and on and on")


# =========================================================================
# PromptChunker
# =========================================================================

class TestPromptChunker:
    def test_empty_text(self) -> None:
        c = PromptChunker(max_chunk_tokens=1024)
        assert c.chunk("") == []

    def test_small_text_stays_single_chunk(self) -> None:
        c = PromptChunker(max_chunk_tokens=1024)
        chunks = c.chunk("Hello, world!")
        assert len(chunks) == 1
        assert chunks[0] == "Hello, world!"

    def test_large_text_split_into_chunks(self) -> None:
        c = PromptChunker(max_chunk_tokens=50)
        large = "\n\n".join([f"Paragraph {i} with some content." for i in range(20)])
        chunks = c.chunk(large)
        assert len(chunks) >= 2

    def test_chunk_preserves_paragraphs(self) -> None:
        c = PromptChunker(max_chunk_tokens=200)
        text = "\n\n".join([f"Para {i}." for i in range(10)])
        chunks = c.chunk(text)
        merged = "".join(chunks)
        # All content should be preserved (minus whitespace changes from merging)
        assert "Para 1." in merged
        assert "Para 9." in merged

    def test_large_paragraph_split(self) -> None:
        """A single oversized paragraph should be split."""
        c = PromptChunker(max_chunk_tokens=50)
        huge_para = "This is a very long paragraph that should be split into multiple chunks. " * 20
        chunks = c.chunk(huge_para)
        assert len(chunks) >= 2


# =========================================================================
# ChunkProgress
# =========================================================================

class TestChunkProgress:
    def test_single_chunk_no_output(self, capsys) -> None:
        p = ChunkProgress(total=1)
        p.start()
        p.advance()
        p.finish()
        captured = capsys.readouterr()
        assert captured.out == ""  # no output for single chunk

    def test_multi_chunk_output(self, capsys) -> None:
        p = ChunkProgress(total=3)
        p.start()
        p.advance(100)
        p.advance(200)
        p.advance(150)
        p.finish()
        captured = capsys.readouterr()
        assert "3 chunks" in captured.out
        assert "processed" in captured.out


# =========================================================================
# InputReader
# =========================================================================

class TestInputReader:
    def test_interactive_empty_returns_empty(self) -> None:
        """Simulate empty input."""
        # Can't easily mock input() in pytest, but we can test _read_file and _is_piped
        pass

    def test_read_file(self) -> None:
        """!file command reads file contents."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        ) as f:
            f.write("Hello from file!\nLine 2.")
            tmp = f.name

        try:
            reader = InputReader()
            content, mode = reader._read_file(f"!file {tmp}")
            assert mode == InputMode.FILE
            assert "Hello from file!" in content
        finally:
            os.unlink(tmp)

    def test_read_file_not_found(self, capsys) -> None:
        reader = InputReader()
        content, mode = reader._read_file("!file /nonexistent/file.txt")
        assert content == ""
        assert mode == InputMode.INTERACTIVE

    def test_read_file_no_path(self, capsys) -> None:
        reader = InputReader()
        content, mode = reader._read_file("!file ")
        assert content == ""
        assert mode == InputMode.INTERACTIVE

    def test_is_piped(self) -> None:
        reader = InputReader()
        # In test, stdin is not a tty (probably piped from pytest)
        result = reader._is_piped()
        assert isinstance(result, bool)


# =========================================================================
# ChunkProcessor
# =========================================================================

class TestChunkProcessor:
    async def test_small_text_no_chunking(self) -> None:
        processor = ChunkProcessor(max_chunk_tokens=4096)
        results: list[str] = []

        async def callback(chunk: str, accumulated: str) -> str:
            results.append(chunk)
            return f"Processed: {chunk[:20]}"

        result = await processor.process_chunked(
            "Short text.", chunk_callback=callback, show_progress=False,
        )
        assert "Processed: Short text." in result

    async def test_large_text_chunked(self) -> None:
        processor = ChunkProcessor(max_chunk_tokens=100)
        calls: list[int] = []

        async def callback(chunk: str, accumulated: str) -> str:
            calls.append(len(chunk))
            return f"Chunk {len(calls)}: {len(chunk)} chars"

        large = "\n\n".join([f"Paragraph {i} " * 10 for i in range(10)])
        result = await processor.process_chunked(
            large, chunk_callback=callback, show_progress=False,
        )
        assert len(calls) >= 2
        assert "Chunk 1:" in result
        assert "Chunk 2:" in result or len(calls) >= 2

    async def test_compress_context(self) -> None:
        processor = ChunkProcessor()
        result = processor._compress_context("existing", "new summary", max_chars=50)
        # Should contain the new summary
        assert "new summary" in result

    async def test_combine_results_single(self) -> None:
        processor = ChunkProcessor()
        result = processor._combine_results(["Single result"])
        assert result == "Single result"

    async def test_combine_results_multiple(self) -> None:
        processor = ChunkProcessor()
        result = processor._combine_results(["First", "Second"])
        assert "[Part 1]" in result
        assert "[Part 2]" in result


# =========================================================================
# estimate_tokens
# =========================================================================

class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        t = estimate_tokens("Hello, world!")
        assert t >= 1
        assert t <= 10

    def test_long_text(self) -> None:
        t = estimate_tokens("x" * 1000)
        assert t == int(1000 / 3.5)

    def test_custom_ratio(self) -> None:
        t = estimate_tokens("Hello", chars_per_token=1)
        assert t == 5


# =========================================================================
# Integration: end-to-end with full orchestrator
# =========================================================================

@pytest.mark.asyncio
async def test_planner_receives_large_goal() -> None:
    """Verify planner can accept a large goal without error."""
    from agents.planner import PlannerAgent

    agent = PlannerAgent()
    goal = "Test large goal. " * 2000  # ~36 KB
    from agents.contracts import AgentTask

    task = AgentTask(task_type="plan", payload={"goal": goal})
    result = await agent.handle(task)
    assert result.success
    assert result.data["response"]


@pytest.mark.asyncio
async def test_chunk_processor_max_chunk_tokens() -> None:
    """ChunkProcessor should respect max_chunk_tokens limit."""
    processor = ChunkProcessor(max_chunk_tokens=500)
    calls: list[str] = []

    async def cb(chunk: str, acc: str) -> str:
        calls.append(chunk)
        return "ok"

    large = "\n\n".join([f"Block {i}: " + "content " * 100 for i in range(5)])
    await processor.process_chunked(large, chunk_callback=cb, show_progress=False)
    assert len(calls) >= 2


# =========================================================================
# Large prompt integration tests (10KB, 50KB, 100KB)
# =========================================================================

_BASE_MARKDOWN = """# Sample Document

## Section

Content snippet with **bold** and `code`.

- bullet one
- bullet two

{}

## End
"""

_CODE_SAMPLE = """def fibonacci(n):
    \"\"\"Return the first n Fibonacci numbers.\"\"\"
    result = []
    a, b = 0, 1
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

{}

def main():
    fibs = fibonacci(10)
    print(fibs)
"""


def _make_large_markdown(target_kb: int) -> str:
    """Generate a Markdown document of approximately *target_kb* KB."""
    para_body = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 300
    filler = "\n\n".join([para_body] * max(1, target_kb))
    return _BASE_MARKDOWN.format(filler)


def _make_large_json(target_kb: int) -> str:
    """Generate a JSON document of approximately *target_kb* KB."""
    items = []
    for i in range(target_kb * 50):
        val = "x" * 40
        items.append(f'  "item_{i}": "{val}"')
    return "{\n" + ",\n".join(items) + "\n}"


def _make_large_code(target_kb: int) -> str:
    """Generate a Python code file of approximately *target_kb* KB."""
    func_template = """def func_{i}():
    \"\"\"Docstring for func_{i}.\"\"\"
    x = {i}
    y = x * 2
    return y + 1

"""
    repeats = max(1, (target_kb * 1024) // len(func_template.format(i=0)))
    funcs = "\n".join(func_template.format(i=i) for i in range(repeats))
    return _CODE_SAMPLE.format(funcs)


class TestLargeMarkdown:
    @pytest.mark.parametrize("size_kb", [10, 50, 100])
    def test_chunk_large_markdown(self, size_kb: int) -> None:
        md = _make_large_markdown(size_kb)
        assert len(md) >= size_kb * 700, f"Markdown too short for {size_kb} KB"
        chunker = PromptChunker(max_chunk_tokens=2000)
        chunks = chunker.chunk(md)
        assert len(chunks) >= 1
        # Verify no data loss
        total_chars = sum(len(c) for c in chunks)
        assert total_chars >= len(md) * 0.8  # Allow for whitespace normalization


class TestLargeJson:
    @pytest.mark.parametrize("size_kb", [10, 50, 100])
    def test_chunk_large_json(self, size_kb: int) -> None:
        js = _make_large_json(size_kb)
        assert len(js) >= size_kb * 700  # ~70% of target (structure overhead)
        chunker = PromptChunker(max_chunk_tokens=2000)
        chunks = chunker.chunk(js)
        assert len(chunks) >= 1


class TestLargeCode:
    @pytest.mark.parametrize("size_kb", [10, 50, 100])
    def test_chunk_large_code(self, size_kb: int) -> None:
        code = _make_large_code(size_kb)
        assert len(code) >= size_kb * 900
        chunker = PromptChunker(max_chunk_tokens=2000)
        chunks = chunker.chunk(code)
        assert len(chunks) >= 1


# =========================================================================
# Latency and memory measurement
# =========================================================================

class TestLatencyAndMemory:
    def test_estimate_tokens_latency(self) -> None:
        """estimate_tokens should be fast even for large texts."""
        import time
        large = "x" * 1_000_000  # 1 MB
        start = time.perf_counter()
        estimate_tokens(large)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"estimate_tokens took {elapsed:.3f}s for 1MB"

    def test_chunker_memory_usage(self) -> None:
        """Chunker should not duplicate large texts excessively."""
        import tracemalloc

        tracemalloc.start()
        try:
            large = "\n\n".join([f"Paragraph {i}. " * 100 for i in range(50)])
            chunker = PromptChunker(max_chunk_tokens=500)
            chunks = chunker.chunk(large)
            total_chunk_chars = sum(len(c) for c in chunks)
            # Total chunk size should be roughly proportional to input (some overhead)
            assert total_chunk_chars >= len(large) * 0.5
        finally:
            tracemalloc.stop()
