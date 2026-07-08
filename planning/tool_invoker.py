"""ToolInvoker — executes one task by delegating to a real backend.

This is the confidence-based routing endpoint: a :class:`TaskNode` names a
``required_tool`` (capability); the invoker resolves it via the
:class:`CapabilityCatalog` and calls the **existing** subsystem that actually
does the work.  It never reimplements Memory, Internet, the Tool engine, or the
language model — it only delegates:

======================  ===========================================
capability / backend    delegated call
======================  ===========================================
memory                  MemoryService.enrich_prompt(...)
internet                InternetKnowledgeService.build_context(...)   (needs_internet-gated)
reasoning               KnowledgeEngine.answer(...) or LLMGuard.generate(...)
calculator/filesystem/  ToolExecutionEngine.execute(tool, ToolContext(timeout), **args)
browser/desktop/…
python                  (unavailable — no safe local backend)
======================  ===========================================

Guarantees
----------
* **Never raises** — every failure becomes a :class:`NodeResult` with a
  terminal status and a safe message.
* **Single authoritative timeout** — the per-task timeout is pushed *into*
  ``ToolExecutionEngine.execute`` via ``ToolContext`` so it isn't masked by the
  engine's own default; the executor's outer ``wait_for`` only backstops the
  service backends (which honour their own internal timeouts).
* **Honest unavailability** — a missing dep / unregistered tool / DANGEROUS
  tool without auto-approve yields ``FAILED`` with a user-safe message, not a
  fabricated success.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from planning.capabilities import (
    BACKEND_INTERNET,
    BACKEND_MEMORY,
    BACKEND_NONE,
    BACKEND_REASONING,
    BACKEND_TOOL,
    CapabilityCatalog,
)
from planning.interfaces import IToolInvoker
from planning.models import NodeResult, TaskNode, TaskStatus
from tools.context import ToolContext

if TYPE_CHECKING:
    from knowledge.internet import InternetKnowledgeService
    from memory import MemoryService
    from runtime.knowledge_engine import KnowledgeEngine
    from runtime.llm_guard import LLMGuard
    from tools.engine import ToolExecutionEngine

_logger = logging.getLogger(__name__)


class ToolInvoker(IToolInvoker):
    """Runs a single :class:`TaskNode` against a real backend. Never raises."""

    def __init__(
        self,
        catalog: CapabilityCatalog,
        *,
        tool_engine: "ToolExecutionEngine | None" = None,
        memory_service: "MemoryService | None" = None,
        internet_service: "InternetKnowledgeService | None" = None,
        knowledge_engine: "KnowledgeEngine | None" = None,
        llm_guard: "LLMGuard | None" = None,
        task_timeout_seconds: float = 30.0,
        reasoning_system_prompt: str | None = None,
    ) -> None:
        self._catalog = catalog
        self._tool_engine = tool_engine
        self._memory = memory_service
        self._internet = internet_service
        self._knowledge = knowledge_engine
        self._guard = llm_guard
        self._timeout = task_timeout_seconds
        self._reasoning_system_prompt = reasoning_system_prompt or (
            "You are Jarvis. Complete the described step concisely and directly. "
            "Use only the provided context; never mention tools or internal machinery."
        )

    async def invoke(self, node: TaskNode, context: str = "") -> NodeResult:
        """Execute *node*, returning a terminal :class:`NodeResult`."""
        start = time.monotonic()
        cap = self._catalog.resolve(node.required_tool)

        if not cap.available:
            return self._unavailable(node, cap.backend, cap.reason, start)

        try:
            if cap.backend == BACKEND_MEMORY:
                output = await self._run_memory(node)
            elif cap.backend == BACKEND_INTERNET:
                output = await self._run_internet(node, local_context_found=bool(context))
            elif cap.backend == BACKEND_REASONING:
                output = await self._run_reasoning(node, context)
            elif cap.backend == BACKEND_TOOL:
                return await self._run_tool(node, cap, start)
            else:  # BACKEND_NONE
                return self._unavailable(node, cap.backend, cap.reason, start)
        except Exception as exc:  # noqa: BLE001 - invoker must never raise
            _logger.exception("Invoker backend %s raised for node %s", cap.backend, node.id)
            return self._result(node, TaskStatus.FAILED, cap.backend, start,
                                 error=str(exc), output="")

        return self._result(node, TaskStatus.SUCCEEDED, cap.backend, start, output=output)

    # ------------------------------------------------------------------
    # Service-backed capabilities (delegation only)
    # ------------------------------------------------------------------

    async def _run_memory(self, node: TaskNode) -> str:
        query = self._query_text(node)
        assert self._memory is not None  # guaranteed by availability check
        _enriched, memories = await self._memory.enrich_prompt(
            query, top_k=5, per_memory_chars=500, max_context_length=2000,
        )
        if not memories:
            return "No relevant local memories were found."
        lines = [f"- {(getattr(m, 'content', '') or '').strip()[:500]}" for m in memories]
        return "Relevant recalled information:\n" + "\n".join(lines)

    async def _run_internet(self, node: TaskNode, *, local_context_found: bool) -> str:
        query = self._query_text(node)
        assert self._internet is not None
        # Reuse the freshness gate so the internet stays a last resort.
        try:
            from knowledge.internet import needs_internet
            if not needs_internet(query, local_context_found=local_context_found):
                return "This step did not require live internet data."
        except Exception:  # noqa: BLE001 - gate optional
            pass
        context = await self._internet.build_context(query, max_results=5)
        return context or "No fresh public information was found for this step."

    async def _run_reasoning(self, node: TaskNode, context: str) -> str:
        prompt = node.description
        if context:
            prompt = f"{context}\n\nStep: {node.description}"
        # Prefer the KnowledgeEngine (carries its own memory-first context);
        # fall back to the raw guard. Both never raise / never return None.
        if self._knowledge is not None and self._knowledge.available:
            return await self._knowledge.answer(prompt)
        if self._guard is not None and self._guard.is_available:
            response = await self._guard.generate(
                prompt, system_prompt=self._reasoning_system_prompt,
            )
            return (response.content or "").strip()
        return ""

    # ------------------------------------------------------------------
    # Tool-backed capabilities (delegate to ToolExecutionEngine)
    # ------------------------------------------------------------------

    async def _run_tool(self, node: TaskNode, cap, start: float) -> NodeResult:
        assert self._tool_engine is not None
        # Single authoritative timeout: push it into the engine so its own
        # default (30s) can't mask the planner's per-task timeout.
        ctx = ToolContext(timeout_seconds=self._timeout)
        args = dict(node.args)
        # If the plan didn't specify the primary arg, derive it from the task
        # text. Some tools need a *structured* argument (e.g. the calculator
        # wants a bare arithmetic expression, not the sentence around it), so a
        # per-tool extractor cleans the text before it is passed through.
        if cap.default_arg and cap.default_arg not in args:
            value = self._extract_arg(cap.tool_name, cap.default_arg, node)
            if value:
                args[cap.default_arg] = value
        result = await self._tool_engine.execute(cap.tool_name, ctx, **args)
        if result.success:
            return self._result(node, TaskStatus.SUCCEEDED, f"tool:{cap.tool_name}",
                                 start, output=result.output or "")
        return self._result(node, TaskStatus.FAILED, f"tool:{cap.tool_name}",
                            start, error=result.error or "tool failed", output="")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_text(node: TaskNode) -> str:
        """The best text to feed a backend: explicit query arg, else description."""
        for key in ("query", "text", "expression", "input"):
            val = node.args.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return node.description.strip()

    def _extract_arg(self, tool_name: str, arg: str, node: TaskNode) -> str:
        """Derive a tool's primary argument from the task text.

        A small local model often names the ``calculator`` capability without
        supplying a clean ``expression`` — it passes the whole sentence. Rather
        than fail on a syntax error, pull the arithmetic sub-string out. Other
        tools receive the plain query text.
        """
        text = self._query_text(node)
        if tool_name == "calculator" and arg == "expression":
            expr = self._extract_expression(text)
            return expr or text
        return text

    @staticmethod
    def _extract_expression(text: str) -> str:
        """Return the arithmetic expression embedded in *text*, if any.

        Keeps only characters valid in a math expression (digits, operators,
        parentheses, decimal points) taken from the first run that contains a
        digit — e.g. "calculate 23*(18+7) then explain" → "23*(18+7)".
        """
        match = re.search(r"[0-9][0-9\s+\-*/().^%]*", text)
        if not match:
            return ""
        expr = match.group(0).strip().rstrip("?.,;!")
        # Require at least one operator, else it's just a number/noise.
        return expr if any(op in expr for op in "+-*/^%") else ""

    def _unavailable(self, node: TaskNode, backend: str, reason: str, start: float) -> NodeResult:
        _logger.info("Capability %r unavailable for node %s: %s",
                     node.required_tool, node.id, reason)
        # Deliberately generic + user-safe; the verifier/composer will keep the
        # real reason out of the final answer.
        message = f"The step '{node.description[:80]}' could not be completed right now."
        return self._result(node, TaskStatus.FAILED, backend, start,
                            error=f"{node.required_tool} unavailable: {reason}",
                            output=message)

    @staticmethod
    def _result(
        node: TaskNode,
        status: TaskStatus,
        backend: str,
        start: float,
        *,
        output: str = "",
        error: str | None = None,
    ) -> NodeResult:
        return NodeResult(
            node_id=node.id,
            status=status,
            output=output,
            error=error,
            backend=backend,
            attempts=1,
            duration_ms=(time.monotonic() - start) * 1000.0,
        )
