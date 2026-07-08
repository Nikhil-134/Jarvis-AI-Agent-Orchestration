"""ResponseVerifier — validates every final response before it reaches the user.

Requirement #5.  Offline, deterministic checks tuned for a small local model:

1. **Non-empty / min length** — reject empty or trivially short answers.
2. **No leaked exceptions / internal machinery** — reuse the shared
   ``core.response_guards`` marker set (same list the runtime composer uses).
3. **No raw artifacts** — tool-call JSON, tracebacks, leaked ``User:/Jarvis:``
   prompt scaffolding.
4. **Tool-result consistency** — a successful tool's concrete output should
   appear in the answer when the goal asked for it (grounding).
5. **No fabricated success** — if a critical task failed or the plan is
   incomplete, the answer must not claim it "successfully" did the thing.
6. **Hallucination heuristic (conditional)** — numbers/URLs in the answer that
   appear in neither the goal nor any task result are flagged, but *only* when
   at least one task produced structured tool output (avoids false positives on
   pure-reasoning answers).
7. **Confidence threshold** — combine plan + node confidence; below the
   threshold the answer is rejected.
8. **Graceful fallback** — a rejected answer is replaced with a safe message;
   the verifier never raises and never returns an unsafe string.
"""

from __future__ import annotations

import logging
import re

from core.response_guards import (
    has_leaked_scaffolding,
    is_user_safe,
    looks_like_tool_json,
)
from planning.interfaces import IResponseVerifier
from planning.models import NodeResult, Plan, TaskStatus, VerificationResult

_logger = logging.getLogger(__name__)

_GRACEFUL_FALLBACK = (
    "I wasn't able to put together a reliable answer for that just now. "
    "Could you rephrase it or try again in a moment?"
)

_PARTIAL_FALLBACK = (
    "I could only partly complete that. Here is what I was able to do:\n\n{done}"
)

_SUCCESS_CLAIM_RE = re.compile(
    r"\b(successfully|done|completed|finished|here (?:is|are) your)\b", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"\b\d[\d,]*\.?\d*\b")
_URL_RE = re.compile(r"https?://\S+")
_TRACEBACK_RE = re.compile(r"traceback \(most recent call last\)", re.IGNORECASE)


class ResponseVerifier(IResponseVerifier):
    """Validates a final response against a plan and its task results."""

    def __init__(self, *, confidence_threshold: float = 0.55) -> None:
        self._threshold = confidence_threshold

    def verify(
        self,
        response: str,
        plan: Plan,
        results: list[NodeResult],
    ) -> VerificationResult:
        """Return a :class:`VerificationResult`. Never raises."""
        try:
            return self._verify(response, plan, results)
        except Exception:  # noqa: BLE001 - verifier must never raise
            _logger.exception("Verifier raised; returning graceful fallback")
            return VerificationResult(
                ok=False, response=_GRACEFUL_FALLBACK, confidence=0.0,
                issues=("verifier_error",),
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _verify(
        self, response: str, plan: Plan, results: list[NodeResult],
    ) -> VerificationResult:
        text = (response or "").strip()
        issues: list[str] = []

        # 1. Non-empty / minimum length.
        if len(text) < 3:
            return self._reject(["empty"], _GRACEFUL_FALLBACK, plan, results)

        # 2. No internal machinery markers (shared list).
        if not is_user_safe(text):
            issues.append("internal_marker")

        # 3. No raw artifacts.
        if looks_like_tool_json(text):
            issues.append("tool_json")
        if _TRACEBACK_RE.search(text):
            issues.append("traceback")
        if has_leaked_scaffolding(text):
            issues.append("scaffolding")

        # Any of the above make the answer fundamentally unsafe → fallback.
        if issues:
            return self._reject(issues, self._salvage(plan, results), plan, results)

        # Determine whether any node produced *structured tool output*.
        tool_results = [
            r for r in results
            if r.success and r.backend.startswith("tool:") and r.output
        ]
        has_tool_output = bool(tool_results)

        # 5. No fabricated success when a critical node failed / plan incomplete.
        critical_failed = any(
            r.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED)
            for r in results
        )
        if critical_failed and _SUCCESS_CLAIM_RE.search(text):
            issues.append("fabricated_success")
            return self._reject(issues, self._salvage(plan, results), plan, results)

        # 4. Tool-result grounding — the concrete output should appear.
        if has_tool_output:
            grounded = any(self._output_present(r.output, text) for r in tool_results)
            if not grounded:
                issues.append("ungrounded_tool_result")

        # 6. Numeric / URL hallucination (only when tool output exists).
        if has_tool_output:
            corpus = plan.goal + "\n" + "\n".join(r.output for r in results if r.output)
            if self._has_unsupported_token(text, corpus):
                issues.append("possible_hallucination")

        # 7. Echo / refusal detection.
        if self._is_echo(text, plan.goal):
            issues.append("echo")
            return self._reject(issues, self._salvage(plan, results), plan, results)

        # 8. Confidence gate.
        confidence = self._confidence(plan, results)
        if confidence < self._threshold:
            issues.append("low_confidence")

        # "Soft" issues (grounding/hallucination/low-confidence) don't force a
        # fallback on their own unless they stack up — a small model often
        # paraphrases. Reject only when the answer is empty/unsafe (handled
        # above) or confidence is below threshold AND something else is wrong.
        hard_reject = ("low_confidence" in issues and len(issues) >= 2)
        if hard_reject:
            return self._reject(issues, self._salvage(plan, results), plan, results)

        return VerificationResult(
            ok=True, response=text, confidence=confidence, issues=tuple(issues),
        )

    # ------------------------------------------------------------------
    # Scoring + salvage
    # ------------------------------------------------------------------

    def _confidence(self, plan: Plan, results: list[NodeResult]) -> float:
        base = plan.overall_confidence if plan.overall_confidence > 0 else 0.5
        succeeded = [r for r in results if r.success]
        if not results:
            return base
        # Penalise by the fraction of nodes that did not succeed.
        success_ratio = len(succeeded) / len(results)
        node_conf = 1.0
        crit_nodes = [n for n in plan.nodes if n.critical]
        if crit_nodes:
            # Confidence of critical nodes that succeeded.
            done_ids = {r.node_id for r in succeeded}
            crit_conf = [n.confidence for n in crit_nodes if n.id in done_ids]
            node_conf = min(crit_conf) if crit_conf else 0.4
        return round(base * success_ratio * node_conf, 4)

    def _salvage(self, plan: Plan, results: list[NodeResult]) -> str:
        """Build an honest partial answer from whatever succeeded."""
        good = [r.output.strip() for r in results if r.success and r.output.strip()]
        # Filter out any internal-ish outputs before showing them.
        good = [g for g in good if is_user_safe(g)]
        if not good:
            return _GRACEFUL_FALLBACK
        done = "\n\n".join(good)
        return _PARTIAL_FALLBACK.format(done=done)

    def _reject(
        self, issues: list[str], replacement: str, plan: Plan,
        results: list[NodeResult],
    ) -> VerificationResult:
        _logger.info("Response rejected by verifier: %s", issues)
        # Ensure the replacement itself is safe.
        safe = replacement if is_user_safe(replacement) else _GRACEFUL_FALLBACK
        return VerificationResult(
            ok=False, response=safe,
            confidence=self._confidence(plan, results), issues=tuple(issues),
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _output_present(output: str, text: str) -> bool:
        """Whether a tool's output is reflected in the answer (loose match)."""
        out = output.strip()
        if not out:
            return True
        low_text = text.lower()
        if out.lower() in low_text:
            return True
        # For numeric tool outputs, match any salient number.
        nums = _NUMBER_RE.findall(out)
        return any(n in text for n in nums)

    @staticmethod
    def _has_unsupported_token(text: str, corpus: str) -> bool:
        """Flag a number or URL in *text* absent from *corpus*."""
        for url in _URL_RE.findall(text):
            if url not in corpus:
                return True
        text_nums = set(_NUMBER_RE.findall(text))
        corpus_nums = set(_NUMBER_RE.findall(corpus))
        # Ignore small integers (0-31) — dates/counts a model reasonably emits.
        for n in text_nums - corpus_nums:
            try:
                if abs(float(n.replace(",", ""))) > 31:
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _is_echo(text: str, goal: str) -> bool:
        t = text.strip().lower()
        g = goal.strip().lower()
        if not g:
            return False
        return t == g or (len(t) < len(g) + 5 and t in g)
