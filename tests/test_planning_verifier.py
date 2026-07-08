"""Tests for planning.verifier.ResponseVerifier — safety + grounding rules."""

from __future__ import annotations

from planning.models import NodeResult, Plan, TaskNode, TaskStatus
from planning.verifier import ResponseVerifier


def _plan(conf: float = 0.9, critical: bool = True) -> Plan:
    node = TaskNode(id="s1", description="do", confidence=conf, critical=critical)
    return Plan(goal="do the thing", nodes=(node,), overall_confidence=conf)


def _ok_result(output: str, backend: str = "reasoning") -> NodeResult:
    return NodeResult(node_id="s1", status=TaskStatus.SUCCEEDED, output=output,
                      backend=backend)


class TestBasicSafety:
    def test_empty_rejected(self) -> None:
        v = ResponseVerifier()
        r = v.verify("", _plan(), [_ok_result("x")])
        assert not r.ok and r.response

    def test_internal_marker_rejected(self) -> None:
        v = ResponseVerifier()
        r = v.verify("Error: memory_id is required for this", _plan(), [_ok_result("x")])
        assert not r.ok
        assert "internal_marker" in r.issues

    def test_tool_json_rejected(self) -> None:
        v = ResponseVerifier()
        r = v.verify('{"name": "calculator", "arguments": {}}', _plan(), [_ok_result("x")])
        assert not r.ok
        assert "tool_json" in r.issues

    def test_traceback_rejected(self) -> None:
        v = ResponseVerifier()
        r = v.verify("Traceback (most recent call last): boom", _plan(), [_ok_result("x")])
        assert not r.ok

    def test_scaffolding_rejected(self) -> None:
        v = ResponseVerifier()
        r = v.verify("User: hi\nJarvis: hello", _plan(), [_ok_result("x")])
        assert not r.ok
        assert "scaffolding" in r.issues


class TestGrounding:
    def test_tool_number_present_ok(self) -> None:
        v = ResponseVerifier()
        results = [_ok_result("575", backend="tool:calculator")]
        r = v.verify("The answer is 575.", _plan(), results)
        assert r.ok

    def test_tool_number_absent_flagged(self) -> None:
        v = ResponseVerifier()
        results = [_ok_result("575", backend="tool:calculator")]
        r = v.verify("The answer is something else entirely.", _plan(), results)
        assert "ungrounded_tool_result" in r.issues

    def test_pure_reasoning_number_not_flagged(self) -> None:
        # No tool output → numbers in a reasoned answer aren't hallucination-flagged.
        v = ResponseVerifier()
        r = v.verify("Recursion has about 1000 stack frames typically.",
                     _plan(), [_ok_result("some prose", backend="reasoning")])
        assert "possible_hallucination" not in r.issues

    def test_unsupported_number_with_tool_output_flagged(self) -> None:
        v = ResponseVerifier()
        results = [_ok_result("575", backend="tool:calculator")]
        r = v.verify("The answer is 575 but also 999999 apples.", _plan(), results)
        assert "possible_hallucination" in r.issues


class TestFabricatedSuccess:
    def test_success_claim_with_failed_node_rejected(self) -> None:
        v = ResponseVerifier()
        failed = NodeResult(node_id="s1", status=TaskStatus.FAILED, error="x")
        r = v.verify("Successfully completed everything!", _plan(), [failed])
        assert not r.ok
        assert "fabricated_success" in r.issues


class TestConfidenceAndEcho:
    def test_low_confidence_with_issue_rejected(self) -> None:
        v = ResponseVerifier(confidence_threshold=0.9)
        # tool output present but ungrounded → 2 issues incl low_confidence.
        results = [_ok_result("12345", backend="tool:calculator")]
        r = v.verify("An unrelated answer.", _plan(conf=0.3), results)
        assert not r.ok

    def test_echo_rejected(self) -> None:
        v = ResponseVerifier()
        r = v.verify("do the thing", _plan(), [_ok_result("do the thing")])
        assert not r.ok
        assert "echo" in r.issues

    def test_happy_path_ok(self) -> None:
        v = ResponseVerifier()
        r = v.verify("Here's a clear, helpful explanation of the topic.",
                     _plan(conf=0.9), [_ok_result("prose", backend="reasoning")])
        assert r.ok
        assert r.confidence > 0


class TestSalvage:
    def test_salvage_uses_successful_output(self) -> None:
        v = ResponseVerifier()
        results = [
            _ok_result("Useful partial result about X.", backend="reasoning"),
            NodeResult(node_id="s2", status=TaskStatus.FAILED, error="y"),
        ]
        # Force a hard reject via internal marker, check salvage surfaces good text.
        r = v.verify("nonetype error leaked", _plan(), results)
        assert not r.ok
        assert "Useful partial result" in r.response

    def test_never_raises(self) -> None:
        v = ResponseVerifier()
        # Pass odd inputs; must return a result, not raise.
        r = v.verify(None, _plan(), [])  # type: ignore[arg-type]
        assert r.response
