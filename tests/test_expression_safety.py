"""Tests for ExpressionSafety — AST-based math validation and evaluation."""

from __future__ import annotations

import pytest

from tools.expression_safety import ExpressionSafety


# =========================================================================
# is_pure_expression — validation only
# =========================================================================

class TestIsPureExpression:
    def test_basic_arithmetic(self) -> None:
        assert ExpressionSafety.is_pure_expression("2+2")
        assert ExpressionSafety.is_pure_expression("234+56")
        assert ExpressionSafety.is_pure_expression("(234*89)+100")
        assert ExpressionSafety.is_pure_expression("2**100")
        assert ExpressionSafety.is_pure_expression("2^100")  # ^ converted to **
        assert ExpressionSafety.is_pure_expression("1e9*1e9")

    def test_math_functions(self) -> None:
        assert ExpressionSafety.is_pure_expression("sqrt(81)")
        assert ExpressionSafety.is_pure_expression("sin(45)")
        assert ExpressionSafety.is_pure_expression("cos(0)")
        assert ExpressionSafety.is_pure_expression("tan(0)")
        assert ExpressionSafety.is_pure_expression("asin(1)")
        assert ExpressionSafety.is_pure_expression("acos(0)")
        assert ExpressionSafety.is_pure_expression("atan(1)")
        assert ExpressionSafety.is_pure_expression("radians(180)")
        assert ExpressionSafety.is_pure_expression("degrees(3.14159)")
        assert ExpressionSafety.is_pure_expression("log(100)")
        assert ExpressionSafety.is_pure_expression("log10(100000)")
        assert ExpressionSafety.is_pure_expression("ln(10)")
        assert ExpressionSafety.is_pure_expression("abs(-5)")
        assert ExpressionSafety.is_pure_expression("ceil(4.2)")
        assert ExpressionSafety.is_pure_expression("floor(4.9)")
        assert ExpressionSafety.is_pure_expression("round(3.7)")
        assert ExpressionSafety.is_pure_expression("factorial(20)")

    def test_multi_arg_functions(self) -> None:
        assert ExpressionSafety.is_pure_expression("gcd(12, 8)")
        assert ExpressionSafety.is_pure_expression("lcm(4, 6)")

    def test_constants(self) -> None:
        assert ExpressionSafety.is_pure_expression("pi")
        assert ExpressionSafety.is_pure_expression("e")
        assert ExpressionSafety.is_pure_expression("pi * 2")
        assert ExpressionSafety.is_pure_expression("e ** 2")

    def test_complex_expressions(self) -> None:
        assert ExpressionSafety.is_pure_expression("(987654321*123456789)+(999999999/37)")
        assert ExpressionSafety.is_pure_expression("sqrt(987654321)")
        assert ExpressionSafety.is_pure_expression("factorial(20)")
        assert ExpressionSafety.is_pure_expression("2**100")
        assert ExpressionSafety.is_pure_expression("sin(45) + cos(45)")
        assert ExpressionSafety.is_pure_expression("sqrt(abs(-100))")

    def test_unsafe_expressions_rejected(self) -> None:
        assert not ExpressionSafety.is_pure_expression("import os")
        assert not ExpressionSafety.is_pure_expression("__import__('os')")
        assert not ExpressionSafety.is_pure_expression("globals()")
        assert not ExpressionSafety.is_pure_expression("lambda x: x")
        assert not ExpressionSafety.is_pure_expression("open('/etc/passwd')")
        assert not ExpressionSafety.is_pure_expression("eval('2+2')")
        assert not ExpressionSafety.is_pure_expression("exec('print(1)')")

    def test_empty_and_invalid(self) -> None:
        assert not ExpressionSafety.is_pure_expression("")
        assert not ExpressionSafety.is_pure_expression("hello world")
        assert not ExpressionSafety.is_pure_expression("what is 2+2")

    def test_explanation_rejected(self) -> None:
        assert not ExpressionSafety.is_pure_expression("explain why 2+2=4")


# =========================================================================
# evaluate — full evaluation
# =========================================================================

class TestEvaluate:
    def test_basic_arithmetic(self) -> None:
        ok, result = ExpressionSafety.evaluate("2+2")
        assert ok
        assert result == 4.0

        ok, result = ExpressionSafety.evaluate("234+56")
        assert ok
        assert result == 290.0

        ok, result = ExpressionSafety.evaluate("(234*89)+100")
        assert ok
        assert result == 20926.0

    def test_math_functions(self) -> None:
        ok, result = ExpressionSafety.evaluate("sqrt(81)")
        assert ok
        assert result == 9.0

        ok, result = ExpressionSafety.evaluate("sin(0)")
        assert ok
        assert float(f"{result:.10f}") == 0.0

        ok, result = ExpressionSafety.evaluate("cos(0)")
        assert ok
        assert result == 1.0

        ok, result = ExpressionSafety.evaluate("abs(-5)")
        assert ok
        assert result == 5.0

        ok, result = ExpressionSafety.evaluate("ceil(4.2)")
        assert ok
        assert result == 5.0

        ok, result = ExpressionSafety.evaluate("floor(4.9)")
        assert ok
        assert result == 4.0

        ok, result = ExpressionSafety.evaluate("round(3.7)")
        assert ok
        assert result == 4.0

    def test_factorial(self) -> None:
        ok, result = ExpressionSafety.evaluate("factorial(20)")
        assert ok
        assert float(f"{result:.0f}") == 2432902008176640000.0

        ok, result = ExpressionSafety.evaluate("factorial(5)")
        assert ok
        assert result == 120.0

    def test_power(self) -> None:
        ok, result = ExpressionSafety.evaluate("2**100")
        assert ok
        assert float(f"{result:.0f}") == 1267650600228229401496703205376.0

        ok, result = ExpressionSafety.evaluate("2^10")
        assert ok
        assert result == 1024.0

    def test_log(self) -> None:
        ok, result = ExpressionSafety.evaluate("log10(100000)")
        assert ok
        assert result == 5.0

        ok, result = ExpressionSafety.evaluate("log(100)")
        assert ok
        assert float(f"{result:.6f}") == 4.605170

        ok, result = ExpressionSafety.evaluate("ln(10)")
        assert ok
        assert float(f"{result:.6f}") == 2.302585

    def test_gcd_lcm(self) -> None:
        ok, result = ExpressionSafety.evaluate("gcd(12, 8)")
        assert ok
        assert result == 4.0

        ok, result = ExpressionSafety.evaluate("lcm(4, 6)")
        assert ok
        assert result == 12.0

    def test_constants(self) -> None:
        ok, result = ExpressionSafety.evaluate("pi")
        assert ok
        assert float(f"{result:.6f}") == 3.141593

        ok, result = ExpressionSafety.evaluate("e")
        assert ok
        assert float(f"{result:.6f}") == 2.718282

        ok, result = ExpressionSafety.evaluate("pi * 2")
        assert ok
        assert float(f"{result:.6f}") == 6.283185

    def test_scientific_notation(self) -> None:
        ok, result = ExpressionSafety.evaluate("1e9*1e9")
        assert ok
        assert float(f"{result:.0e}") == 1e18

    def test_multi_arg_functions(self) -> None:
        ok, result = ExpressionSafety.evaluate("gcd(12, 8)")
        assert ok
        assert result == 4.0

        ok, result = ExpressionSafety.evaluate("lcm(4, 6, 12)")
        assert ok
        assert result == 12.0

    def test_unsafe_rejected(self) -> None:
        ok, _ = ExpressionSafety.evaluate("import os")
        assert not ok

        ok, _ = ExpressionSafety.evaluate("__import__('os')")
        assert not ok

        ok, _ = ExpressionSafety.evaluate("open('/etc/passwd')")
        assert not ok

    def test_empty(self) -> None:
        ok, msg = ExpressionSafety.evaluate("")
        assert not ok
        assert "Empty" in msg

    def test_syntax_error(self) -> None:
        ok, msg = ExpressionSafety.evaluate("2++")
        assert not ok
        assert "Syntax error" in msg

    def test_unsupported_function(self) -> None:
        ok, _ = ExpressionSafety.evaluate("nonexistent_func(42)")
        assert not ok

    def test_string_constant_rejected(self) -> None:
        ok, _ = ExpressionSafety.evaluate("'hello'")
        assert not ok

    def test_list_rejected(self) -> None:
        ok, _ = ExpressionSafety.evaluate("[1,2,3]")
        assert not ok

    def test_dict_rejected(self) -> None:
        ok, _ = ExpressionSafety.evaluate("{'a': 1}")
        assert not ok


# =========================================================================
# Cycle 9 — pow() and companions, exact large integers, clean errors, guards
# =========================================================================

class TestPowAndCompanions:
    """pow() was a required function that was missing (leaked maths to the LLM)."""

    def test_pow_is_supported(self) -> None:
        assert ExpressionSafety.is_pure_expression("pow(2, 10)")
        ok, result = ExpressionSafety.evaluate("pow(2, 10)")
        assert ok
        assert result == 1024

    def test_pow_fractional_exponent(self) -> None:
        ok, result = ExpressionSafety.evaluate("pow(2, 0.5)")
        assert ok
        assert abs(result - 1.4142135623730951) < 1e-9

    def test_pow_three_arg_modular(self) -> None:
        ok, result = ExpressionSafety.evaluate("pow(2, 10, 1000)")
        assert ok
        assert result == 24  # 1024 % 1000

    @pytest.mark.parametrize("expr,expected", [
        ("exp(0)", 1.0),
        ("log2(8)", 3.0),
        ("hypot(3, 4)", 5.0),
    ])
    def test_new_companion_functions(self, expr: str, expected: float) -> None:
        ok, result = ExpressionSafety.evaluate(expr)
        assert ok
        assert abs(result - expected) < 1e-9

    def test_atan2(self) -> None:
        ok, result = ExpressionSafety.evaluate("atan2(1, 1)")
        assert ok
        assert abs(result - 0.7853981633974483) < 1e-9


class TestExactLargeIntegers:
    """Integral results stay exact — a blanket float() cast lost precision."""

    def test_big_power_is_exact_not_overflow(self) -> None:
        ok, result = ExpressionSafety.evaluate("2**5000")
        assert ok
        assert isinstance(result, int)
        assert result == 2 ** 5000  # exact, no "int too large to convert to float"

    def test_big_factorial_is_exact(self) -> None:
        import math
        ok, result = ExpressionSafety.evaluate("factorial(100)")
        assert ok
        assert isinstance(result, int)
        assert result == math.factorial(100)

    def test_small_integral_result_is_int(self) -> None:
        ok, result = ExpressionSafety.evaluate("2+2")
        assert ok
        assert result == 4  # 4 == 4.0 so backward-compatible

    def test_non_integral_stays_float(self) -> None:
        ok, result = ExpressionSafety.evaluate("10/3")
        assert ok
        assert isinstance(result, float)


class TestCleanErrorMessages:
    """Raw library exceptions must never reach the user."""

    @pytest.mark.parametrize("expr", ["sqrt(-1)", "log(0)", "factorial(-1)", "factorial(1.5)"])
    def test_domain_errors_are_clean(self, expr: str) -> None:
        ok, msg = ExpressionSafety.evaluate(expr)
        assert not ok
        # No leaked library phrasing
        assert "domain" not in msg.lower()
        assert "nonnegative" not in msg.lower()
        assert "traceback" not in msg.lower()
        assert "isn't defined" in msg.lower() or "not defined" in msg.lower()

    def test_division_by_zero_is_clean(self) -> None:
        ok, msg = ExpressionSafety.evaluate("1/0")
        assert not ok
        assert "zero" in msg.lower()
        assert "ZeroDivisionError" not in msg

    def test_unsupported_construct_is_clean(self) -> None:
        ok, msg = ExpressionSafety.evaluate("__import__('os')")
        assert not ok
        assert "can't calculate" in msg.lower() or "cannot" in msg.lower()


class TestReliabilityGuards:
    """Pathological inputs are refused cleanly instead of hanging the process."""

    def test_absurd_exponent_refused(self) -> None:
        ok, msg = ExpressionSafety.evaluate("2**999999999")
        assert not ok
        assert "too large" in msg.lower()

    def test_absurd_factorial_refused(self) -> None:
        ok, msg = ExpressionSafety.evaluate("factorial(999999)")
        assert not ok
        assert "too large" in msg.lower()

    def test_legitimate_large_still_allowed(self) -> None:
        # Well within the guard — must still compute exactly.
        ok, result = ExpressionSafety.evaluate("2**100")
        assert ok
        assert result == 2 ** 100


# =========================================================================
# Performance — ensure < 100 ms latency
# =========================================================================

class TestPerformance:
    def test_evaluate_latency(self) -> None:
        import time
        expression = "(987654321*123456789)+(999999999/37)"
        start = time.perf_counter()
        for _ in range(100):
            ExpressionSafety.evaluate(expression)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"Average evaluation took {avg_ms:.2f} ms"

    def test_validation_latency(self) -> None:
        import time
        start = time.perf_counter()
        for _ in range(100):
            ExpressionSafety.is_pure_expression("sqrt(987654321)")
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"Average validation took {avg_ms:.2f} ms"
