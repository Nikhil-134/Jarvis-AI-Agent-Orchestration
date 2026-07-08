"""Expression safety — AST-based validation and evaluation of mathematical expressions.

Uses Python's ``ast`` module to parse and validate expressions, rejecting
any unsafe AST nodes.  Only a whitelist of math functions, operators,
and constants is allowed.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Reliability guards
# ---------------------------------------------------------------------------
# The engine must support "large integers" and factorial, but a naive
# ``2**10**9`` or ``factorial(10**9)`` would hang the process / exhaust memory
# (a real reliability failure). These caps are far above any legitimate
# calculator use — ``2**100000`` is ~30 100 digits, ``factorial(10000)`` is
# ~35 660 digits, both computed in well under a second — yet they refuse the
# pathological inputs with a clean, honest error instead of freezing.
_MAX_POWER_EXPONENT = 100_000
_MAX_FACTORIAL_INPUT = 10_000


def _safe_pow(base: Any, exp: Any, mod: Any = None) -> Any:
    """``pow`` mirroring built-in semantics, with a size guard on 2-arg powers.

    The 3-arg (modular) form is inherently bounded by the modulus, so it is
    passed straight through. Exact large integers are preserved (built-in
    ``pow``), unlike ``math.pow`` which would overflow to float.
    """
    if mod is not None:
        return pow(base, exp, mod)
    try:
        oversized = abs(exp) > _MAX_POWER_EXPONENT
    except TypeError:  # non-numeric exponent — let pow raise a clean error
        oversized = False
    if oversized:
        raise OverflowError("exponent too large")
    return pow(base, exp)


def _safe_factorial(n: Any) -> int:
    """``factorial`` with a domain + size guard (clean errors, never hangs)."""
    if isinstance(n, float):
        if not n.is_integer():
            raise ValueError("factorial requires a whole number")
        n = int(n)
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("factorial requires a whole number")
    if n < 0:
        raise ValueError("factorial is not defined for negative numbers")
    if n > _MAX_FACTORIAL_INPUT:
        raise OverflowError("factorial input too large")
    return math.factorial(n)


# ---------------------------------------------------------------------------
# Whitelist of supported operators
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: _safe_pow,  # guarded (size cap) — mirrors built-in ** semantics
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# ---------------------------------------------------------------------------
# Whitelist of supported math functions
# ---------------------------------------------------------------------------

_MATH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    # Built-in pow (not math.pow) preserves exact large-integer results
    # (pow(2, 5000)) and returns int for integral results — math.pow would
    # coerce to float and overflow. Guarded against pathological exponents.
    "pow": _safe_pow,
    "exp": math.exp,
    "hypot": math.hypot,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "radians": math.radians,
    "degrees": math.degrees,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "ln": math.log,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "factorial": _safe_factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
}

_MATH_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

# Pattern to match standalone function calls at the start of expressions.
# Used to detect pure-math queries like "sqrt(81)" or "sin(45)".
_FUNC_CALL_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s*\(")


class ExpressionSafety:
    """Validates and evaluates mathematical expressions using the AST.

    Usage::

        es = ExpressionSafety()
        ok, result = es.evaluate("2 + 2")
        ok, result = es.evaluate("sqrt(81)")
        ok, result = es.evaluate("import os")  # False — unsafe
    """

    @staticmethod
    def is_pure_expression(text: str) -> bool:
        """Return ``True`` if *text* looks like a pure math expression.

        Checks that the text consists only of math tokens — digits,
        operators, function names, parentheses, constants.
        """
        stripped = text.strip()
        if not stripped:
            return False
        # Normalise ^ to ** before checking
        normalised = _normalise_expression(stripped)
        try:
            tree = ast.parse(normalised, mode="eval")
        except SyntaxError:
            return False
        return _validate_node(tree.body)

    @staticmethod
    def evaluate(text: str) -> tuple[bool, float | int | str]:
        """Evaluate *text* as a mathematical expression.

        Returns ``(True, result)`` on success or ``(False, error_message)``
        on failure (invalid expression, unsafe construct, etc.).

        Integral results are returned as ``int`` (so ``factorial(100)`` and
        ``2**5000`` stay exact — a blanket ``float()`` cast used to lose
        precision on the former and raise "int too large to convert to float"
        on the latter). Non-integral results are returned as ``float``.
        Error messages are mapped to clean, user-facing text — a raw library
        exception (``math domain error``, ``division by zero``) must never
        reach the user (see PROJECT_BRAIN response-quality rule).
        """
        stripped = text.strip()
        if not stripped:
            return False, "Empty expression."

        normalised = _normalise_expression(stripped)
        try:
            tree = ast.parse(normalised, mode="eval")
        except SyntaxError:
            # Clean, user-facing text — the raw SyntaxError detail is never
            # surfaced. "Syntax error" is kept as a recognisable, safe phrase.
            return False, "Syntax error — that expression isn't valid math. Please check it and try again."

        if not _validate_node(tree.body):
            return False, "That expression uses something I can't calculate. Try numbers, operators, or supported functions."

        try:
            result = _eval_node(tree.body)
        except Exception as exc:  # noqa: BLE001 - mapped to clean text below
            return False, _friendly_error(exc)

        return True, _coerce_number(result)


def _normalise_expression(text: str) -> str:
    """Pre-process *text* for AST parsing.

    - Replace ``^`` with ``**`` (power operator)
    - Replace named constants with their values
    """
    result = text.replace("^", "**")

    # Replace constant names with their float values.
    # We use word-boundary matching to avoid partial replacements.
    for name, value in _MATH_CONSTANTS.items():
        result = re.sub(rf"\b{re.escape(name)}\b", str(value), result)

    return result


def _validate_node(node: ast.AST) -> bool:
    """Recursively check that *node* (and its children) are all whitelisted."""
    if isinstance(node, ast.Expression):
        return _validate_node(node.body)

    # Constants (numbers)
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))

    # Binary operators: + - * / // % **
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY_OPS:
            return False
        return _validate_node(node.left) and _validate_node(node.right)

    # Unary operators: + -
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            return False
        return _validate_node(node.operand)

    # Function calls: sqrt(81), sin(45), etc.
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return False
        if node.func.id not in _MATH_FUNCTIONS:
            return False
        return all(_validate_node(arg) for arg in node.args) and all(
            _validate_node(kw.value) for kw in node.keywords
        )

    return False


def _eval_node(node: ast.AST) -> int | float:
    """Evaluate a validated AST node and return the numeric result.

    Returns native ``int`` or ``float`` — the caller (``evaluate``)
    converts to ``float`` for the final result.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        return node.value  # keep as int or float

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op = _BINARY_OPS[type(node.op)]
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS[type(node.op)]
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        func = _MATH_FUNCTIONS[node.func.id]
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)

    raise ValueError(f"Unsupported node type: {type(node).__name__}")


def _coerce_number(result: Any) -> int | float:
    """Return *result* as ``int`` when it is integral, else ``float``.

    Preserving ``int`` keeps arbitrarily large integers exact (``factorial(50)``,
    ``2**5000``) — a blanket ``float()`` cast lost precision on big values and
    raised "int too large to convert to float" on very big ones. ``bool`` is
    normalised to ``int`` defensively (no boolean-valued ops are whitelisted).
    """
    if isinstance(result, bool):
        return int(result)
    if isinstance(result, int):
        return result
    return float(result)


def _friendly_error(exc: Exception) -> str:
    """Map an evaluation exception to clean, user-facing text.

    A raw library message (``math domain error``, ``division by zero``,
    ``int too large to convert to float``) must never reach the user — the
    project's response-quality rule forbids leaking internal exceptions.
    """
    if isinstance(exc, ZeroDivisionError):
        return "That doesn't work — you can't divide by zero."
    if isinstance(exc, OverflowError):
        return "That number is too large for me to compute."
    if isinstance(exc, ValueError):
        return (
            "That maths operation isn't defined for those inputs — for example, "
            "the square root or log of a negative number, or a factorial of a "
            "negative or fractional number."
        )
    if isinstance(exc, (TypeError, RecursionError)):
        return "That expression isn't a valid calculation."
    return "I couldn't evaluate that expression."
