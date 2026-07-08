"""Calculator tool — evaluates mathematical expressions using AST-based safety."""

from __future__ import annotations

from typing import Any

from tools.expression_safety import ExpressionSafety
from tools.interfaces import ITool, PermissionLevel, ToolSpec


class CalculatorTool(ITool):
    """Evaluates a mathematical expression and returns the result.

    Supports +, -, *, /, //, %, **, ^, parentheses, plus:

    - ``sqrt(x)``, ``pow(x, y)``, ``exp(x)``, ``hypot(a, b)``
    - ``sin(x)``, ``cos(x)``, ``tan(x)``, ``asin(x)``, ``acos(x)``, ``atan(x)``, ``atan2(y, x)``
    - ``radians(x)``, ``degrees(x)``
    - ``log(x)``, ``log2(x)``, ``log10(x)``, ``ln(x)``
    - ``abs(x)``, ``ceil(x)``, ``floor(x)``, ``round(x)``
    - ``factorial(x)``
    - ``gcd(a, b)``, ``lcm(a, b)``
    - Constants ``pi`` and ``e``, large integers, and scientific notation

    Integral results (``factorial(50)``, ``2**500``) stay exact. All
    calculation is deterministic (AST evaluation) — the LLM never does maths.
    """

    def __init__(self) -> None:
        self._safety = ExpressionSafety()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description=(
                "Evaluate a mathematical expression. "
                "Supports +, -, *, /, //, %, **, ^, parentheses, "
                "sqrt, pow, exp, hypot, sin, cos, tan, asin, acos, atan, atan2, "
                "radians, degrees, log, log2, log10, ln, "
                "abs, ceil, floor, round, factorial, gcd, lcm, "
                "constants pi, e, large integers, and scientific notation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate.",
                    },
                },
                "required": ["expression"],
            },
        )

    @property
    def category(self) -> str:
        return "utility"

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.SAFE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        expression: str = str(kwargs.get("expression", ""))
        if not expression:
            return {"success": False, "output": "", "error": "No expression provided."}

        ok, result = self._safety.evaluate(expression)
        if ok:
            # Format large integers without scientific notation
            if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
                formatted = str(int(result))
            elif isinstance(result, float) and abs(result) > 1e10:
                formatted = f"{result:.6e}"
            else:
                formatted = str(result)
            return {"success": True, "output": formatted}
        return {"success": False, "output": "", "error": str(result)}
