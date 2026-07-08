"""Intent detection & classification — matches natural language queries to
intent categories (greeting, conversation, tool) and tool intents.

Allows the orchestrator to route tool-able requests directly to the
ToolAgent without involving the LLM, and to route simple greetings
and chit-chat without any specialist agents.
"""

from __future__ import annotations

import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)

from tools.expression_safety import ExpressionSafety
from tools.manager import ToolManager


# Confidence threshold: intents below this are treated as ambiguous and
# routed to the LLM directly without invoking specialist agents.
CONFIDENCE_THRESHOLD = 0.70


class IntentClassification:
    """Result of intent classification with label and confidence score.

    Supports equality comparison against strings for backward compatibility::

        result = detector.classify("hello")
        result == "greeting"          # True
        result.label                  # "greeting"
        result.confidence             # 0.99
    """

    __slots__ = ("_label", "_confidence")

    def __init__(self, label: str, confidence: float) -> None:
        self._label = label
        self._confidence = confidence

    @property
    def label(self) -> str:
        return self._label

    @property
    def confidence(self) -> float:
        return self._confidence

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._label == other
        if isinstance(other, IntentClassification):
            return self._label == other._label
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __repr__(self) -> str:
        return f"IntentClassification(label={self._label!r}, confidence={self._confidence})"

    def __hash__(self) -> int:
        return hash(self._label)


def _match(
    query: str,
    pattern: str,
    *,
    search: bool = False,
) -> re.Match[str] | None:
    """Match *query* against *pattern* with ``re.IGNORECASE``.

    Uses ``re.match`` by default or ``re.search`` when *search* is ``True``.
    The returned match object refers to the *original* query (not lowercased)
    so capture groups preserve original case.
    """
    flags = re.IGNORECASE
    if search:
        return re.search(pattern, query, flags)
    return re.match(pattern, query, flags)


# Prefixes we strip from queries to extract a math expression.
# E.g. "what is 2+2" -> "2+2"
_MATH_PREFIX_PATTERNS: list[str] = [
    r"^(?:what\s+)?(?:is\s+|are\s+|does\s+)?(?:the\s+)?(?:value\s+)?(?:of\s+)?(?:result\s+)?(?:of\s+)?",
    r"^calculate\s+",
    r"^solve\s+",
    r"^compute\s+",
    r"^evaluate\s+",
    r"^find\s+",
    r"^determine\s+",
    r"^work\s+out\s+",
    r"^figure\s+out\s+",
]

# Phrases that indicate the user wants an explanation, not just a calculation.
_EXPLANATION_PHRASES: set[str] = {
    "explain", "why", "how does", "how do", "how is",
    "what is the reason", "what does", "describe",
    "tell me about", "how come", "why does", "why is",
    "show me how", "walk me through",
}

_EXPRESSION_SAFETY = ExpressionSafety()


# Math function names as a regex fragment so we can build patterns
# that match things like "sqrt(81)" or "sin(45)".
# MUST stay in sync with ExpressionSafety._MATH_FUNCTIONS — a name that is
# evaluable but missing here is not recognised as a pure-math expression, so
# it leaks to the LLM (which must NEVER do maths). Longer names first so the
# alternation matches e.g. "log10"/"log2" before "log", "atan2" before "atan".
_MATH_FUNC_NAMES = (
    "sqrt|log10|log2|atan2|asin|acos|atan|sin|cos|tan|"
    "radians|degrees|log|ln|exp|pow|hypot|"
    "abs|ceil|floor|round|factorial|gcd|lcm"
)
_MATH_IDENTIFIER = rf"(?:{_MATH_FUNC_NAMES}|pi|e|[a-zA-Z_]\w*)"
_MATH_NUMBER = r"\d+(?:\.\d+)?(?:e[+-]?\d+)?"
_MATH_TOKEN = rf"(?:{_MATH_IDENTIFIER}|{_MATH_NUMBER}|[+\-*/%^()\[\],])"

# Pure math expression: starts with a math token and contains only math tokens.
_PURE_MATH_RE = re.compile(
    rf"^\s*{_MATH_TOKEN}\s*({_MATH_TOKEN}\s*)*$",
    re.IGNORECASE,
)


_UUID_PATTERNS: list[str] = [
    r"\b(generate|create|new)\s+(a\s+|an\s+)?(uuid|guid)\b",
    r"\buuid\s+(generate|create|new)\b",
]

_BASE64_PATTERNS: dict[str, list[str]] = {
    "encode": [
        r"(?:base64\s+)?encode\s+['\"]?(.+?)['\"]?\s+(?:in\s+)?(?:to\s+)?base64\b",
        r"base64\s+encode\s+['\"]?(.+?)['\"]?",
        r"encode\s+['\"]?(.+?)['\"]?\s+to\s+base64\b",
    ],
    "decode": [
        r"(?:base64\s+)?decode\s+['\"]?(.+?)['\"]?\s+(?:from\s+)?base64\b",
        r"base64\s+decode\s+['\"]?(.+?)['\"]?",
        r"decode\s+['\"]?(.+?)['\"]?\s+from\s+base64\b",
    ],
}

_HASH_PATTERNS: list[tuple[str, str, str]] = [
    (r"(?:generate\s+|compute\s+|create\s+)?(sha256|md5)\s+(?:hash\s+)?(?:of\s+)?['\"]?(.+?)['\"]?$", "algo", "data"),
    (r"(?:generate\s+|compute\s+|create\s+)?hash\s+['\"]?(.+?)['\"]?\s+(?:using|with)\s+(sha256|md5)", "data", "algo"),
    (r"^(sha256|md5)\s+(?:hash\s+)?(?:of\s+)?['\"]?(.+?)['\"]?$", "algo", "data"),
]

_JSON_PRETTY_PATTERNS: list[str] = [
    r"(?:pretty\s+print|prettify|format)\s+(?:this\s+)?json\s*[:\-]?\s*(.+)",
    r"json\s+(?:pretty\s+print|prettify|format)\s*(.+)",
]

_JSON_VALIDATE_PATTERNS: list[str] = [
    r"validate\s+(?:this\s+)?json\s*[:\-]?\s*(.+)",
    r"is\s+(?:this\s+)?(?:a\s+)?valid\s+json\s*[:\-]?\s*(.+)",
]

_DATE_PATTERNS: list[str] = [
    r"^(?:(?:what|what's)\s+)?(?:is\s+)?(?:the\s+)?(?:current\s+)?(?:date\s+(?:and\s+)?time|time\s+(?:and\s+)?date|date|time)\??$",
    r"^(?:what\s+)?(?:is\s+)?today(?:'s\s+date)?\??$",
    r"^(?:tell\s+me\s+)?(?:the\s+)?(?:current\s+)?(?:date|time|datetime)\??$",
]

_SYSTEM_INFO_PATTERNS: list[str] = [
    r"^(?:show\s+)?(?:system|computer|machine)\s+(?:info|information|details)\??$",
    r"^(?:what\s+)?(?:is\s+)?(?:my\s+)?(?:os|cpu|processor|ram|memory|system)\s*(?:info|information)?\??$",
    r"^(?:show|display|get)\s+(?:system|computer)\s*(?:info|information)?\??$",
]

_TEXT_SUMMARIZE_PATTERNS: list[str] = [
    r"summarize\s+['\"]?(.+?)['\"]?",
    r"summarize\s+(?:this\s+)?text\s*[:\-]?\s*(.+)",
]

_TEXT_WORD_COUNT_PATTERNS: list[str] = [
    r"(?:count|number\s+of)\s+words\s+(?:in\s+)?['\"]?(.+?)['\"]?",
    r"word\s+count\s+(?:of\s+)?['\"]?(.+?)['\"]?",
]

_SHELL_PATTERNS: list[str] = [
    r"^(?:run|execute)\s+(?:command\s+)?['\"]?(.+?)['\"]?$",
    r"^(?:run|execute)\s+(?:the\s+)?command\s*:\s*(.+)$",
]


# Priority-ordered intent classification patterns.
# Each entry is (label, patterns) where patterns are checked in order.
_GREETING_PATTERNS: list[str] = [
    r"^(hi|hello|hey|greetings|howdy)(\s+jarvis|\s+there)?[!\.]?$",
    r"^good\s+(morning|afternoon|evening|day)(\s+jarvis)?[!\.]?$",
    r"^(hi|hello|hey)\s+(there|everyone|guys|folks)[!\.]?$",
]

_CONVERSATION_PATTERNS: list[str] = [
    # Thanks
    r"^(thanks|thank you|thankyou|cheers|thx)(\s+jarvis)?[!\.]?$",
    r"^(appreciate|much appreciated)(\s+it)?[!\.]?$",
    r"^no\s+problem$",
    r"^(you're|you are)\s+welcome$",
    # Goodbye
    r"^(bye|goodbye|good bye|see\s+(you|ya)|cya|later|ttyl)[!\.]?$",
    r"^(have a|have) (good|great|nice)\s+(day|one|weekend)[!\.]?$",
    r"^take\s+care[!\.]?$",
    # Chit-chat
    r"^(how are you|how's it going|how are you doing|how do you do)[\?!]?$",
    r"^(what's up|sup|wassup)[\?!]?$",
    r"^(nice|nice to meet you)(\s+jarvis)?[!\.]?$",
    r"^(how are you today|how is it going)[\?!]?$",
    r"^(how's your day|how was your day)[\?!]?$",
]


# ── Explicit security keywords ──────────────────────────────────────────
# Security Agent activates ONLY for these explicit request patterns.
_SECURITY_PATTERNS: list[str] = [
    r"\bscan\b",
    r"\bvulnerab",           # matches vulnerability, vulnerabilities, vulnerable
    r"\bmalware\b",
    r"\bexploit\b",
    r"\bpentest\b",
    r"security\s+audit\b",
    r"\bfirewall\b",
    r"\bports?\b",
    r"\bnetwork\s+scan\b",
]

# ── Explicit devops keywords ────────────────────────────────────────────
# DevOps Agent activates ONLY for these explicit request patterns.
_DEVOPS_PATTERNS: list[str] = [
    r"\b(docker|kubernetes|k8s)\b",
    r"\b(ci/cd|ci\s+cd)\b",
    r"\b(deployment|terraform|ansible)\b",
    r"build\s+pipeline",
]

# ── Explicit shell execution keywords ───────────────────────────────────
_SHELL_EXPLICIT_PATTERNS: list[str] = [
    r"^(run|execute)\s+(a\s+)?(command|shell|terminal)\b",
    r"\b(shell\s+command|terminal\s+command)\b",
]

# ── Knowledge question patterns ─────────────────────────────────────────
# General factual questions that should be answered by the LLM directly,
# NOT routed to specialist agents.
_KNOWLEDGE_QUESTION_PATTERNS: list[tuple[str, float]] = [
    (r"^who\s+is\s+.+\?*$", 0.85),
    (r"^what\s+is\s+.+\?*$", 0.80),
    (r"^what\s+was\s+.+\?*$", 0.80),
    (r"^what\s+are\s+.+\?*$", 0.80),
    (r"^what\s+does\s+.+\?*$", 0.75),
    (r"^what\s+is\s+the\s+(meaning|definition|difference|purpose|goal)\s+of\s+.+\?*$", 0.85),
    (r"^what\s+do\s+you\s+know\s+about\s+.+\?*$", 0.80),
    (r"^where\s+(is|are|was|were|do|does|can)\s+.+\?*$", 0.75),
    (r"^when\s+(did|was|were|will|do|does|is|are)\s+.+\?*$", 0.75),
    (r"^why\s+(do|does|did|is|are|was|were|would|could|should)\s+.+\?*$", 0.80),
    (r"^how\s+(do|does|did|can|could|would|should|is|are|was|were|to)\s+.+\?*$", 0.75),
    (r"^explain\s+.+\?*$", 0.85),
    (r"^describe\s+.+\?*$", 0.80),
    (r"^tell\s+me\s+about\s+.+\?*$", 0.80),
    (r"^define\s+.+\?*$", 0.85),
    (r"^what\s+did\s+.+\?*$", 0.75),
    (r".*\btell\s+me\s+a\s+(joke|story|fact)\b.*", 0.90),
    (r".*\btell\s+me\s+something\b.*", 0.75),
]

# ── Browser/desktop/calendar/reminder/email/notes patterns ──────────────
_BROWSER_PATTERNS: list[str] = [
    r"^(open|launch|start)\s+(chrome|browser|firefox|edge|safari|opera)\b",
    r"\b(search|browse|navigate)\s+(the\s+)?(web|internet|website)\b",
    r"\b(go\s+to|open)\s+(https?://|www\.)\S+",
]

_DESKTOP_PATTERNS: list[str] = [
    r"(take|get|capture)\s+(a\s+)?(screenshot|screen\s+shot|snapshot)",
    r"(lock|sleep|shutdown|restart|log\s+off)\s+(the\s+)?(computer|pc|system|desktop)",
    r"(show|open|list)\s+(desktop|windows?|applications?)",
]

_CALENDAR_PATTERNS: list[str] = [
    r"\b(schedule|create|add|set|make|book)\s+(a\s+)?(meeting|appointment|event|calendar)",
    r"\b(what|show|list|check)\s+(my\s+)?(calendar|schedule|agenda)",
    r"\b(remind|reminder)\s+(me\s+)?(about|to|for)",
]

_EMAIL_PATTERNS: list[str] = [
    r"\b(send|compose|write|draft)\s+(an?\s+)?(email|mail|e-?mail|message)\b",
    r"\b(check|read|show|list)\s+(my\s+)?(inbox|email|messages)\b",
]

_NOTES_PATTERNS: list[str] = [
    r"\b(take|make|write|create|add)\s+(a\s+)?(note|notes)\b",
    r"\b(note\s+down|jot\s+down|record)\b",
]

# ── Follow-up / reference patterns ─────────────────────────────────────
# These indicate the user is referring to a previous topic.
_FOLLOW_UP_PATTERNS: list[str] = [
    r"\b(all of it|all of that|all that|all those|all this)\b",
    r"\b(the same|that one)\b",
    r"\b(more about|tell me more|elaborate|go on|continue)\b",
    r"\b(what about|how about)\s+(him|her|it|them|that)\b",
    r"^\s*(it|he|she|they)\s+.+",
]

# ── Current-info patterns (weather, news, stocks, time) ───────────────
# These should NEVER be answered by LLM hallucination — route to browser/search.
_CURRENT_INFO_PATTERNS: list[tuple[str, float, list[str]]] = [
    ("current_info", 0.90, [
        r"\bweather\b.*\b(today|tomorrow|now|current|forecast)\b",
        r"\b(weather|temperature|forecast)\s+(in|at|for)\s+\w+",
        r"\bwhat('s| is)\s+(the\s+)?(weather|temperature|forecast)\b",
        r"\bis\s+it\s+(raining|sunny|cloudy|cold|hot|warm)\b",
    ]),
    ("current_info", 0.85, [
        r"\bnews\b.*\b(today|latest|current|breaking|headlines)\b",
        r"\b(today|latest|current|breaking|headlines)\b.*\bnews\b",
        r"\bwhat('s| is)\s+(the\s+)?(news|headlines)\b",
    ]),
    ("current_info", 0.85, [
        r"\b(stock|share|market)\s+(price|quote|value|rate)\s+(of|for)\s+\w+",
        r"\bwhat('s| is)\s+(the\s+)?(stock|share)\s+(price|value)\s+(of|for)\s+\w+",
    ]),
    ("current_info", 0.90, [
        r"\b(current\s+)?time\s+(in|at|for)\s+\w+",
        r"\bwhat('s| is)\s+(the\s+)?(current\s+)?(time|date)\s+(in|at|for)\s+\w+",
    ]),
]

# ── Coding patterns (higher confidence) ─────────────────────────────────
_CODING_PATTERNS: list[str] = [
    r"\b(write|generate|create|implement|develop)\b.*\b(code|program|function|script|class|algorithm|module|search|sort)\b",
    r"\b(refactor|review|analyze|optimize)\s+(a\s+)?(code|function|class|module)\b",
]


class IntentDetector:
    """Detects whether a natural language query can be served by a tool.

    Usage::

        detector = IntentDetector(tool_manager)
        result = detector.detect("What is 25 * 76?")
        if result:
            tool_name, kwargs = result
            # route to ToolAgent

    Use :meth:`classify` for priority-ordered intent classification
    with confidence scoring::

        intent = detector.classify("Who is Purna?")
        intent.label       # "knowledge_question"
        intent.confidence  # 0.85
    """

    def __init__(self, tool_manager: ToolManager) -> None:
        self._tm = tool_manager

    def classify(self, query: str) -> IntentClassification:
        """Classify *query* into a priority-ordered intent label with
        confidence score.

        Priority tiers (checked in order):

        1. ``greeting`` (0.99)
        2. ``conversation`` (0.99)
        3. ``security`` (0.99) — ONLY explicit keywords
        4. ``devops`` (0.99) — ONLY explicit keywords
        5. ``shell`` (0.99) — ONLY explicit shell execution request
        6. ``follow_up`` (0.90) — user referring to a previous topic
        7. ``current_info`` (0.85-0.90) — weather, news, stocks, time
        8. ``tool`` (0.95) — explicit tool intent (calculator, datetime, etc.)
        9. ``browser`` / ``desktop`` / ``calendar`` / ``reminder`` / ``email`` / ``notes`` (0.85-0.95)
        10. ``coding`` (0.90)
        11. ``knowledge_question`` (0.75-0.95) — general factual questions
        12. ``unknown`` (0.0)

        Intents with confidence below :const:`CONFIDENCE_THRESHOLD` (0.70)
        should be treated as ambiguous and routed to the LLM directly.
        """
        stripped = query.strip()
        if not stripped:
            return IntentClassification("unknown", 0.0)

        _logger.debug("IntentDetector classify '%s'", stripped[:60])

        # 1. Greeting (highest priority)
        for pattern in _GREETING_PATTERNS:
            if _match(stripped, pattern):
                _logger.debug("  → greeting (0.99)")
                return IntentClassification("greeting", 0.99)

        # 2. Conversation
        for pattern in _CONVERSATION_PATTERNS:
            if _match(stripped, pattern):
                _logger.debug("  → conversation (0.99)")
                return IntentClassification("conversation", 0.99)

        # 3. Security (ONLY explicit keywords — checked BEFORE tool so that
        #    "scan my computer" → security, not detected as shell tool)
        if self._check_security(stripped):
            _logger.debug("  → security (0.99)")
            return IntentClassification("security", 0.99)

        # 4. DevOps (ONLY explicit keywords)
        if self._check_devops(stripped):
            _logger.debug("  → devops (0.99)")
            return IntentClassification("devops", 0.99)

        # 5. Shell (ONLY explicit shell execution request)
        if self._check_shell_explicit(stripped):
            _logger.debug("  → shell (0.99)")
            return IntentClassification("shell", 0.99)

        # 6. Follow-up / reference (user referring to a previous topic)
        for pattern in _FOLLOW_UP_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                _logger.debug("  → follow_up (0.90)")
                return IntentClassification("follow_up", 0.90)

        # 7. Current-info (weather, news, stocks, time — NEVER LLM)
        current = self._check_current_info(stripped)
        if current is not None:
            return current

        # 8. Tool (explicit tool intent — checked BEFORE knowledge questions
        #    so that "what is my processor" → tool, not knowledge_question)
        if self.detect(stripped) is not None:
            _logger.debug("  → tool (0.95)")
            return IntentClassification("tool", 0.95)

        # 9. Agent-specific intents (browser, desktop, calendar, etc.)
        agent_intent = self._check_agent_specific(stripped)
        if agent_intent is not None:
            return agent_intent

        # 10. Coding (medium-high confidence)
        for pattern in _CODING_PATTERNS:
            if _match(stripped, pattern, search=True):
                _logger.debug("  → coding (0.90)")
                return IntentClassification("coding", 0.90)

        # 11. Knowledge question (general factual questions → LLM direct)
        knowledge = self._check_knowledge_question(stripped)
        if knowledge is not None:
            return knowledge

        # 12. Unknown (zero confidence — route through LLM)
        _logger.debug("  → unknown (0.0)")
        return IntentClassification("unknown", 0.0)

    def _check_agent_specific(self, query: str) -> IntentClassification | None:
        """Check for browser, desktop, calendar, reminder, email, notes intents."""
        for pattern in _BROWSER_PATTERNS:
            if _match(query, pattern, search=True):
                _logger.debug("  → browser (0.90)")
                return IntentClassification("browser", 0.90)

        for pattern in _DESKTOP_PATTERNS:
            if _match(query, pattern, search=True):
                _logger.debug("  → desktop (0.85)")
                return IntentClassification("desktop", 0.85)

        for pattern in _CALENDAR_PATTERNS:
            if _match(query, pattern, search=True):
                _logger.debug("  → calendar (0.90)")
                return IntentClassification("calendar", 0.90)

        for pattern in _EMAIL_PATTERNS:
            if _match(query, pattern, search=True):
                _logger.debug("  → email (0.90)")
                return IntentClassification("email", 0.90)

        for pattern in _NOTES_PATTERNS:
            if _match(query, pattern, search=True):
                _logger.debug("  → notes (0.85)")
                return IntentClassification("notes", 0.85)

        return None

    @staticmethod
    def _check_security(query: str) -> bool:
        """Return True only for explicit security-related keywords."""
        lower = query.lower()
        for pattern in _SECURITY_PATTERNS:
            if re.search(pattern, lower):
                return True
        return False

    @staticmethod
    def _check_devops(query: str) -> bool:
        """Return True only for explicit devops-related keywords."""
        lower = query.lower()
        for pattern in _DEVOPS_PATTERNS:
            if re.search(pattern, lower):
                return True
        return False

    @staticmethod
    def _check_shell_explicit(query: str) -> bool:
        """Return True only for explicit shell execution requests."""
        for pattern in _SHELL_EXPLICIT_PATTERNS:
            if _match(query, pattern):
                return True
        return False

    @staticmethod
    def _check_current_info(query: str) -> IntentClassification | None:
        """Check for weather, news, stocks, time queries — route to browser, NEVER LLM."""
        lower = query.lower()
        for label, confidence, patterns in _CURRENT_INFO_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, lower, re.IGNORECASE):
                    _logger.debug("  → %s (%.2f)", label, confidence)
                    return IntentClassification(label, confidence)
        return None

    @staticmethod
    def _check_knowledge_question(query: str) -> IntentClassification | None:
        """Check if *query* is a general knowledge question.

        Returns the classification with appropriate confidence or None.
        """
        lower = query.strip().lower()
        for pattern, confidence in _KNOWLEDGE_QUESTION_PATTERNS:
            if re.match(pattern, lower, re.IGNORECASE):
                _logger.debug("  → knowledge_question (%.2f)", confidence)
                return IntentClassification("knowledge_question", confidence)
        return None

    def detect(self, query: str) -> tuple[str, dict[str, Any]] | None:
        """Return ``(tool_name, kwargs)`` if a tool intent is detected, else ``None``."""
        detectors = [
            self._detect_calculator,
            self._detect_uuid,
            self._detect_base64,
            self._detect_hash,
            self._detect_json,
            self._detect_datetime,
            self._detect_system_info,
            self._detect_text_summarize,
            self._detect_text_word_count,
            self._detect_shell,
        ]
        stripped = query.strip()
        for detector in detectors:
            result = detector(stripped)
            if result is not None:
                tool_name, kwargs = result
                if self._tm.is_tool_enabled(tool_name):
                    return (tool_name, kwargs)
        return None

    # ------------------------------------------------------------------
    # Calculator — tries several strategies in order:
    #   1. Pure math expression (e.g. "234+56", "sqrt(81)", "2^100")
    #   2. Math expression with natural-language prefix stripped
    #      (e.g. "what is (234*567)" → "(234*567)")
    #   3. Classic natural-language pattern
    #      (e.g. "what is 2+2", "calculate 100/5")
    #
    # Queries containing explanation keywords ("explain", "why", …)
    # are NOT routed here — they go to the LLM.
    # ------------------------------------------------------------------

    def _detect_calculator(self, query: str) -> tuple[str, dict[str, Any]] | None:
        stripped = query.strip()
        if not stripped:
            return None

        # If the user is asking for an explanation, do NOT route to calculator.
        if self._wants_explanation(stripped):
            return None

        # Strategy 1: pure math expression
        expr = self._try_pure_expression(stripped)
        if expr is not None:
            return ("calculator", {"expression": expr})

        # Strategy 2: strip common English prefixes, then validate remainder
        expr = self._try_prefix_stripped(stripped)
        if expr is not None:
            return ("calculator", {"expression": expr})

        # Strategy 3: classic natural-language patterns
        expr = self._try_natural_language(stripped)
        if expr is not None:
            return ("calculator", {"expression": expr})

        return None

    @staticmethod
    def _wants_explanation(query: str) -> bool:
        """Check if *query* contains explanation-seeking phrases."""
        lower = query.lower()
        for phrase in _EXPLANATION_PHRASES:
            if phrase in lower:
                return True
        return False

    @staticmethod
    def _try_pure_expression(text: str) -> str | None:
        """Check if *text* is a standalone math expression.

        Handles:
        - ``234+56``
        - ``(234*89)+100``
        - ``sqrt(81)``
        - ``sin(45)``
        - ``2^100``
        - ``1e9*1e9``
        """
        if not text:
            return None

        # Quick structural check: does it look like math?
        # Must contain at least one digit or start with a math keyword.
        has_digit = bool(re.search(r"\d", text))
        starts_with_math_kw = bool(re.match(rf"^\s*({_MATH_FUNC_NAMES})\s*\(", text, re.IGNORECASE))
        if not has_digit and not starts_with_math_kw:
            return None

        # Check that every token is a valid math token
        if not _PURE_MATH_RE.match(text):
            return None

        # Validate via AST parsing
        if _EXPRESSION_SAFETY.is_pure_expression(text):
            return text
        return None

    @staticmethod
    def _try_prefix_stripped(text: str) -> str | None:
        """Strip common English prefixes and check if the remainder is math.

        Handles:
        - ``what is (234*567)`` → ``(234*567)``
        - ``what is sqrt(81)`` → ``sqrt(81)``
        - ``what's 2+2``       → ``2+2``
        """
        lower_text = text
        for prefix in _MATH_PREFIX_PATTERNS:
            m = re.match(prefix, lower_text, re.IGNORECASE)
            if m:
                rest = text[m.end():].strip().rstrip("?.,!")
                if rest and _EXPRESSION_SAFETY.is_pure_expression(rest):
                    return rest
        return None

    @staticmethod
    def _try_natural_language(text: str) -> str | None:
        """Legacy regex-based natural language math detection.

        Handles remaining cases like:
        - ``what is 2+2`` (if prefix stripping didn't catch it)
        - ``25 * 4 = ?``
        """
        # Fallback patterns for formats not caught by prefix stripping
        patterns = [
            r"^what\s+is\s+([\d\s+\-*/()%^.,a-zA-Z]+)\?*$",
            r"^(\d[\d\s+\-*/()%^.,a-zA-Z]*)\s*=\s*\?*$",
        ]
        for pattern in patterns:
            m = _match(text, pattern)
            if m:
                expr = m.group(1).strip().rstrip("?.,!")
                if _EXPRESSION_SAFETY.is_pure_expression(expr):
                    return expr
        return None

    # ------------------------------------------------------------------
    # UUID
    # ------------------------------------------------------------------

    def _detect_uuid(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for pattern in _UUID_PATTERNS:
            if _match(query, pattern, search=True):
                m = _match(query, r"(\d+)\s+(uuid|guid)", search=True)
                count = int(m.group(1)) if m else 1
                return ("uuid", {"count": min(count, 100)})
        return None

    # ------------------------------------------------------------------
    # Base64 (preserves original case in captured data)
    # ------------------------------------------------------------------

    def _detect_base64(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for operation, patterns in _BASE64_PATTERNS.items():
            for pattern in patterns:
                m = _match(query, pattern, search=True)
                if m:
                    data = m.group(1).strip().strip("'\"")
                    return ("base64", {"operation": operation, "data": data})
        return None

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    def _detect_hash(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for pattern, group1, group2 in _HASH_PATTERNS:
            m = _match(query, pattern, search=True)
            if m:
                data = m.group(2).strip().strip("'\"") if group2 == "data" else m.group(1).strip().strip("'\"")
                algo = m.group(1).strip() if group1 == "algo" else m.group(2).strip()
                algo_map = {"sha256": "sha256", "md5": "md5"}
                normalized = algo_map.get(algo.lower())
                if normalized and data:
                    return ("hash", {"algorithm": normalized, "data": data})
        return None

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _detect_json(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for pattern in _JSON_PRETTY_PATTERNS:
            m = _match(query, pattern, search=True)
            if m:
                return ("json", {"operation": "pretty_print", "data": m.group(1).strip()})

        for pattern in _JSON_VALIDATE_PATTERNS:
            m = _match(query, pattern, search=True)
            if m:
                return ("json", {"operation": "validate", "data": m.group(1).strip()})

        if re.search(r"\bpretty\s*print\b", query, re.IGNORECASE) and re.search(r"\bjson\b", query, re.IGNORECASE):
            return ("json", {"operation": "pretty_print", "data": "{}"})

        return None

    # ------------------------------------------------------------------
    # Datetime
    # ------------------------------------------------------------------

    def _detect_datetime(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for pattern in _DATE_PATTERNS:
            if _match(query, pattern):
                return ("datetime", {})
        return None

    # ------------------------------------------------------------------
    # System info
    # ------------------------------------------------------------------

    def _detect_system_info(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for pattern in _SYSTEM_INFO_PATTERNS:
            if _match(query, pattern):
                return ("system_info", {})
        return None

    # ------------------------------------------------------------------
    # Text (summarize)
    # ------------------------------------------------------------------

    def _detect_text_summarize(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for pattern in _TEXT_SUMMARIZE_PATTERNS:
            m = _match(query, pattern, search=True)
            if m:
                text = m.group(1).strip().strip("'\"")
                return ("text", {"operation": "summarize", "text": text})
        return None

    # ------------------------------------------------------------------
    # Text (word count)
    # ------------------------------------------------------------------

    def _detect_text_word_count(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for pattern in _TEXT_WORD_COUNT_PATTERNS:
            m = _match(query, pattern, search=True)
            if m:
                text = m.group(1).strip().strip("'\"")
                return ("text", {"operation": "word_count", "text": text})
        return None

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    def _detect_shell(self, query: str) -> tuple[str, dict[str, Any]] | None:
        for pattern in _SHELL_PATTERNS:
            m = _match(query, pattern)
            if m:
                command = m.group(1).strip()
                return ("shell", {"command": command})
        return None
