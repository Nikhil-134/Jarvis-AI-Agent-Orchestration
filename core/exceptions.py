"""Unified exception hierarchy for Jarvis.

All Jarvis exceptions inherit from :class:`JarvisError`, enabling
top-level catch-all handling without losing type specificity.
"""


class JarvisError(Exception):
    """Base exception for all Jarvis system errors."""


# ── Orchestrator ──────────────────────────────────────────────────────────

class OrchestratorError(JarvisError):
    """Base exception for orchestrator errors."""


class AgentAlreadyRegisteredError(OrchestratorError):
    """Raised when an agent name is registered more than once."""


class NoAgentForTaskError(OrchestratorError):
    """Raised when no registered agent can handle a task."""


class AgentNotRegisteredError(OrchestratorError):
    """Raised when an agent name is not registered."""


# ── LLM ───────────────────────────────────────────────────────────────────

class LLMError(JarvisError):
    """Base exception for LLM provider failures."""


class LLMProviderError(LLMError):
    """Raised when a provider returns an invalid or failed response."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM provider request times out."""


# ── Memory ────────────────────────────────────────────────────────────────

class MemoryError(JarvisError):
    """Base exception for memory system failures."""


class MemoryStorageError(MemoryError):
    """Raised when a storage backend operation fails."""


class MemoryRetrievalError(MemoryError):
    """Raised when a memory retrieval operation fails."""


class MemorySearchError(MemoryError):
    """Raised when a semantic search operation fails."""


class MemoryEmbeddingError(MemoryError):
    """Raised when embedding generation fails."""


class MemoryNotFoundError(MemoryError):
    """Raised when a requested memory does not exist."""


class MemoryConfigurationError(MemoryError):
    """Raised when memory system configuration is invalid."""


class MemoryDeduplicationError(MemoryError):
    """Raised when deduplication processing fails."""


class MemoryValidationError(MemoryError):
    """Raised when a memory object fails validation (corrupt, oversized, unsafe)."""


# ── Tools ─────────────────────────────────────────────────────────────────

class ToolError(JarvisError):
    """Base exception for all tool system errors."""


class ToolNotFoundError(ToolError):
    """Raised when a tool is not found in registry."""


class ToolAlreadyRegisteredError(ToolError):
    """Raised when registering a tool with a duplicate name."""


class ToolExecutionError(ToolError):
    """Raised when a tool execution fails."""


class ToolPermissionDeniedError(ToolError):
    """Raised when tool execution is denied by permissions."""


class ToolValidationError(ToolError):
    """Raised when tool arguments fail validation."""


# ── Agent ─────────────────────────────────────────────────────────────────

class AgentError(JarvisError):
    """Base exception for agent-level failures."""


class AgentConfigurationError(AgentError):
    """Raised when an agent is misconfigured."""


class AgentExecutionError(AgentError):
    """Raised when an agent fails during task execution."""
