"""Custom exceptions for orchestration failures."""


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""


class AgentAlreadyRegisteredError(OrchestratorError):
    """Raised when an agent name is registered more than once."""


class NoAgentForTaskError(OrchestratorError):
    """Raised when no registered agent can handle a task."""


class AgentNotRegisteredError(OrchestratorError):
    """Raised when an agent name is not registered."""
