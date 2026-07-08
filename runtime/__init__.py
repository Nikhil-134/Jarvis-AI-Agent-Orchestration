"""Runtime layer — single entry point for all user interaction.

The runtime wraps existing agents, planners, workflows, and tools
with a production-grade orchestration layer that handles intent
detection, context tracking, personality, error recovery, and
response formatting.

Usage::

    runtime = ConversationRuntime(orchestrator, settings)
    result = await runtime.process("Hello!")
    print(result)
"""

from runtime.conversation_runtime import ConversationRuntime
from runtime.runtime import Runtime

__all__ = [
    "ConversationRuntime",
    "Runtime",
]
