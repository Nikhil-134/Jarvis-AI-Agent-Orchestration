"""Conversation manager — persistent multi-turn conversation state.

Manages ChatSession lifecycle across user turns, with automatic
context window management, memory enrichment, and summarization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from llm import BaseLLMProvider, ChatSession, LLMError, ToolDefinition
from llm.interfaces import LLMResponse
from memory import MemoryService

_logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """A single conversation with session tracking."""

    session_id: str
    chat_session: ChatSession
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn_count: int = 0
    summary: str = ""


class ConversationManager:
    """Manages persistent multi-turn conversations.

    Maintains a ChatSession across CLI turns, handles context window
    management, automatic memory enrichment, and conversation
    summarization for long-running dialogs.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        system_prompt: str | None = None,
        max_history_tokens: int = 2048,
        max_session_age_minutes: int = 60,
    ) -> None:
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._system_prompt = system_prompt or (
            "You are Jarvis, an AI operating system. You are helpful, "
            "conversational, and remember context from previous turns."
        )
        self._max_history_tokens = max_history_tokens
        self._max_session_age_minutes = max_session_age_minutes
        self._conversation: Conversation | None = None

    @property
    def conversation(self) -> Conversation | None:
        return self._conversation

    @property
    def session_id(self) -> str | None:
        return self._conversation.session_id if self._conversation else None

    @property
    def turn_count(self) -> int:
        return self._conversation.turn_count if self._conversation else 0

    async def get_or_create_session(self) -> Conversation:
        if self._conversation is None:
            self._conversation = await self._create_session()
        else:
            self._conversation.last_active = datetime.now(timezone.utc)
        return self._conversation

    async def _create_session(self) -> Conversation:
        session_id = str(uuid4())
        chat_session = ChatSession(
            provider=self._llm_provider,
            system_prompt=self._system_prompt,
        ) if self._llm_provider else ChatSession(
            provider=None,
            system_prompt=self._system_prompt,
        )
        conv = Conversation(session_id=session_id, chat_session=chat_session)
        _logger.info("Created conversation session '%s'", session_id)
        return conv

    async def process_turn(
        self,
        user_input: str,
        tools: list[ToolDefinition] | None = None,
    ) -> str:
        conversation = await self.get_or_create_session()
        conversation.turn_count += 1
        conversation.last_active = datetime.now(timezone.utc)

        memory_context = await self._get_memory_context(user_input)

        enriched_input = memory_context + "\n\n" + user_input if memory_context else user_input

        try:
            if self._llm_provider is None:
                return f"I received: {user_input}"

            response: LLMResponse = await conversation.chat_session.send(
                enriched_input,
                tools=tools,
            )

            if response.tool_calls:
                _logger.info("LLM requested %d tool call(s)", len(response.tool_calls))
                tool_result = self._format_tool_calls(response.tool_calls)
                follow_up = await conversation.chat_session.send(
                    f"The following tool results were obtained:\n{tool_result}\n\nPlease provide a final response.",
                )
                follow_up_content = follow_up.content or ""
                conversation.chat_session.replace_last_assistant(follow_up_content)
                return follow_up_content

            result = response.content or ""

            if self._memory_service is not None:
                try:
                    await self._memory_service.store_interaction(user_input, result)
                except Exception:
                    _logger.exception("Failed to store interaction in memory")

            await self._maybe_summarize(conversation)

            return result

        except LLMError as exc:
            _logger.exception("LLM error during conversation turn")
            return f"I encountered an error processing your request: {exc}"

    async def _get_memory_context(self, user_input: str) -> str:
        if self._memory_service is None:
            return ""
        try:
            enriched, memories = await self._memory_service.enrich_prompt(
                user_input, top_k=5, per_memory_chars=1000, max_context_length=3000,
            )
            if memories:
                return enriched
        except Exception:
            _logger.exception("Memory enrichment failed")
        return ""

    async def get_context(self, conversation: Conversation | None = None) -> str:
        """Return conversation context string for enriching prompts.

        Includes the conversation summary (if available) and recent
        history for context restoration across turns.
        """
        conv = conversation or self._conversation
        if conv is None:
            return ""

        parts: list[str] = []

        if conv.summary:
            parts.append(f"Conversation summary: {conv.summary}")

        history = conv.chat_session.history
        if history:
            recent = "\n".join(
                f"{m.role}: {m.content[:200]}"
                for m in history[-6:]
            )
            parts.append(f"Recent conversation:\n{recent}")

        return "\n\n".join(parts) if parts else ""

    async def summarize(self, conversation: Conversation | None = None) -> str:
        """Explicitly summarize the conversation, returning the summary text."""
        conv = conversation or self._conversation
        if conv is None:
            return ""

        history = conv.chat_session.history
        if len(history) < 4:
            conv.summary = ""
            return ""

        try:
            recent = "\n".join(
                f"{m.role}: {m.content[:200]}"
                for m in history[-10:]
            )
            prompt = (
                "Summarize the key points of this conversation so far, "
                "focusing on user preferences, facts discussed, and any "
                "action items:\n\n" + recent
            )
            summary = await conv.chat_session.provider.generate(
                prompt,
                system_prompt="You are a conversation summarizer.",
            )
            conv.summary = summary.content
            _logger.info("Conversation summarized at turn %d", conv.turn_count)
            return conv.summary
        except Exception:
            _logger.exception("Conversation summarization failed")
            return conv.summary or ""

    async def _maybe_summarize(self, conversation: Conversation) -> None:
        if conversation.turn_count < 5:
            return
        if conversation.turn_count % 10 != 0:
            return

        history = conversation.chat_session.history
        if len(history) < 20:
            return

        await self.summarize(conversation)

    def _format_tool_calls(self, tool_calls: Any) -> str:
        results: list[str] = []
        for tc in tool_calls:
            results.append(f"Tool '{tc.name}' called with args: {tc.arguments}")
        return "\n".join(results)

    def clear(self) -> None:
        self._conversation = None
        _logger.info("Conversation cleared")
