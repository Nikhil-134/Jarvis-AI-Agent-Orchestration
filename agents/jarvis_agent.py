"""Jarvis Prime Agent — single entry point for all user requests.

Every user request flows through JarvisPrimeAgent. It:

1. Maintains conversation state via ConversationManager
2. Enriches input with memory context via RAG pipeline
3. Decomposes goals into specialist task types
4. Builds and executes workflow plans via WorkflowEngine
5. Merges all agent outputs via ResponseComposer
6. Persists interactions into long-term memory
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask
from agents.context_manager import ContextManager
from agents.conversation_manager import ConversationManager
from agents.planner import PlannerAgent
from agents.response_composer import ResponseComposer
from llm import BaseLLMProvider, PromptManager
from memory import MemoryService
from orchestrator.workflow import WorkflowEngine, WorkflowPlan, WorkflowStep

_logger = logging.getLogger(__name__)


# Simple conversational phrases that bypass specialist agents and workflow.
# These are matched exactly (case-insensitive, after stripping punctuation).
_SIMPLE_CONVERSATIONS: frozenset[str] = frozenset({
    # Greetings
    "hi", "hello", "hey", "greetings", "howdy",
    "good morning", "good afternoon", "good evening", "good day",
    "hi jarvis", "hello jarvis", "hey jarvis",
    "hi there", "hello there", "hey there",
    # Gratitude
    "thanks", "thank you", "thankyou", "cheers",
    "thanks jarvis", "thank you jarvis",
    "appreciate it", "much appreciated",
    "no problem", "you're welcome", "you are welcome",
    # Goodbye
    "bye", "goodbye", "good bye", "see you",
    "see you later", "see ya", "bye bye",
    "bye jarvis", "goodbye jarvis",
    "have a good day", "take care",
    # Chit-chat
    "how are you", "how's it going", "how are you doing",
    "how do you do", "what's up", "sup",
    "nice", "nice to meet you",
    "how are you today", "how is it going",
})

_GREETING_WORDS: frozenset[str] = frozenset({
    "hi", "hello", "hey", "greetings", "howdy", "good", "morning",
    "afternoon", "evening", "day", "how", "are", "you", "how's",
    "it", "going", "what's", "whats", "up", "sup", "thanks", "thank",
    "bye", "goodbye", "see", "later", "ya", "nice", "meet", "to",
    "doing", "do", "appreciate", "much", "no", "problem", "you're",
    "welcome", "your", "have", "a", "take", "care", "jarvis",
    "there", "dear", "sir", "ma'am", "today", "is", "bye",
})


class JarvisPrimeAgent(Agent):
    """Central orchestrator agent — single entry point for all user requests.

    Delegates task execution to specialist agents through the
    WorkflowEngine and merges their outputs into one coherent response.

    Owns the ConversationManager for persistent multi-turn state and
    the PlannerAgent for LLM-driven response generation on simple queries.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        workflow_engine: WorkflowEngine | None = None,
        planner_agent: PlannerAgent | None = None,
        response_composer: ResponseComposer | None = None,
        prompt_manager: PromptManager | None = None,
        *,
        max_context_tokens: int = 4096,
        max_chunk_tokens: int = 2048,
        chars_per_token: float = 3.5,
    ) -> None:
        supported = ("jarvis.process", "plan", "tool.execute", "knowledge", "conversation")
        super().__init__(name="jarvis", supported_task_types=supported)

        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._workflow_engine = workflow_engine
        self._prompt_manager = prompt_manager or PromptManager()
        self._max_context_tokens = max_context_tokens
        self._max_chunk_tokens = max_chunk_tokens
        self._chars_per_token = chars_per_token

        self._planner_agent = planner_agent or PlannerAgent(
            llm_provider=llm_provider,
            memory_service=memory_service,
            max_context_tokens=max_context_tokens,
            max_chunk_tokens=max_chunk_tokens,
            chars_per_token=chars_per_token,
        )

        self._response_composer = response_composer or ResponseComposer(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
        )

        self._conversation_manager = ConversationManager(
            llm_provider=llm_provider,
            memory_service=memory_service,
            max_history_tokens=max_context_tokens // 2,
        )

        self._context_manager = ContextManager(window_size=5)

        self._specialist_map: dict[str, str] = {
            "research": "friday",
            "information.retrieve": "friday",
            "information.synthesize": "friday",
            "code.generate": "veronica",
            "code.review": "veronica",
            "code.refactor": "veronica",
            "code.analyze": "veronica",
            "vision.analyze": "vision",
            "vision.ocr": "vision",
            "vision.screenshot": "vision",
            "vision.describe": "vision",
            "security.scan": "ultron",
            "security.analyze": "ultron",
            "system.monitor": "ultron",
            "strategy.plan": "athena",
            "task.decompose": "athena",
            "workflow.design": "athena",
            "build.compile": "stark",
            "build.deploy": "stark",
            "project.setup": "stark",
            "test.run": "steve",
            "test.create": "steve",
            "test.analyze": "steve",
            "coverage.report": "steve",
            "knowledge.store": "oracle",
            "knowledge.query": "oracle",
            "knowledge.index": "oracle",
            "knowledge.search": "oracle",
            "browser.navigate": "gecko",
            "browser.scrape": "gecko",
            "web.automate": "gecko",
            "web.fetch": "gecko",
            "compute.process": "hercules",
            "data.transform": "hercules",
            "batch.execute": "hercules",
            "storage.organize": "hulk",
            "storage.backup": "hulk",
            "storage.cleanup": "hulk",
            "storage.analyze": "hulk",
            "devops.configure": "jerome",
            "devops.deploy": "jerome",
            "system.admin": "jerome",
            "devops.monitor": "jerome",
            "ux.notify": "pepper",
            "ux.display": "pepper",
            "ux.interact": "pepper",
            "ux.speak": "pepper",
            "memory.store": "memory",
            "memory.retrieve": "memory",
            "memory.search": "memory",
            "memory.forget": "memory",
            "memory.stats": "memory",
            "tool.execute": "tool",
        }

    @property
    def conversation_manager(self) -> ConversationManager:
        return self._conversation_manager

    @property
    def context_manager(self) -> ContextManager:
        return self._context_manager

    @property
    def planner_agent(self) -> PlannerAgent:
        return self._planner_agent

    @property
    def response_composer(self) -> ResponseComposer:
        return self._response_composer

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"JarvisPrimeAgent cannot handle task type: {task.task_type}",
            )

        goal = str(task.payload.get("goal", ""))
        task_type = task.payload.get("task_type", task.task_type)

        if task_type == "tool.execute":
            return await self._handle_tool_execution(task)

        if task_type == "knowledge":
            return await self._handle_knowledge_question(goal, task)

        if task_type in ("jarvis.process", "plan", "conversation"):
            return await self._process_goal(goal, task)

        return await self._process_goal(goal, task)

    async def _process_goal(self, goal: str, task: AgentTask) -> AgentResult:
        """Core processing pipeline for a user goal."""
        try:
            # 0. Resolve pronouns and follow-up references using context
            goal = self._context_manager.enrich(goal)

            # 1. Maintain conversation state
            conversation = await self._conversation_manager.get_or_create_session()
            conversation.turn_count += 1

            # 1a. Handle simple conversation directly — no memory enrichment,
            #     no decomposition, no workflow, no specialist agents.
            if self._is_simple_conversation(goal):
                result = await self._handle_conversation(goal, task)
                self._context_manager.update(goal, result.data.get("response", ""))
                return result

            # 2. Enrich with memory context
            enriched_goal = await self._enrich_with_memory(goal)

            # 3. Retrieve memories for the planner
            memories = await self._planner_agent._retrieve_memories(enriched_goal)

            # 4. Decompose into task types
            task_types = await self._decompose_goal(enriched_goal)

            # 5. Execute via workflow
            if not task_types:
                result = await self._planner_agent.handle(
                    AgentTask(task_type="plan", payload={"goal": enriched_goal})
                )
                response = result.data.get("response", "")
                memories = result.data.get("memory_count", 0) > 0
            else:
                response = await self._execute_and_merge(goal, task_types)

            # 6. Persist interaction
            await self._store_interaction(goal, response)

            self._context_manager.update(goal, response)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=True,
                message="",
                data={
                    "status": "completed",
                    "response": response,
                    "memory_enriched": len(memories) > 0 if isinstance(memories, list) else memories,
                    "memory_count": len(memories) if isinstance(memories, list) else (1 if memories else 0),
                },
            )

        except Exception as exc:
            _logger.exception("JarvisPrimeAgent processing failed")
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=str(exc),
                data={"status": "failed", "error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Simple conversation detection & handling
    # ------------------------------------------------------------------

    @staticmethod
    def _is_simple_conversation(goal: str) -> bool:
        """Return True if *goal* is a simple conversational phrase (greeting,
        thanks, goodbye, chit-chat) that needs NO specialist agents or workflow.

        Matches exact phrases first, then falls back to checking that every
        word in the input is a recognised greeting/conversation word.  This
        ensures compound greetings like ``"hello, how are you?"`` still match
        while task requests like ``"hello world program in python"`` do not.
        """
        lower = goal.strip().lower().rstrip("?!.,;:")
        if not lower:
            return False

        # Exact phrase match (fast path)
        if lower in _SIMPLE_CONVERSATIONS:
            return True

        # Word-level check: every word must be from the greeting vocabulary
        words = lower.split()
        if not words or len(words) > 6:
            return False  # too long to be a simple greeting
        return all(w.strip("?!.,;:") in _GREETING_WORDS for w in words)

    async def _handle_conversation(self, goal: str, task: AgentTask) -> AgentResult:
        """Generate a direct conversational response for simple greetings,
        thanks, goodbyes and chit-chat — bypassing PlannerAgent, WorkflowEngine
        and all specialist agents."""
        lower = goal.strip().lower().rstrip("?!.,;: ")

        # Determine the response tone based on what the user said
        if any(w in lower for w in ("bye", "goodbye", "see you", "see ya", "take care")):
            response = "Goodbye! Have a great day!"
        elif any(w in lower for w in ("thanks", "thank you", "cheers", "appreciate", "welcome")):
            response = "You're welcome! Let me know if you need anything else."
        elif any(w in lower for w in ("morning",)):
            response = "Good morning! How can I help you today?"
        elif any(w in lower for w in ("afternoon",)):
            response = "Good afternoon! How can I help you?"
        elif any(w in lower for w in ("evening",)):
            response = "Good evening! How can I help you?"
        elif any(w in lower for w in ("how are you", "how's it going", "how do you do", "what's up", "sup")):
            response = "I'm doing well, thank you! How can I assist you today?"
        elif any(w in lower for w in ("nice",)):
            response = "Nice to meet you too! How can I help you?"
        else:
            response = "Hello! How can I help you today?"

        _logger.info("Routed '%s' as direct conversation (tone=%s)", goal, response[:40])

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="",
            data={
                "status": "completed",
                "response": response,
                "memory_enriched": False,
                "memory_count": 0,
            },
        )

    async def _handle_knowledge_question(self, goal: str, task: AgentTask) -> AgentResult:
        """Handle a general knowledge question directly through the LLM,
        bypassing decomposition, workflow, and all specialist agents.

        This path is used for:
        - Who/What/Where/When/Why/How questions
        - Explain/Describe/Define requests
        - Any ambiguous or low-confidence intent
        """
        _logger.info("Routing '%s' as knowledge question (direct LLM, no specialists)", goal[:60])

        # Resolve pronouns and follow-up references using context
        enriched_goal = self._context_manager.enrich(goal)

        # Use PlannerAgent for LLM-driven response generation
        result = await self._planner_agent.handle(
            AgentTask(task_type="plan", payload={"goal": enriched_goal})
        )
        response = result.data.get("response", "")

        self._context_manager.update(goal, response)

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="",
            data={
                "status": "completed",
                "response": response,
                "memory_enriched": False,
                "memory_count": 0,
            },
        )

    async def _execute_and_merge(self, goal: str, task_types: list[str]) -> str:
        """Execute task types through the workflow engine and merge results."""
        plan = WorkflowPlan(goal=goal)

        for tt in task_types:
            plan.add_step(WorkflowStep(
                task_type=tt,
                payload={"goal": goal},
            ))

        if self._workflow_engine is None:
            _logger.warning("WorkflowEngine not available, using planner fallback")
            result = await self._planner_agent.handle(
                AgentTask(task_type="plan", payload={"goal": goal})
            )
            return result.data.get("response", "Task completed.")

        step_results = await self._workflow_engine.execute(plan)
        return await self._response_composer.merge(goal, step_results)

    # ------------------------------------------------------------------
    # Memory enrichment
    # ------------------------------------------------------------------

    async def _enrich_with_memory(self, goal: str) -> str:
        if self._memory_service is None:
            return goal

        try:
            enriched, _ = await self._memory_service.enrich_prompt(
                goal, top_k=5, per_memory_chars=1000, max_context_length=3000,
            )
            return enriched
        except Exception:
            _logger.exception("Memory enrichment failed")
            return goal

    async def _store_interaction(self, goal: str, response: str) -> None:
        if self._memory_service is None:
            return
        try:
            await self._memory_service.store_interaction(goal, response)
        except Exception:
            _logger.exception("Failed to store interaction")

    # ------------------------------------------------------------------
    # Goal decomposition
    # ------------------------------------------------------------------

    async def _decompose_goal(self, goal: str) -> list[str]:
        """Decompose *goal* into a list of specialist task types.

        Uses LLM when available, otherwise rule-based keyword matching.
        Returns an empty list for simple conversational queries.
        """
        # Simple conversational queries need NO specialist decomposition
        if self._is_simple_conversation(goal):
            _logger.info("Skipping decomposition for simple conversation '%s'", goal[:60])
            return []

        if self._llm_provider is None:
            return await self._rule_based_decompose(goal)

        try:
            prompt = self._prompt_manager.render(
                "decomposer",
                goal=goal,
                specialist_list="\n".join(
                    f"  - {v} ({k})" for k, v in sorted(self._specialist_map.items())
                ),
            )
            response = await self._llm_provider.generate_text(
                prompt,
                system_prompt=(
                    "You are a task decomposition specialist. "
                    "Given a user request, identify which specialist "
                    "task types are needed. If the request is a simple "
                    "conversation (greeting, chit-chat, simple question), "
                    "return an empty response. "
                    "Otherwise return ONLY a comma-separated list of "
                    "task_type values, nothing else."
                ),
            )
            # Defensive: a provider may hand back None/empty — never assume str.
            task_types = [t.strip() for t in (response or "").split(",") if t.strip()]
            valid = [t for t in task_types if t in self._specialist_map]
            return valid or await self._rule_based_decompose(goal)
        except Exception:
            _logger.exception("LLM-based decomposition failed")
            return await self._rule_based_decompose(goal)

    async def _rule_based_decompose(self, goal: str) -> list[str]:
        """Rule-based decomposition with keyword matching."""
        lower = goal.lower()
        types: list[str] = []

        # Simple conversation — no specialist needed
        if self._is_simple_conversation(goal):
            return types

        # NOTE: "what is", "who is", "tell me about" are deliberately excluded
        # from research keywords. They are general knowledge questions handled
        # by the LLM directly via the "knowledge" task_type path.
        if any(w in lower for w in ("search", "research", "find", "look up", "lookup", "investigate")):
            types.append("information.retrieve")

        if any(w in lower for w in ("generate", "write code", "create a", "implement", "develop")):
            types.append("code.generate")

        if any(w in lower for w in ("review", "refactor", "analyze code")):
            types.append("code.review")

        if any(w in lower for w in ("scan", "security", "vulnerability", "monitor")):
            types.append("security.scan")

        if any(w in lower for w in ("test", "coverage", "qa")):
            types.append("test.run")

        if any(w in lower for w in ("deploy", "build", "compile")):
            types.append("build.compile")

        if any(w in lower for w in ("backup", "organize", "cleanup", "storage")):
            types.append("storage.organize")

        if any(w in lower for w in ("configure", "devops", "deploy to", "system admin")):
            types.append("devops.configure")

        if any(w in lower for w in ("fetch", "scrape", "web", "browser", "url", "http")):
            types.append("web.fetch")

        if any(w in lower for w in ("compute", "calculate", "transform", "process data")):
            types.append("compute.process")

        return types

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _handle_tool_execution(self, task: AgentTask) -> AgentResult:
        tool_name = task.payload.get("tool_name", "")
        arguments = task.payload.get("tool_arguments", task.payload.get("arguments", {}))
        original_goal = task.payload.get("goal", "")
        _logger.info("Tool execution requested: %s with args=%s", tool_name, arguments)

        # Preserve original goal; if planner needs to detect the tool, it has the full goal
        goal = original_goal if original_goal else f"Execute the {tool_name} tool"

        if self._workflow_engine is not None:
            plan = WorkflowPlan(goal=goal)
            plan.add_step(WorkflowStep(
                task_type="tool.execute",
                payload={"tool_name": tool_name, "arguments": arguments, "goal": goal},
            ))
            step_results = await self._workflow_engine.execute(plan)

            # Extract raw tool output and tool_name for ResponseSynthesizer
            raw_output = ""
            executed_tool = tool_name
            if step_results:
                step_data = step_results[0].data or {}
                if step_results[0].success:
                    raw_output = step_data.get("output", "")
                    executed_tool = step_data.get("tool_name", tool_name)
                else:
                    raw_output = step_data.get("output", "")
                    _logger.warning("Tool step failed: %s", step_data.get("error", "unknown"))

            merged = await self._response_composer.merge(goal, step_results)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=any(r.success for r in step_results),
                message="",
                data={
                    "status": "completed" if any(r.success for r in step_results) else "error",
                    "response": merged,
                    "tool_name": executed_tool,
                    "tool_output": raw_output,
                },
            )

        plan_result = await self._planner_agent.handle(
            AgentTask(task_type="plan", payload={"goal": goal})
        )
        raw_output = (plan_result.data or {}).get("response", "")
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=plan_result.success,
            message=plan_result.message,
            data={
                **(plan_result.data or {}),
                "tool_name": tool_name,
                "tool_output": raw_output,
            },
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, object]:
        base = await super().health_check()
        base["conversation_turns"] = self._conversation_manager.turn_count
        base["has_workflow_engine"] = self._workflow_engine is not None
        return base
