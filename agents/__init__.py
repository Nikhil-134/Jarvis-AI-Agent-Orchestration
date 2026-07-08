"""Agent package exports."""

from agents.athena_agent import AthenaAgent
from agents.base import Agent
from agents.browser_agent import BrowserAgent
from agents.calendar_agent import CalendarAgent
from agents.contracts import AgentMessage, AgentResult, AgentTask
from agents.context_manager import ContextManager
from agents.conversation_manager import ConversationManager
from agents.desktop_agent import DesktopAgent
from agents.email_agent import EmailAgent
from agents.friday_agent import FridayAgent
from agents.gecko_agent import GeckoAgent
from agents.hercules_agent import HerculesAgent
from agents.hulk_agent import HulkAgent
from agents.interfaces import IAgent
from agents.jarvis_agent import JarvisPrimeAgent
from agents.jerome_agent import JeromeAgent
from agents.memory_agent import MemoryAgent
from agents.notes_agent import NotesAgent
from agents.oracle_agent import OracleAgent
from agents.pepper_agent import PepperAgent
from agents.planner import PlannerAgent
from agents.reminder_agent import ReminderAgent
from agents.response_composer import ResponseComposer
from agents.stark_agent import StarkAgent
from agents.steve_agent import SteveAgent
from agents.tool_agent import ToolAgent
from agents.ultron_agent import UltronAgent
from agents.veronica_agent import VeronicaAgent
from agents.vision_agent import VisionAgent
from agents.voice_agent import VoiceAgent

__all__ = [
    "Agent",
    "AgentMessage",
    "AgentResult",
    "AgentTask",
    "AthenaAgent",
    "BrowserAgent",
    "CalendarAgent",
    "ContextManager",
    "ConversationManager",
    "DesktopAgent",
    "EmailAgent",
    "FridayAgent",
    "GeckoAgent",
    "HerculesAgent",
    "HulkAgent",
    "IAgent",
    "JarvisPrimeAgent",
    "JeromeAgent",
    "MemoryAgent",
    "NotesAgent",
    "OracleAgent",
    "PepperAgent",
    "PlannerAgent",
    "ReminderAgent",
    "ResponseComposer",
    "StarkAgent",
    "SteveAgent",
    "ToolAgent",
    "UltronAgent",
    "VeronicaAgent",
    "VisionAgent",
    "VoiceAgent",
]
