"""Personality Engine — handles greetings, humor, sarcasm, compliments, small talk.

Provides natural, friendly responses that resemble a modern conversational AI
rather than a robotic command-line interface.
"""

from __future__ import annotations

import logging
import random
import re

_logger = logging.getLogger(__name__)

_GREETINGS = [
    "Hello! How can I help you today?",
    "Hey there! What can I do for you?",
    "Hi! Ready to assist you.",
    "Greetings! How may I help?",
    "Hello! What's on your mind?",
]

_MORNING_GREETINGS = [
    "Good morning! Hope you're having a great start to your day.",
    "Good morning! How can I brighten your day?",
    "Morning! Ready to help.",
]

_AFTERNOON_GREETINGS = [
    "Good afternoon! How can I assist you?",
    "Good afternoon! What are we working on?",
]

_EVENING_GREETINGS = [
    "Good evening! How can I help you this evening?",
    "Good evening! What can I do for you?",
]

_THANKS_RESPONSES = [
    "You're welcome! Let me know if you need anything else.",
    "Happy to help! Anything else?",
    "Anytime! That's what I'm here for.",
    "Glad I could help!",
    "My pleasure!",
]

_GOODBYE_RESPONSES = [
    "Goodbye! Have a great day!",
    "See you later! Take care.",
    "Bye! Come back anytime.",
    "Until next time!",
]

_HOW_ARE_YOU_RESPONSES = [
    "I'm doing great, thanks for asking! How can I help you?",
    "All systems operational! What can I do for you?",
    "Doing well! Ready to assist. What do you need?",
]

_JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why did the AI break up with the database? Too many relationships.",
    "What's a computer's favorite snack? Microchips!",
    "Why was the JavaScript developer sad? They didn't know how to 'null' their feelings.",
    "How many programmers does it take to change a light bulb? None — that's a hardware problem.",
    "Why do Python programmers prefer snakes? Because they hate spiders (web bugs).",
    "I told my computer I needed a break. Now it won't stop sending me vacation ads.",
    "What did the router say to the doctor? 'I need a bandwidth-aid.'",
]

_COMPLIMENTS = [
    "Thank you! I try my best.",
    "That's kind of you to say!",
    "I appreciate that!",
    "Thanks! I'm here to help.",
]

_INSULTS = [
    "I understand you're frustrated. Let me know how I can help.",
    "I'm sorry you feel that way. I'm here to help if you need anything.",
    "Let's focus on solving your problem. What can I do for you?",
    "I respect your opinion. Is there something specific I can help with?",
]

_SARCASM_DETECTORS = [
    re.compile(r"\b(sure|obviously|clearly|totally)\s+(janvi|jarvis|ai|bot)", re.IGNORECASE),
    re.compile(r"\b(oh\s+really|no\s+way|you\s+don't\s+say)\b", re.IGNORECASE),
    re.compile(r"\bgreat\.?\s*(just\s+what\s+I\s+needed|another\s+problem)", re.IGNORECASE),
    re.compile(r"\b(thanks?\s+(a\s+)?lot|thanks?\s+for\s+nothing)\b", re.IGNORECASE),
    re.compile(r"\b(yeah|yep|sure)\s+(right|janvi|jarvis)\b", re.IGNORECASE),
]

_SARCASM_RESPONSES = [
    "I sense some sarcasm there! 😄 But I'm here to help regardless.",
    "Is that sarcasm I detect? 😄 I'm not perfect, but I'm trying!",
    "I'll take that as constructive feedback! 😄 What can I actually help with?",
    "I know, I know — I'm a work in progress! 😄 How can I assist?",
]

_WAKE_UP_PATTERNS = [
    re.compile(r"\bwake\s+up\b", re.IGNORECASE),
    re.compile(r"\b(hello|hey|hi)\s+(jarvis|janvi|assistant)\b", re.IGNORECASE),
]

_WAKE_UP_RESPONSES = [
    "I'm online and ready! 😄 How can I help?",
    "Good to see you! I'm ready when you are.",
    "Awake and alert! What's up?",
    "Online and operational! What do you need?",
]


class PersonalityEngine:
    """Handles conversational personality — greetings, humor, sarcasm, small talk.

    Detects conversational patterns and returns appropriate natural responses.
    Falls through to None if the input requires actual task processing.
    """

    def __init__(self) -> None:
        self._use_emojis = True

    def process(self, text: str) -> str | None:
        """Process *text* for personality patterns.

        Returns a response string if the input is purely conversational,
        or None if it should be passed through to the intent engine.
        """
        stripped = text.strip()
        lower = stripped.lower().rstrip("?!.,;: ")

        if not lower:
            return None

        response = self._check_wake_up(lower)
        if response:
            return response

        response = self._check_sarcasm(lower)
        if response:
            return response

        if self._is_insult(lower):
            return random.choice(_INSULTS)

        if self._is_joke_request(lower):
            return random.choice(_JOKES)

        if self._is_compliment(lower):
            return random.choice(_COMPLIMENTS)

        if self._is_greeting(lower):
            return self._greeting_response(lower)

        if self._is_thanks(lower):
            return random.choice(_THANKS_RESPONSES)

        if self._is_goodbye(lower):
            return random.choice(_GOODBYE_RESPONSES)

        if self._is_how_are_you(lower):
            return random.choice(_HOW_ARE_YOU_RESPONSES)

        return None

    def _check_wake_up(self, lower: str) -> str | None:
        for pattern in _WAKE_UP_PATTERNS:
            if pattern.search(lower):
                return random.choice(_WAKE_UP_RESPONSES)
        return None

    def _check_sarcasm(self, lower: str) -> str | None:
        for pattern in _SARCASM_DETECTORS:
            if pattern.search(lower):
                return random.choice(_SARCASM_RESPONSES)
        return None

    @staticmethod
    def _is_insult(lower: str) -> bool:
        insults = {
            "useless", "stupid", "dumb", "idiot", "moron", "waste",
            "terrible", "awful", "horrible", "incompetent", "pathetic",
            "piece of trash", "garbage", "sucks", "worst",
        }
        return any(w in lower for w in insults) and len(lower.split()) <= 8

    @staticmethod
    def _is_joke_request(lower: str) -> bool:
        patterns = [
            r"\btell\s+me\s+a\s+(joke|funny\s+story)\b",
            r"\bmake\s+me\s+laugh\b",
            r"\bsay\s+something\s+funny\b",
            r"\bjoke\b",
            r"\bhumor\s+me\b",
        ]
        return any(re.search(p, lower) for p in patterns)

    @staticmethod
    def _is_compliment(lower: str) -> bool:
        compliments = {
            "great", "awesome", "amazing", "fantastic", "excellent",
            "wonderful", "brilliant", "smart", "intelligent", "helpful",
            "good job", "nice work", "well done", "perfect", "love it",
            "you're the best", " you rock",
        }
        return any(c in lower for c in compliments) and len(lower.split()) <= 10

    @staticmethod
    def _is_greeting(lower: str) -> bool:
        greetings = {
            "hi", "hello", "hey", "greetings", "howdy",
            "good morning", "good afternoon", "good evening", "good day",
            "hi there", "hello there", "hey there",
            "morning", "afternoon", "evening",
        }
        exact = lower.rstrip("?!.,;:")
        return exact in greetings or any(
            exact.startswith(g) for g in ("hi ", "hello ", "hey ", "good morning", "good afternoon", "good evening")
        )

    @staticmethod
    def _is_thanks(lower: str) -> bool:
        thanks = {
            "thanks", "thank you", "thankyou", "cheers", "thx",
            "appreciate it", "much appreciated", "thanks jarvis",
            "thank you jarvis", "thanks janvi",
        }
        return lower.rstrip("?!.,;:") in thanks

    @staticmethod
    def _is_goodbye(lower: str) -> bool:
        goodbyes = {
            "bye", "goodbye", "good bye", "see you", "see ya",
            "cya", "later", "ttyl", "bye bye", "take care",
            "see you later", "have a good day", "good night",
        }
        return lower.rstrip("?!.,;:") in goodbyes

    @staticmethod
    def _is_how_are_you(lower: str) -> bool:
        patterns = [
            "how are you", "how's it going", "how are you doing",
            "how do you do", "how's your day", "how was your day",
            "how is it going", "how are you today",
            "you good", "you alright",
        ]
        return lower.rstrip("?!.,;:") in patterns or any(
            lower.startswith(p) for p in patterns
        )

    def _greeting_response(self, lower: str) -> str:
        if any(w in lower for w in ("morning",)):
            return random.choice(_MORNING_GREETINGS)
        if any(w in lower for w in ("afternoon",)):
            return random.choice(_AFTERNOON_GREETINGS)
        if any(w in lower for w in ("evening",)):
            return random.choice(_EVENING_GREETINGS)
        return random.choice(_GREETINGS)
