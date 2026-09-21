"""Compact, explicitly non-authoritative context for an agent provider."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

SYSTEM_PROMPT = """You inhabit a simulated body. You only know what appears in the observation and memory below. Choose exactly one permitted structured action. You cannot alter reality directly, invent objects, execute tools, or use hidden knowledge. Preserve survival, learn from failed actions, and provide only a concise decision summary, never private reasoning."""

@dataclass
class AgentContext:
    memory_mode: str = "recent"
    observations: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    facts: set[str] = field(default_factory=set)
    memory_window: int = 5
    def __post_init__(self) -> None:
        if type(self.memory_window) is not int or not 1 <= self.memory_window <= 100:
            raise ValueError("memory_window must be an integer between 1 and 100")
    def record(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        events: list[dict[str, Any]],
        perceived_messages: list[str] | None = None,
    ) -> None:
        """Retain facts only when the sensor exposed the event message."""
        if self.memory_mode == "none": return
        self.observations.append(observation); self.actions.append(action)
        self.observations = self.observations[-self.memory_window:]; self.actions = self.actions[-self.memory_window:]
        allowed_messages = set(perceived_messages or [])
        self.facts.update(
            event.get("message", "")
            for event in events
            if event.get("type") in {"NpcSpoke", "DoorOpened", "ItemPickedUp"}
            and event.get("message") in allowed_messages
        )
    def payload(self) -> dict[str, Any]:
        return {"recent_actions": self.actions[-self.memory_window:], "known_facts": sorted(self.facts)[-12:]} if self.memory_mode == "recent" else {}
