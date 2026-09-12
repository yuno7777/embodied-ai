"""Provider boundary: a model can choose an action, never mutate an environment."""
from __future__ import annotations
import asyncio, json, os, time
from typing import Protocol
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from .context import AgentContext, SYSTEM_PROMPT
from .engine import Action
from .schemas import AgentDecision, ActionRequest

class AgentProvider(Protocol):
    name: str
    def choose_action(self, observation: dict) -> Action: ...

class ScriptedProvider:
    name = "scripted"
    """Deterministic success route; safe for tests and repeatable benchmarks."""
    route=[("move","east"),("move","east"),("move","east"),("move","north"),("move","north"),("open",None),("pickup",None),("move","south"),("move","south"),("move","south"),("move","east"),("move","east"),("open",None),("move","east"),("move","east"),("move","east"),("move","south"),("move","east"),("move","east"),("move","east"),("move","east"),("move","north"),("move","north"),("open",None),("move","east")]
    def __init__(self): self.index=0
    def choose_action(self, observation: dict) -> Action:
        kind,direction=self.route[min(self.index,len(self.route)-1)]; self.index+=1
        target = "key_locker" if kind=="open" and self.index==6 else ("service_door" if kind=="open" and self.index==13 else ("exit_door" if kind=="open" else None))
        return Action(type=kind,direction=direction,target_id=target)

class RandomValidProvider:
    name="random_valid"
    def __init__(self, seed: int = 0): import random; self.random=random.Random(seed)
    def choose_action(self, observation: dict) -> Action:
        return Action(type="move", direction=self.random.choice(["north","south","east","west"]))

class ExplorerProvider:
    """Deterministic coverage baseline alternating local inspection and movement."""
    name = "explorer"
    directions = ("east", "north", "west", "south")
    def __init__(self): self.turn = 0
    def choose_action(self, observation: dict) -> Action:
        self.turn += 1
        if self.turn % 5 == 1:
            return Action(type="inspect")
        return Action(type="move", direction=self.directions[(self.turn - 2) % len(self.directions)])

class CautiousProvider:
    """Deterministic baseline that avoids observed adjacent active hazards and walls."""
    name = "cautious"
    directions = (("east", (1, 0)), ("south", (0, 1)), ("north", (0, -1)), ("west", (-1, 0)))
    def __init__(self): self.turn = 0
    def choose_action(self, observation: dict) -> Action:
        self.turn += 1
        if self.turn == 1:
            return Action(type="inspect")
        cells = {(cell.get("relative_position", {}).get("x"), cell.get("relative_position", {}).get("y")): cell for cell in observation.get("visible_cells", [])}
        for direction, relative in self.directions:
            cell = cells.get(relative)
            if not cell or cell.get("terrain") == "wall":
                continue
            unsafe = any(entity.get("type") == "hazard" and entity.get("state") != "inactive" for entity in cell.get("entities", []))
            blocked = any(entity.get("type") == "door" and entity.get("state") != "open" for entity in cell.get("entities", []))
            if not unsafe and not blocked:
                return Action(type="move", direction=direction)
        return Action(type="wait")

class MockReasoningProvider(ScriptedProvider):
    name="mock_reasoning"
    def decision(self, observation: dict) -> AgentDecision:
        action=self.choose_action(observation); return AgentDecision(action=ActionRequest(**action.__dict__), decision_summary="Following a deterministic safe test policy.")

class GeminiProvider:
    """Opt-in official SDK adapter. Unit tests never instantiate it with a live key."""
    name="gemini"
    def __init__(self, model: str | None = None, timeout_seconds: float = 30):
        self.model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"); self.timeout_seconds=timeout_seconds
        self.last_token_usage: dict[str, int] | None = None
        self.last_attempts = 0
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("ProviderUnavailable: GEMINI_API_KEY is not configured")
        from google import genai
        self.client=genai.Client(api_key=key)
    async def choose(self, observation: dict, context: AgentContext) -> tuple[AgentDecision, float]:
        schema=AgentDecision.model_json_schema(); prompt=json.dumps({"observation":observation,"memory":context.payload()})
        self.last_token_usage = None
        self.last_attempts = 0
        started=time.perf_counter()
        self.last_backoff_ms = []
        def record_backoff(retry_state):
            self.last_backoff_ms.append(round(retry_state.next_action.sleep * 1000))
        def request(): return self.client.models.generate_content(model=self.model, contents=prompt, config={"system_instruction":SYSTEM_PROMPT,"response_mime_type":"application/json","response_json_schema":schema})
        try:
            async for attempt in AsyncRetrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.25, min=0.25, max=2), retry=retry_if_exception_type((TimeoutError, ConnectionError)), before_sleep=record_backoff, reraise=True):
                with attempt:
                    self.last_attempts = attempt.retry_state.attempt_number
                    response=await asyncio.wait_for(asyncio.to_thread(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as error:
            raise RuntimeError("ProviderTimeout: Gemini request exceeded its deadline") from error
        except (TimeoutError, ConnectionError) as error:
            raise RuntimeError("ProviderUnavailable: Gemini request failed after retries") from error
        try: decision=AgentDecision.model_validate_json(response.text)
        except Exception as error: raise RuntimeError(f"ProviderMalformedResponse: {error}") from error
        usage = getattr(response, "usage_metadata", None)
        fields = {
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "cached_tokens": getattr(usage, "cached_content_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }
        recorded = {name: value for name, value in fields.items() if isinstance(value, int)}
        self.last_token_usage = recorded or None
        return decision, (time.perf_counter()-started)*1000
