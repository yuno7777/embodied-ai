"""Provider boundary: a model can choose an action, never mutate an environment."""
from __future__ import annotations
import asyncio, json, os, time
from collections.abc import Callable
from typing import Any, Protocol
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from .context import AgentContext, SYSTEM_PROMPT
from .schemas import AgentDecision, ActionRequest

class Policy(Protocol):
    name: str
    def reset(self, seed: int | None = None) -> None: ...
    def act(self, observation: dict) -> ActionRequest: ...


class DecisionPolicy(Protocol):
    """Optional synchronous policy contract with bounded decision metadata."""
    name: str
    def reset(self, seed: int | None = None) -> None: ...
    def decide(self, observation: dict) -> AgentDecision: ...


class AgentProvider(Policy, Protocol):
    """Compatibility name for policies used by the existing runner."""
    def choose_action(self, observation: dict) -> ActionRequest: ...


class CallablePolicy:
    """Adapter for custom public-observation policies without simulation access."""

    def __init__(self, name: str, action_fn: Callable[[dict[str, Any]], AgentDecision | ActionRequest | dict[str, Any]], reset_fn: Callable[[int | None], None] | None = None):
        if not name:
            raise ValueError("policy name must be non-empty")
        self.name, self._action_fn, self._reset_fn = name, action_fn, reset_fn

    def reset(self, seed: int | None = None) -> None:
        if self._reset_fn is not None:
            self._reset_fn(seed)

    def act(self, observation: dict[str, Any]) -> ActionRequest:
        return self.decide(observation).action

    def decide(self, observation: dict[str, Any]) -> AgentDecision:
        result = self._action_fn(observation)
        if isinstance(result, AgentDecision):
            return result
        if isinstance(result, dict) and "action" in result:
            return AgentDecision.model_validate(result)
        action = result if isinstance(result, ActionRequest) else ActionRequest.model_validate(result)
        return AgentDecision(action=action, decision_summary="custom policy action submitted to Rust authority")

class ScriptedProvider:
    name = "scripted"
    """Deterministic success route; safe for tests and repeatable benchmarks."""
    route=[("move","east"),("move","east"),("move","east"),("move","north"),("move","north"),("open",None),("pickup",None),("move","south"),("move","south"),("move","south"),("move","east"),("move","east"),("open",None),("move","east"),("move","east"),("move","east"),("move","south"),("move","east"),("move","east"),("move","east"),("move","east"),("move","north"),("move","north"),("open",None),("move","east")]
    def __init__(self): self.reset()
    def reset(self, seed: int | None = None) -> None: self.index=0
    def act(self, observation: dict) -> ActionRequest: return self.choose_action(observation)
    def choose_action(self, observation: dict) -> ActionRequest:
        kind,direction=self.route[min(self.index,len(self.route)-1)]; self.index+=1
        target = "key_locker" if kind=="open" and self.index==6 else ("service_door" if kind=="open" and self.index==13 else ("exit_door" if kind=="open" else None))
        return ActionRequest(type=kind,direction=direction,target_id=target)

class RandomValidProvider:
    name="random_valid"
    def __init__(self, seed: int = 0): self.seed=seed; self.reset(seed)
    def reset(self, seed: int | None = None) -> None:
        import random
        self.seed = self.seed if seed is None else seed
        self.random=random.Random(self.seed)
    def act(self, observation: dict) -> ActionRequest: return self.choose_action(observation)
    def choose_action(self, observation: dict) -> ActionRequest:
        return ActionRequest(type="move", direction=self.random.choice(["north","south","east","west"]))

class ExplorerProvider:
    """Deterministic coverage baseline alternating local inspection and movement."""
    name = "explorer"
    directions = ("east", "north", "west", "south")
    def __init__(self): self.reset()
    def reset(self, seed: int | None = None) -> None: self.turn = 0
    def act(self, observation: dict) -> ActionRequest: return self.choose_action(observation)
    def choose_action(self, observation: dict) -> ActionRequest:
        self.turn += 1
        if self.turn % 5 == 1:
            return ActionRequest(type="inspect")
        return ActionRequest(type="move", direction=self.directions[(self.turn - 2) % len(self.directions)])

class CautiousProvider:
    """Deterministic baseline that avoids observed adjacent active hazards and walls."""
    name = "cautious"
    directions = (("east", (1, 0)), ("south", (0, 1)), ("north", (0, -1)), ("west", (-1, 0)))
    def __init__(self): self.reset()
    def reset(self, seed: int | None = None) -> None: self.turn = 0
    def act(self, observation: dict) -> ActionRequest: return self.choose_action(observation)
    def choose_action(self, observation: dict) -> ActionRequest:
        self.turn += 1
        if self.turn == 1:
            return ActionRequest(type="inspect")
        cells = {(cell.get("relative_position", {}).get("x"), cell.get("relative_position", {}).get("y")): cell for cell in observation.get("visible_cells", [])}
        for direction, relative in self.directions:
            cell = cells.get(relative)
            if not cell or cell.get("terrain") == "wall":
                continue
            unsafe = any(entity.get("type") == "hazard" and entity.get("state") != "inactive" for entity in cell.get("entities", []))
            blocked = any(entity.get("type") == "door" and entity.get("state") != "open" for entity in cell.get("entities", []))
            if not unsafe and not blocked:
                return ActionRequest(type="move", direction=direction)
        return ActionRequest(type="wait")

class MockReasoningProvider(ScriptedProvider):
    name="mock_reasoning"
    def decide(self, observation: dict) -> AgentDecision:
        action=self.choose_action(observation); return AgentDecision(action=action, decision_summary="Following a deterministic safe test policy.")
    decision = decide

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
    def reset(self, seed: int | None = None) -> None:
        self.last_token_usage = None
        self.last_attempts = 0
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
