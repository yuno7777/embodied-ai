"""Headless RL-style adapter over the authoritative Rust simulation service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runner import RustRunClient
from .schemas import ActionRequest


@dataclass(frozen=True)
class EmbodiedEnvConfig:
    """Configuration identifying an authoritative world distribution member."""

    server_url: str = "http://127.0.0.1:8080"
    scenario_id: str | None = None
    generated_world: dict[str, Any] | None = None
    observation_mode: str = "normal"
    max_steps: int | None = None


class EmbodiedEnv:
    """A single independently controlled Rust episode.

    This deliberately follows Gymnasium's ``reset``/``step`` return semantics
    without making Gymnasium a dependency. ``info['evaluator']`` is privileged
    evaluation output and must not be passed into a policy as perception.
    """

    def __init__(self, config: EmbodiedEnvConfig | None = None):
        self.config = config or EmbodiedEnvConfig()
        self._client = RustRunClient(self.config.server_url)
        self._run_id: str | None = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def reset(self, *, seed: int = 42, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        options = options or {}
        scenario_id = options.get("scenario_id", self.config.scenario_id)
        generated_world = options.get("generated_world", self.config.generated_world)
        observation_mode = options.get("observation_mode", self.config.observation_mode)
        max_steps = options.get("max_steps", self.config.max_steps)
        created = self._client.create(seed, max_steps, observation_mode, scenario_id, generated_world)
        self._run_id = created["run_id"]
        observation = created["observation"]
        return observation, {
            "run_id": self._run_id,
            "scenario_id": scenario_id,
            "world_manifest": created.get("world_manifest"),
            "seed": seed,
            "observation_mode": observation_mode,
        }

    def step(self, action: ActionRequest | dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._run_id is None:
            raise RuntimeError("reset must be called before step")
        request = action if isinstance(action, ActionRequest) else ActionRequest.model_validate(action)
        result = self._client.step(self._run_id, request)
        terminal_reason = result.get("terminal_reason")
        done = bool(result.get("done"))
        truncated = done and terminal_reason in {"timeout", "time_limit", "client_timeout", "token_budget_exhausted"}
        terminated = done and not truncated
        invalid = any(event.get("type") == "InvalidAction" for event in result.get("events", []))
        return result["observation"], float(result["reward"]), terminated, truncated, {
            "run_id": self._run_id,
            "action_valid": not invalid,
            "terminal_reason": terminal_reason,
            "events": result.get("events", []),
            "evaluator": {
                "reward_breakdown": result.get("reward_breakdown", {}),
                "metrics": result.get("metrics", {}),
                "simulation_time": result.get("simulation_time"),
            },
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EmbodiedEnv":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
