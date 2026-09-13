"""Headless RL-style adapter over the authoritative Rust simulation service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .runner import RustRunClient
from .schemas import ActionRequest


@dataclass(frozen=True)
class EmbodiedEnvConfig:
    """Configuration identifying an authoritative world distribution member."""

    server_url: str = "http://127.0.0.1:8080"
    scenario_id: str | None = None
    generated_world: dict[str, Any] | None = None
    distribution: Literal["train", "validation", "test"] | None = None
    world_partition: Literal["train", "validation", "test"] | None = None
    reward_config: dict[str, int] | None = None
    observation_mode: str = "normal"
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.distribution is not None and self.world_partition is not None and self.distribution != self.world_partition:
            raise ValueError("distribution conflicts with world_partition")


class EmbodiedEnv:
    """A single independently controlled Rust episode.

    This deliberately follows Gymnasium's ``reset``/``step`` return semantics
    without making Gymnasium a dependency. ``info['evaluator']`` is privileged
    evaluation output and must not be passed into a policy as perception.
    """

    def __init__(
        self,
        config: EmbodiedEnvConfig | None = None,
        *,
        distribution: Literal["train", "validation", "test"] | None = None,
        observation_mode: str | None = None,
        server_url: str | None = None,
    ):
        """Create an explicit config or use the concise research-loop keywords."""
        if config is not None and any(value is not None for value in (distribution, observation_mode, server_url)):
            raise ValueError("pass either config or direct environment keywords")
        self.config = config or EmbodiedEnvConfig(
            distribution=distribution,
            observation_mode=observation_mode or "normal",
            server_url=server_url or "http://127.0.0.1:8080",
        )
        self._client = RustRunClient(self.config.server_url)
        self._run_id: str | None = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def reset(self, *, seed: int = 42, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        options = options or {}
        scenario_id = options.get("scenario_id", self.config.scenario_id)
        generated_world = options.get("generated_world", self.config.generated_world)
        distribution = options.get("distribution", self.config.distribution)
        legacy_partition = options.get("world_partition", self.config.world_partition)
        if distribution is not None and legacy_partition is not None and distribution != legacy_partition:
            raise ValueError("distribution conflicts with world_partition")
        world_partition = distribution or legacy_partition
        reward_config = options.get("reward_config", self.config.reward_config)
        observation_mode = options.get("observation_mode", self.config.observation_mode)
        max_steps = options.get("max_steps", self.config.max_steps)
        if world_partition is not None:
            if scenario_id is not None:
                raise ValueError("world_partition requires a generated world, not a catalog scenario")
            generated_world = dict(generated_world or {"seed": seed})
            existing_partition = generated_world.get("partition")
            if existing_partition is not None and existing_partition != world_partition:
                raise ValueError("generated_world partition conflicts with world_partition")
            generated_world["partition"] = world_partition
        created = self._client.create(seed, max_steps, observation_mode, scenario_id, generated_world, reward_config)
        self._run_id = created["run_id"]
        observation = created["observation"]
        return observation, {
            "run_id": self._run_id,
            "scenario_id": scenario_id,
            "world_manifest": created.get("world_manifest"),
            "reward_config": created.get("reward_config"),
            "seed": seed,
            "observation_mode": observation_mode,
            "distribution": world_partition,
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
