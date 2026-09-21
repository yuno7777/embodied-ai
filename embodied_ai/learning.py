"""Small tabular RL baseline operating exclusively through public observations."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .schemas import ActionRequest
from .action_candidates import public_action_candidates


def _episode_result(seed: int, step: int, total_reward: float, terminal_reason: str | None, info: dict[str, Any], control_elapsed_ms: float, reset_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Preserve evaluator-only metrics beside an episode without policy exposure."""
    metrics = info.get("evaluator", {}).get("metrics", {}) if isinstance(info.get("evaluator"), dict) else {}
    result = {
        "seed": seed, "steps": step, "total_reward": total_reward, "terminal_reason": terminal_reason,
        "invalid_actions": metrics.get("invalid_actions"),
        "exploration_coverage": metrics.get("exploration_coverage"),
        "resource_efficiency": metrics.get("resource_efficiency"),
        "control_elapsed_ms": round(control_elapsed_ms, 3),
    }
    if isinstance(reset_info, dict):
        for field in ("run_id", "scenario_id", "world_manifest", "reward_config", "observation_mode", "distribution"):
            if field in reset_info:
                result[field] = reset_info[field]
    return result


@dataclass(frozen=True)
class TabularQConfig:
    learning_rate: float = 0.2
    discount: float = 0.95
    epsilon: float = 0.2

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate <= 1 or not 0 <= self.discount <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("learning_rate, discount, and epsilon must be probabilities")


class TabularQPolicy:
    """A checkpointable Q-learning policy over compact, non-privileged observations.

    Object IDs are proposed only when a public observation makes the interaction
    geometrically valid; inventory actions use only public carried-item IDs.
    More capable policies can use the same environment contract with richer
    action proposal mechanisms.
    """

    name = "tabular_q"
    _actions = (
        ActionRequest(type="move", direction="north"),
        ActionRequest(type="move", direction="south"),
        ActionRequest(type="move", direction="east"),
        ActionRequest(type="move", direction="west"),
        ActionRequest(type="inspect"),
        ActionRequest(type="wait"),
    )
    _open_index = len(_actions)
    _pickup_index = len(_actions) + 1
    _use_item_index = len(_actions) + 2

    def __init__(self, config: TabularQConfig | None = None, seed: int = 0):
        self.config = config or TabularQConfig()
        self.q_values: dict[str, list[float]] = {}
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.random = random.Random(seed)

    @staticmethod
    def observation_key(observation: dict[str, Any]) -> str:
        agent = observation.get("agent", {})
        cells = []
        for cell in observation.get("visible_cells", []):
            relative = cell.get("relative_position", {})
            entities = sorted(
                (entity.get("type"), entity.get("state"))
                for entity in cell.get("entities", [])
                if isinstance(entity, dict)
            )
            cells.append((relative.get("x"), relative.get("y"), cell.get("terrain"), entities))
        compact = {
            "health": int(agent.get("health", 0)) // 25,
            "energy": int(agent.get("energy", 0)) // 25,
            "hydration": int(agent.get("hydration", 0)) // 25,
            "inventory": sorted(str(item) for item in agent.get("inventory", [])),
            "cells": sorted(cells),
        }
        return json.dumps(compact, sort_keys=True, separators=(",", ":"))

    def action_index(self, action: ActionRequest) -> int:
        if action.type == "open" and action.target_id:
            return self._open_index
        if action.type == "pickup" and action.item_id:
            return self._pickup_index
        if action.type == "use_item" and action.item_id:
            return self._use_item_index
        try:
            return self._actions.index(action)
        except ValueError as error:
            raise ValueError("action is outside the tabular baseline action set") from error

    @property
    def _action_count(self) -> int:
        return len(self._actions) + 3

    def _values(self, state: str, *, create: bool = True) -> list[float]:
        values = self.q_values.get(state)
        if values is None:
            return [0.0] * self._action_count if not create else self.q_values.setdefault(state, [0.0] * self._action_count)
        if not create:
            return values
        values.extend([0.0] * (self._action_count - len(values)))
        return values

    def _candidate_actions(self, observation: dict[str, Any]) -> list[tuple[int, ActionRequest]]:
        return [(self.action_index(action), action) for action in public_action_candidates(observation)]

    @staticmethod
    def _value_at(values: list[float], index: int) -> float:
        """Read older, short checkpoint vectors without modifying them."""
        return values[index] if index < len(values) else 0.0

    def _available_actions(self, observation: dict[str, Any]) -> list[tuple[int, ActionRequest]]:
        state = self.observation_key(observation)
        allowed = set(observation.get("allowed_action_types", []))
        return [(index, action) for index, action in self._candidate_actions(observation) if action.type in allowed]

    def _greedy_action(self, observation: dict[str, Any]) -> ActionRequest:
        state = self.observation_key(observation)
        available = self._available_actions(observation)
        if not available:
            return ActionRequest(type="wait")
        values = self._values(state, create=False)
        best = max(self._value_at(values, index) for index, _ in available)
        return next(action for index, action in available if self._value_at(values, index) == best)

    def act(self, observation: dict[str, Any]) -> ActionRequest:
        available = self._available_actions(observation)
        if not available:
            return ActionRequest(type="wait")
        if self.random.random() < self.config.epsilon:
            return self.random.choice(available)[1]
        return self._greedy_action(observation)

    def update(self, observation: dict[str, Any], action: ActionRequest, reward: float, next_observation: dict[str, Any], terminated: bool) -> None:
        state, next_state = self.observation_key(observation), self.observation_key(next_observation)
        index = self.action_index(action)
        values = self._values(state)
        next_available = self._available_actions(next_observation)
        next_values = self._values(next_state)
        bootstrap = max((self._value_at(next_values, next_index) for next_index, _ in next_available), default=0.0)
        target = reward if terminated else reward + self.config.discount * bootstrap
        values[index] += self.config.learning_rate * (target - values[index])

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "config": asdict(self.config), "q_values": self.q_values}, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path, seed: int = 0) -> "TabularQPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("q_values"), dict):
            raise ValueError("unsupported tabular Q checkpoint")
        policy = cls(TabularQConfig(**payload["config"]), seed)
        policy.q_values = {str(state): [float(value) for value in values] for state, values in payload["q_values"].items()}
        return policy


def train_tabular_q(
    environment_factory: Callable[[], Any], policy: TabularQPolicy, seeds: list[int], max_steps: int = 256,
    reset_options_for_seed: Callable[[int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run online Q-learning episodes against an RL-style authoritative environment."""
    if not seeds or max_steps < 1:
        raise ValueError("training requires seeds and a positive max_steps")
    episodes = []
    with environment_factory() as environment:
        for seed in seeds:
            policy.reset(seed)
            if reset_options_for_seed:
                observation, reset_info = environment.reset(seed=seed, options=reset_options_for_seed(seed))
            else:
                observation, reset_info = environment.reset(seed=seed)
            total_reward = 0.0
            last_info: dict[str, Any] = {}
            started = time.perf_counter()
            for step in range(1, max_steps + 1):
                action = policy.act(observation)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                last_info = info
                done = terminated or truncated
                # Gymnasium semantics: a truncation is a boundary imposed by
                # the runner, not a terminal state of the world, so Q-learning
                # must retain its bootstrap target.
                policy.update(observation, action, reward, next_observation, terminated=terminated)
                observation, total_reward = next_observation, total_reward + reward
                if done:
                    episodes.append(_episode_result(seed, step, total_reward, info.get("terminal_reason"), info, (time.perf_counter() - started) * 1000, reset_info))
                    break
            else:
                episodes.append(_episode_result(seed, max_steps, total_reward, "trainer_step_limit", last_info, (time.perf_counter() - started) * 1000, reset_info))
    return episodes


def evaluate_tabular_q(
    environment_factory: Callable[[], Any], policy: TabularQPolicy, seeds: list[int], max_steps: int = 256,
    reset_options_for_seed: Callable[[int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a frozen greedy checkpoint without mutating its Q-values."""
    if not seeds or max_steps < 1:
        raise ValueError("evaluation requires seeds and a positive max_steps")
    episodes = []
    with environment_factory() as environment:
        for seed in seeds:
                if reset_options_for_seed:
                    observation, reset_info = environment.reset(seed=seed, options=reset_options_for_seed(seed))
                else:
                    observation, reset_info = environment.reset(seed=seed)
                total_reward = 0.0
                last_info: dict[str, Any] = {}
                started = time.perf_counter()
                for step in range(1, max_steps + 1):
                    next_observation, reward, terminated, truncated, info = environment.step(policy._greedy_action(observation))
                    last_info = info
                    observation, total_reward = next_observation, total_reward + reward
                    if terminated or truncated:
                        episodes.append(_episode_result(seed, step, total_reward, info.get("terminal_reason"), info, (time.perf_counter() - started) * 1000, reset_info))
                        break
                else:
                    episodes.append(_episode_result(seed, max_steps, total_reward, "evaluator_step_limit", last_info, (time.perf_counter() - started) * 1000, reset_info))
    return episodes


def evaluate_tabular_partitions(
    environment_factory: Callable[[], Any], policy: TabularQPolicy, partitions: dict[str, list[int]], max_steps: int = 256,
    reset_options_for_partition: Callable[[str, int], dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate one frozen checkpoint on explicit, disjoint seed distributions."""
    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("partitions must contain train, validation, and test")
    seeds = [seed for partition in partitions.values() for seed in partition]
    if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("partition seeds must be non-negative, non-empty, and disjoint")
    return {
        name: evaluate_tabular_q(
            environment_factory, policy, partition_seeds, max_steps,
            (lambda seed, partition=name: reset_options_for_partition(partition, seed)) if reset_options_for_partition else None,
        )
        for name, partition_seeds in sorted(partitions.items())
    }
