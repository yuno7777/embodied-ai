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


def _episode_result(seed: int, step: int, total_reward: float, terminal_reason: str | None, info: dict[str, Any], control_elapsed_ms: float) -> dict[str, Any]:
    """Preserve evaluator-only metrics beside an episode without policy exposure."""
    metrics = info.get("evaluator", {}).get("metrics", {}) if isinstance(info.get("evaluator"), dict) else {}
    return {
        "seed": seed, "steps": step, "total_reward": total_reward, "terminal_reason": terminal_reason,
        "invalid_actions": metrics.get("invalid_actions"),
        "exploration_coverage": metrics.get("exploration_coverage"),
        "resource_efficiency": metrics.get("resource_efficiency"),
        "control_elapsed_ms": round(control_elapsed_ms, 3),
    }


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

    def _values(self, state: str) -> list[float]:
        values = self.q_values.setdefault(state, [0.0] * (len(self._actions) + 3))
        values.extend([0.0] * (len(self._actions) + 3 - len(values)))
        return values

    def _candidate_actions(self, observation: dict[str, Any]) -> list[tuple[int, ActionRequest]]:
        return [(self.action_index(action), action) for action in public_action_candidates(observation)]

    def act(self, observation: dict[str, Any]) -> ActionRequest:
        state = self.observation_key(observation)
        allowed = set(observation.get("allowed_action_types", []))
        available = [(index, action) for index, action in self._candidate_actions(observation) if action.type in allowed]
        if not available:
            return ActionRequest(type="wait")
        if self.random.random() < self.config.epsilon:
            return self.random.choice(available)[1]
        values = self._values(state)
        best = max(values[index] for index, _ in available)
        return next(action for index, action in available if values[index] == best)

    def update(self, observation: dict[str, Any], action: ActionRequest, reward: float, next_observation: dict[str, Any], terminated: bool) -> None:
        state, next_state = self.observation_key(observation), self.observation_key(next_observation)
        index = self.action_index(action)
        values = self._values(state)
        target = reward if terminated else reward + self.config.discount * max(self._values(next_state))
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
                observation, _ = environment.reset(seed=seed, options=reset_options_for_seed(seed))
            else:
                observation, _ = environment.reset(seed=seed)
            total_reward = 0.0
            last_info: dict[str, Any] = {}
            started = time.perf_counter()
            for step in range(1, max_steps + 1):
                action = policy.act(observation)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                last_info = info
                done = terminated or truncated
                policy.update(observation, action, reward, next_observation, done)
                observation, total_reward = next_observation, total_reward + reward
                if done:
                    episodes.append(_episode_result(seed, step, total_reward, info.get("terminal_reason"), info, (time.perf_counter() - started) * 1000))
                    break
            else:
                episodes.append(_episode_result(seed, max_steps, total_reward, "trainer_step_limit", last_info, (time.perf_counter() - started) * 1000))
    return episodes


def evaluate_tabular_q(
    environment_factory: Callable[[], Any], policy: TabularQPolicy, seeds: list[int], max_steps: int = 256,
    reset_options_for_seed: Callable[[int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a frozen greedy checkpoint without mutating its Q-values."""
    if not seeds or max_steps < 1:
        raise ValueError("evaluation requires seeds and a positive max_steps")
    original_epsilon = policy.config.epsilon
    policy.config = TabularQConfig(policy.config.learning_rate, policy.config.discount, 0)
    try:
        episodes = []
        with environment_factory() as environment:
            for seed in seeds:
                policy.reset(seed)
                if reset_options_for_seed:
                    observation, _ = environment.reset(seed=seed, options=reset_options_for_seed(seed))
                else:
                    observation, _ = environment.reset(seed=seed)
                total_reward = 0.0
                last_info: dict[str, Any] = {}
                started = time.perf_counter()
                for step in range(1, max_steps + 1):
                    next_observation, reward, terminated, truncated, info = environment.step(policy.act(observation))
                    last_info = info
                    observation, total_reward = next_observation, total_reward + reward
                    if terminated or truncated:
                        episodes.append(_episode_result(seed, step, total_reward, info.get("terminal_reason"), info, (time.perf_counter() - started) * 1000))
                        break
                else:
                    episodes.append(_episode_result(seed, max_steps, total_reward, "evaluator_step_limit", last_info, (time.perf_counter() - started) * 1000))
        return episodes
    finally:
        policy.config = TabularQConfig(policy.config.learning_rate, policy.config.discount, original_epsilon)


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
