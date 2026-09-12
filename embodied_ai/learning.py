"""Small tabular RL baseline operating exclusively through public observations."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .schemas import ActionRequest


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

    It intentionally uses only movement, wait, and inspect actions because those
    are valid without guessing hidden object IDs. More capable policies can use
    the same environment contract with richer action proposal mechanisms.
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
        try:
            return self._actions.index(action)
        except ValueError as error:
            raise ValueError("action is outside the tabular baseline action set") from error

    def _values(self, state: str) -> list[float]:
        return self.q_values.setdefault(state, [0.0] * len(self._actions))

    def act(self, observation: dict[str, Any]) -> ActionRequest:
        state = self.observation_key(observation)
        allowed = set(observation.get("allowed_action_types", []))
        available = [index for index, action in enumerate(self._actions) if action.type in allowed]
        if not available:
            return ActionRequest(type="wait")
        if self.random.random() < self.config.epsilon:
            return self._actions[self.random.choice(available)]
        values = self._values(state)
        best = max(values[index] for index in available)
        return self._actions[next(index for index in available if values[index] == best)]

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
    environment_factory: Callable[[], Any], policy: TabularQPolicy, seeds: list[int], max_steps: int = 256
) -> list[dict[str, Any]]:
    """Run online Q-learning episodes against an RL-style authoritative environment."""
    if not seeds or max_steps < 1:
        raise ValueError("training requires seeds and a positive max_steps")
    episodes = []
    with environment_factory() as environment:
        for seed in seeds:
            policy.reset(seed)
            observation, _ = environment.reset(seed=seed)
            total_reward = 0.0
            for step in range(1, max_steps + 1):
                action = policy.act(observation)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                done = terminated or truncated
                policy.update(observation, action, reward, next_observation, done)
                observation, total_reward = next_observation, total_reward + reward
                if done:
                    episodes.append({"seed": seed, "steps": step, "total_reward": total_reward, "terminal_reason": info.get("terminal_reason")})
                    break
            else:
                episodes.append({"seed": seed, "steps": max_steps, "total_reward": total_reward, "terminal_reason": "trainer_step_limit"})
    return episodes
