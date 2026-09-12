import pytest

from embodied_ai.environment import EmbodiedEnv, EmbodiedEnvConfig
from embodied_ai import environment


class FakeClient:
    def __init__(self, _url):
        self.created = []
        self.closed = False

    def create(self, seed, max_steps, observation_mode, scenario_id, generated_world=None, reward_config=None):
        self.created.append((seed, max_steps, observation_mode, scenario_id, generated_world, reward_config))
        return {"run_id": "episode-1", "observation": {"visible_cells": [], "allowed_action_types": ["wait"]}}

    def step(self, run_id, action):
        assert run_id == "episode-1"
        assert action.type == "wait"
        return {
            "observation": {"visible_cells": []}, "reward": 3, "done": True,
            "terminal_reason": "escaped", "events": [], "reward_breakdown": {"terminal": 100},
            "metrics": {"escaped": True}, "simulation_time": 4,
        }

    def close(self):
        self.closed = True


def test_headless_environment_uses_only_rust_client(monkeypatch):
    monkeypatch.setattr(environment, "RustRunClient", FakeClient)
    env = EmbodiedEnv(EmbodiedEnvConfig(scenario_id="generated_grid", observation_mode="minimal", max_steps=9))
    observation, info = env.reset(seed=123)
    assert observation["allowed_action_types"] == ["wait"]
    assert info == {"run_id": "episode-1", "scenario_id": "generated_grid", "world_manifest": None, "reward_config": None, "seed": 123, "observation_mode": "minimal"}
    next_observation, reward, terminated, truncated, step_info = env.step({"type": "wait"})
    assert next_observation == {"visible_cells": []}
    assert (reward, terminated, truncated) == (3.0, True, False)
    assert step_info["evaluator"]["metrics"] == {"escaped": True}
    env.close()
    assert env._client.closed


def test_headless_environment_requires_reset(monkeypatch):
    monkeypatch.setattr(environment, "RustRunClient", FakeClient)
    with pytest.raises(RuntimeError, match="reset"):
        EmbodiedEnv().step({"type": "wait"})


def test_headless_environment_can_request_a_generated_world(monkeypatch):
    monkeypatch.setattr(environment, "RustRunClient", FakeClient)
    world = {"seed": 321}
    env = EmbodiedEnv(EmbodiedEnvConfig(generated_world=world))
    _, info = env.reset(seed=123)
    assert env._client.created == [(123, None, "normal", None, world, None)]
    assert info["scenario_id"] is None


def test_headless_environment_forwards_reward_configuration(monkeypatch):
    monkeypatch.setattr(environment, "RustRunClient", FakeClient)
    rewards = {"baseline_per_step": -3}
    env = EmbodiedEnv(EmbodiedEnvConfig(reward_config=rewards))
    _, info = env.reset()
    assert env._client.created == [(42, None, "normal", None, None, rewards)]
    assert info["reward_config"] is None
