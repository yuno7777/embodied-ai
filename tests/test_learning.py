from embodied_ai.learning import TabularQConfig, TabularQPolicy, evaluate_tabular_q, train_tabular_q


def observation(health=100):
    return {"agent": {"health": health, "energy": 80, "hydration": 70, "inventory": []}, "visible_cells": [{"relative_position": {"x": 0, "y": 0}, "terrain": "floor", "entities": []}], "allowed_action_types": ["move", "inspect", "wait"]}


def test_tabular_q_uses_only_public_observation_and_updates_values(tmp_path):
    policy = TabularQPolicy(TabularQConfig(learning_rate=1, discount=0, epsilon=0), seed=2)
    action = policy.act(observation())
    assert action.type == "move"
    policy.update(observation(), action, 3, observation(75), True)
    assert policy.q_values[policy.observation_key(observation())][0] == 3
    checkpoint = policy.save(tmp_path / "policy.json")
    loaded = TabularQPolicy.load(checkpoint, seed=2)
    assert loaded.q_values == policy.q_values


def test_tabular_q_training_uses_rl_style_environment_only():
    class Environment:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def reset(self, *, seed):
            self.seed = seed
            return observation(), {"run_id": str(seed)}
        def step(self, action):
            assert action.type in {"move", "inspect", "wait"}
            return observation(75), 2.0, True, False, {"terminal_reason": "escaped"}
    policy = TabularQPolicy(TabularQConfig(learning_rate=1, discount=0, epsilon=0))
    episodes = train_tabular_q(Environment, policy, [1, 2])
    assert [episode["terminal_reason"] for episode in episodes] == ["escaped", "escaped"]
    assert all(episode["total_reward"] == 2 for episode in episodes)


def test_tabular_q_proposes_only_locally_visible_object_actions():
    policy = TabularQPolicy(TabularQConfig(epsilon=0), seed=1)
    observed = observation()
    observed["allowed_action_types"].append("pickup")
    observed["visible_cells"][0]["entities"] = [{"id": "local_key", "type": "item"}]
    policy.q_values[policy.observation_key(observed)] = [0] * 8
    policy.q_values[policy.observation_key(observed)][policy._pickup_index] = 4
    assert policy.act(observed).model_dump(exclude_none=True) == {"type": "pickup", "item_id": "local_key"}


def test_tabular_q_evaluation_is_greedy_and_does_not_mutate_values():
    class Environment:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def reset(self, *, seed): return observation(), {}
        def step(self, _action): return observation(), 1, True, False, {"terminal_reason": "escaped"}
    policy = TabularQPolicy(TabularQConfig(epsilon=1), seed=1)
    policy.q_values[policy.observation_key(observation())] = [2, 0, 0, 0, 0, 0]
    before = dict(policy.q_values)
    assert evaluate_tabular_q(Environment, policy, [9])[0]["terminal_reason"] == "escaped"
    assert policy.config.epsilon == 1 and policy.q_values == before
