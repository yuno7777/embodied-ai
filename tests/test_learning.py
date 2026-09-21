from embodied_ai.learning import TabularQConfig, TabularQPolicy, evaluate_tabular_partitions, evaluate_tabular_q, train_tabular_q
from embodied_ai.schemas import ActionRequest


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


def test_tabular_episodes_preserve_evaluator_metrics_without_policy_access():
    class Environment:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def reset(self, *, seed): return observation(), {}
        def step(self, _action): return observation(), 2, True, False, {"terminal_reason": "escaped", "evaluator": {"metrics": {"invalid_actions": 0, "exploration_coverage": .5, "resource_efficiency": .8}}}
    result = evaluate_tabular_q(Environment, TabularQPolicy(), [1])[0]
    assert result["invalid_actions"] == 0
    assert result["exploration_coverage"] == .5
    assert result["resource_efficiency"] == .8
    assert result["control_elapsed_ms"] >= 0


def test_tabular_step_limits_keep_the_last_authoritative_evaluator_metrics():
    class Environment:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def reset(self, *, seed): return observation(), {}
        def step(self, _action): return observation(), 1, False, False, {"terminal_reason": None, "evaluator": {"metrics": {"exploration_coverage": .25}}}
    result = evaluate_tabular_q(Environment, TabularQPolicy(), [1], max_steps=1)[0]
    assert result["terminal_reason"] == "evaluator_step_limit"
    assert result["exploration_coverage"] == .25


def test_tabular_q_proposes_only_locally_visible_object_actions():
    policy = TabularQPolicy(TabularQConfig(epsilon=0), seed=1)
    observed = observation()
    observed["allowed_action_types"].append("pickup")
    observed["visible_cells"][0]["entities"] = [{"id": "local_key", "type": "item"}]
    policy.q_values[policy.observation_key(observed)] = [0] * 8
    policy.q_values[policy.observation_key(observed)][policy._pickup_index] = 4
    assert policy.act(observed).model_dump(exclude_none=True) == {"type": "pickup", "item_id": "local_key"}


def test_tabular_q_uses_only_reachable_entities_and_public_inventory_items():
    policy = TabularQPolicy(TabularQConfig(epsilon=0), seed=1)
    observed = observation()
    observed["allowed_action_types"] += ["open", "pickup", "use_item"]
    observed["agent"]["inventory"] = ["water"]
    observed["visible_cells"] = [
        {"relative_position": {"x": 0, "y": 0}, "terrain": "floor", "entities": [{"id": "near_key", "type": "item"}]},
        {"relative_position": {"x": 1, "y": 0}, "terrain": "floor", "entities": [{"id": "near_door", "type": "door"}]},
        {"relative_position": {"x": 2, "y": 0}, "terrain": "floor", "entities": [{"id": "far_item", "type": "item"}, {"id": "far_door", "type": "door"}]},
    ]
    candidates = policy._candidate_actions(observed)
    payloads = [action.model_dump(exclude_none=True) for _, action in candidates]
    assert {"type": "pickup", "item_id": "near_key"} in payloads
    assert {"type": "open", "target_id": "near_door"} in payloads
    assert {"type": "use_item", "item_id": "water"} in payloads
    assert all("far_" not in str(payload) for payload in payloads)
    policy.q_values[policy.observation_key(observed)] = [0] * 9
    policy.q_values[policy.observation_key(observed)][policy._use_item_index] = 5
    assert policy.act(observed).model_dump(exclude_none=True) == {"type": "use_item", "item_id": "water"}


def test_tabular_q_evaluation_is_greedy_and_does_not_mutate_values():
    class Environment:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def reset(self, *, seed): return observation(), {}
        def step(self, _action): return observation(), 1, True, False, {"terminal_reason": "escaped"}
    policy = TabularQPolicy(TabularQConfig(epsilon=1), seed=1)
    policy.q_values[policy.observation_key(observation())] = [2, 0, 0, 0, 0, 0]
    before = {state: list(values) for state, values in policy.q_values.items()}
    assert evaluate_tabular_q(Environment, policy, [9])[0]["terminal_reason"] == "escaped"
    assert policy.config.epsilon == 1 and policy.q_values == before


def test_tabular_q_evaluation_does_not_create_values_for_unseen_observations():
    class Environment:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def reset(self, *, seed): return observation(25), {}
        def step(self, _action): return observation(50), 1, True, False, {"terminal_reason": "escaped"}
    policy = TabularQPolicy(TabularQConfig(epsilon=1), seed=1)
    assert evaluate_tabular_q(Environment, policy, [9])[0]["terminal_reason"] == "escaped"
    assert policy.q_values == {}


def test_tabular_q_evaluation_reads_short_legacy_vectors_without_extending_them():
    observed = observation()
    observed["allowed_action_types"].append("pickup")
    observed["visible_cells"][0]["entities"] = [{"id": "key", "type": "item"}]
    policy = TabularQPolicy(TabularQConfig(epsilon=1), seed=1)
    policy.q_values[policy.observation_key(observed)] = [0] * 6
    before = {state: list(values) for state, values in policy.q_values.items()}
    assert policy._greedy_action(observed).type == "move"
    assert policy.q_values == before


def test_tabular_q_bootstraps_truncations_using_only_available_next_actions():
    policy = TabularQPolicy(TabularQConfig(learning_rate=1, discount=.5, epsilon=0), seed=1)
    before, after = observation(), observation(75)
    after["allowed_action_types"] = ["wait"]
    before_key, after_key = policy.observation_key(before), policy.observation_key(after)
    policy.q_values[after_key] = [999, 0, 0, 0, 0, 5, 0, 0, 0]
    policy.update(before, ActionRequest(type="move", direction="north"), 1, after, terminated=False)
    assert policy.q_values[before_key][0] == 3.5
    policy.update(before, ActionRequest(type="move", direction="north"), 2, after, terminated=True)
    assert policy.q_values[before_key][0] == 2


def test_tabular_training_passes_truncation_as_nonterminal_to_q_update():
    class Environment:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def reset(self, *, seed): return observation(), {}
        def step(self, _action): return observation(75), 1, False, True, {"terminal_reason": "timeout"}
    class Policy(TabularQPolicy):
        def __init__(self): super().__init__(); self.terminated = []
        def update(self, *args, **kwargs): self.terminated.append(kwargs["terminated"])
    policy = Policy()
    train_tabular_q(Environment, policy, [1])
    assert policy.terminated == [False]


def test_tabular_partition_evaluation_keeps_distributions_disjoint_and_frozen():
    calls = []
    class Environment:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def reset(self, *, seed, options): calls.append((seed, options)); return observation(), {}
        def step(self, _action): return observation(), 1, True, False, {"terminal_reason": "escaped"}
    policy = TabularQPolicy(TabularQConfig(epsilon=1), seed=1)
    result = evaluate_tabular_partitions(Environment, policy, {"train": [1], "validation": [2], "test": [3]}, reset_options_for_partition=lambda partition, seed: {"partition": partition, "seed": seed})
    assert {name: episodes[0]["seed"] for name, episodes in result.items()} == {"train": 1, "validation": 2, "test": 3}
    assert calls == [(3, {"partition": "test", "seed": 3}), (1, {"partition": "train", "seed": 1}), (2, {"partition": "validation", "seed": 2})]
    assert policy.config.epsilon == 1
