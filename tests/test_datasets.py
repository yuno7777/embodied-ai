from embodied_ai.datasets import world_model_transitions


def test_world_model_transitions_preserve_episode_boundaries_and_policy_observations():
    records = [
        {"run_id": "a", "step": 1, "observation": {"step": 0}, "chosen_action": {"type": "wait"}, "reward": 1, "done": False, "research_snapshot": {"hidden": 1}},
        {"run_id": "a", "step": 2, "observation": {"step": 1}, "chosen_action": {"type": "wait"}, "reward": 2, "done": True, "terminal_reason": "escaped", "research_snapshot": {"hidden": 2}},
        {"run_id": "b", "step": 1, "observation": {"step": 0}, "chosen_action": {"type": "wait"}, "reward": 0, "done": True, "terminal_reason": "timeout"},
    ]
    transitions = world_model_transitions(records)
    assert transitions[0]["observation_t_plus_1"] == {"step": 1}
    assert transitions[1]["observation_t_plus_1"] == {"step": 1}
    assert "privileged_state_t" not in transitions[0]
    assert transitions[1]["terminated_t"] and not transitions[1]["truncated_t"]
    assert transitions[2]["truncated_t"] and not transitions[2]["terminated_t"]


def test_world_model_transitions_require_explicit_privileged_opt_in():
    records = [{"run_id": "a", "step": 1, "observation": {}, "chosen_action": {"type": "wait"}, "research_snapshot": {"secret": True}}]
    transition = world_model_transitions(records, include_privileged_state=True)[0]
    assert transition["privileged_state_t"] == {"secret": True}
