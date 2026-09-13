from embodied_ai.datasets import summarize_world_model_dataset, world_model_transitions


def test_world_model_transitions_preserve_episode_boundaries_and_policy_observations():
    records = [
        {"experiment_id": "experiment-a", "world_manifest": {"world_hash": "fnv1a64:a"}, "run_id": "a", "step": 1, "observation": {"step": 0}, "next_observation": {"step": 1}, "chosen_action": {"type": "wait"}, "reward": 1, "done": False, "research_snapshot": {"hidden": 1}},
        {"experiment_id": "experiment-a", "world_manifest": {"world_hash": "fnv1a64:a"}, "run_id": "a", "step": 2, "observation": {"step": 1}, "next_observation": {"step": 2, "terminal": True}, "chosen_action": {"type": "wait"}, "reward": 2, "done": True, "terminal_reason": "escaped", "research_snapshot": {"hidden": 2}},
        {"run_id": "b", "step": 1, "observation": {"step": 0}, "chosen_action": {"type": "wait"}, "reward": 0, "done": True, "terminal_reason": "timeout"},
    ]
    transitions = world_model_transitions(records)
    assert transitions[0]["observation_t_plus_1"] == {"step": 1}
    assert transitions[0]["experiment_id"] == "experiment-a"
    assert transitions[0]["world_manifest"] == {"world_hash": "fnv1a64:a"}
    assert transitions[1]["observation_t_plus_1"] == {"step": 2, "terminal": True}
    assert "privileged_state_t" not in transitions[0]
    assert transitions[1]["terminated_t"] and not transitions[1]["truncated_t"]
    assert transitions[2]["truncated_t"] and not transitions[2]["terminated_t"]


def test_world_model_transitions_require_explicit_privileged_opt_in():
    records = [{"run_id": "a", "step": 1, "observation": {}, "chosen_action": {"type": "wait"}, "research_snapshot": {"secret": True}, "next_research_snapshot": {"secret": "after"}}]
    transition = world_model_transitions(records, include_privileged_state=True)[0]
    assert transition["privileged_state_t"] == {"secret": True}
    assert transition["privileged_state_t_plus_1"] == {"secret": "after"}


def test_world_model_transitions_reject_mixed_run_provenance():
    records = [
        {"run_id": "a", "step": 1, "experiment_id": "one", "observation": {}, "chosen_action": {"type": "wait"}},
        {"run_id": "a", "step": 2, "experiment_id": "two", "observation": {}, "chosen_action": {"type": "wait"}},
    ]
    try:
        world_model_transitions(records)
    except ValueError as error:
        assert "mixed" in str(error)
    else:
        raise AssertionError("mixed experiment provenance was accepted")


def test_dataset_summary_reports_coverage_and_incomplete_privileged_pairs():
    report = summarize_world_model_dataset([
        {"run_id": "a", "step": 1, "observation_mode": "normal", "policy_state_mode": "reset", "next_observation": {}, "chosen_action": {"type": "wait"}, "agent_metadata": {"confidence": .8, "planner": {"name": "astar"}}, "reward_breakdown": {"baseline": -1, "progress": 5, "invalid_action_penalty": 0, "hazard_penalty": 0, "terminal": 0}, "research_snapshot": {}, "next_research_snapshot": {}},
        {"run_id": "b", "step": 1, "observation_mode": "noisy", "policy_state_mode": "preserve", "chosen_action": {"type": "move"}, "done": True, "terminal_reason": "timeout", "research_snapshot": {}},
    ])
    assert report["runs"] == 2
    assert report["action_counts"] == {"move": 1, "wait": 1}
    assert report["terminal_reasons"] == {"timeout": 1}
    assert report["observation_modes"] == {"noisy": 1, "normal": 1}
    assert report["policy_state_modes"] == {"preserve": 1, "reset": 1}
    assert report["runs_with_mixed_policy_state_mode"] == 0
    assert report["records_with_agent_metadata"] == 1
    assert report["agent_metadata_field_counts"] == {"confidence": 1, "planner": 1}
    assert report["planner_metadata_records"] == 1
    assert report["records_with_exact_next_observation"] == 1
    assert report["records_with_reward_breakdown"] == 1
    assert report["reward_component_totals"] == {"baseline": -1, "progress": 5, "invalid_action_penalty": 0, "hazard_penalty": 0, "terminal": 0}
    assert report["privileged_snapshot_pairs"] == 1
    assert report["partial_privileged_snapshot_records"] == 1


def test_dataset_summary_flags_mixed_policy_state_modes_within_one_run():
    report = summarize_world_model_dataset([
        {"run_id": "a", "step": 1, "policy_state_mode": "reset", "observation": {}, "chosen_action": {"type": "wait"}},
        {"run_id": "a", "step": 2, "policy_state_mode": "preserve", "observation": {}, "chosen_action": {"type": "wait"}},
    ])
    assert report["runs_with_mixed_policy_state_mode"] == 1
