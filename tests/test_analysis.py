import pytest

from embodied_ai.analysis import TrajectoryValidationError, check_reproducibility, filter_trajectory, replay_fingerprint, summarize_trajectory, validate_trajectory, verify_replay


def records():
    return [
        {"run_id": "run-1", "step": 1, "chosen_action": {"type": "inspect"}, "events": [{"type": "InspectionCompleted"}], "action_valid": True, "latency_ms": 4, "done": False, "terminal_reason": None, "metrics": {}},
        {"run_id": "run-1", "step": 2, "chosen_action": {"type": "move"}, "events": [{"type": "AgentMoved"}], "action_valid": True, "latency_ms": 6, "done": True, "terminal_reason": "escaped", "metrics": {"normalized_score": 100.0, "score_task_success": 100.0}},
    ]


def test_trajectory_summary_is_based_on_exported_records_only():
    summary = summarize_trajectory(records())
    assert summary["run_id"] == "run-1"
    assert summary["outcome"] == "escaped"
    assert summary["valid_action_rate"] == 1
    assert summary["action_counts"] == {"inspect": 1, "move": 1}
    assert summary["event_counts"] == {"AgentMoved": 1, "InspectionCompleted": 1}
    assert summary["mean_provider_latency_ms"] == 5


@pytest.mark.parametrize("mutator", [
    lambda rows: rows.clear(),
    lambda rows: rows.__setitem__(1, {**rows[1], "step": 3}),
    lambda rows: rows.__setitem__(0, {**rows[0], "done": True}),
    lambda rows: rows.__setitem__(1, {**rows[1], "terminal_reason": None}),
    lambda rows: rows.__setitem__(0, {**rows[0], "terminal_reason": "escaped"}),
    lambda rows: rows.__setitem__(0, {**rows[0], "chosen_action": {}}),
])
def test_trajectory_validation_rejects_incomplete_or_inconsistent_records(mutator):
    rows = records()
    mutator(rows)
    with pytest.raises(TrajectoryValidationError):
        validate_trajectory(rows)

def test_trajectory_filters_and_offline_replay_verification():
    assert len(filter_trajectory(records(),action_type="move",valid_only=True,event_type="AgentMoved"))==1
    replay={"replay_version":1,"snapshot":{"run_id":"r","step":1},"timeline":[{"run_id":"r","step":0},{"run_id":"r","step":1}],"observations":[{},{}],"events":[],"decisions":[{}],"actions":[{"type":"wait"}]}
    assert verify_replay(replay)=={"valid":True,"run_id":"r","frames":2,"events":0,"decisions":1}
    replay["observations"].pop()
    with pytest.raises(TrajectoryValidationError): verify_replay(replay)

def test_replay_reproducibility_ignores_only_generated_identifiers_and_timestamps():
    left={"replay_version":1,"engine_version":"rust-v1","observation_mode":"normal","max_steps":10,"reward_config":{"baseline_per_step":-1},"scenario_id":"s","scenario_version":1,"seed":7,"snapshot":{"run_id":"a","step":0,"agent":{"health":100}},"timeline":[{"run_id":"a","step":0}],"observations":[{}],"events":[{"event_id":"one","run_id":"a","timestamp":"now","type":"Created"}],"decisions":[],"actions":[]}
    right={**left,"snapshot":{**left["snapshot"],"run_id":"b","agent":{"health":100}},"timeline":[{"run_id":"b","step":0}],"events":[{"event_id":"two","run_id":"b","timestamp":"later","type":"Created"}]}
    left["timeline"] = [left["snapshot"]]
    right["timeline"] = [right["snapshot"]]
    left["control"] = {"paused_ms": 15}
    right["control"] = {"paused_ms": 200}
    assert replay_fingerprint(left)==replay_fingerprint(right)
    assert check_reproducibility(left,right)["reproducible"] is True
    right["snapshot"]["agent"]["health"]=99
    assert check_reproducibility(left,right)["reproducible"] is False


def test_replay_reproducibility_rejects_different_generated_world_or_reward_inputs():
    base = {"replay_version":1,"engine_version":"rust-v1", "observation_mode":"normal", "max_steps":10, "reward_config":{"baseline_per_step":-1}, "scenario_id":"generated", "scenario_version":1, "seed":7, "world_manifest":{"world_hash":"fnv1a64:a", "seed":99}, "snapshot":{"run_id":"r","step":0}, "timeline":[{"run_id":"r","step":0}], "observations":[{}], "events":[], "decisions":[], "actions":[]}
    changed_world = {**base, "world_manifest": {"world_hash":"fnv1a64:b", "seed":99}}
    with pytest.raises(TrajectoryValidationError, match="manifests"):
        check_reproducibility(base, changed_world)
    changed_rewards = {**base, "reward_config": {"baseline_per_step":-2}}
    with pytest.raises(TrajectoryValidationError, match="reward"):
        check_reproducibility(base, changed_rewards)


def test_replay_verifier_accepts_external_interrupt_without_a_simulation_step():
    initial = {"run_id": "r", "step": 0, "done": False}
    final = {**initial, "done": True, "terminal_reason": "aborted"}
    replay = {"replay_version": 1, "snapshot": final, "timeline": [initial, final], "observations": [{"step": 0}, {"step": 0}], "actions": []}
    assert verify_replay(replay)["valid"]


@pytest.mark.parametrize("corruption", ["foreign_run", "bad_frame", "bool_step", "snapshot", "observation"])
def test_replay_verifier_rejects_corruption(corruption):
    initial = {"run_id": "r", "step": 0}
    final = {"run_id": "r", "step": 1}
    replay = {"replay_version": 1, "snapshot": dict(final), "timeline": [initial, final], "observations": [{"step": 0}, {"step": 1}], "actions": [{"type": "wait"}]}
    if corruption == "foreign_run": initial["run_id"] = "other"
    elif corruption == "bad_frame": replay["timeline"][0] = None
    elif corruption == "bool_step": initial["step"] = False
    elif corruption == "snapshot": replay["snapshot"]["health"] = 1
    else: replay["observations"][1]["step"] = 2
    with pytest.raises(TrajectoryValidationError):
        verify_replay(replay)


@pytest.mark.parametrize("replay", [
    {"replay_version": 2, "snapshot": {"run_id": "r", "step": 0}, "timeline": [{"run_id": "r", "step": 0}], "observations": [{}], "actions": []},
    {"replay_version": 1, "snapshot": {"run_id": "r", "step": 1}, "timeline": [{"run_id": "r", "step": 0}, {"run_id": "r", "step": 1}], "observations": [{}, {}], "actions": []},
    {"replay_version": 1, "snapshot": {"run_id": "r", "step": 1}, "timeline": [{"run_id": "r", "step": 0}, {"run_id": "r", "step": 1}], "observations": [{}, {}], "actions": [{"type": "move"}]},
])
def test_replay_verifier_rejects_unsupported_or_unreconstructable_action_streams(replay):
    with pytest.raises(TrajectoryValidationError):
        verify_replay(replay)
