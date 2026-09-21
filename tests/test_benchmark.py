import json

import embodied_ai.benchmark as benchmark
from embodied_ai.providers import CautiousProvider, ExplorerProvider, MockReasoningProvider, RandomValidProvider, ScriptedProvider
from embodied_ai.runner import RemoteRunResult


def test_benchmark_provider_factory_preserves_selected_provider():
    assert isinstance(benchmark.provider_for("scripted", 7), ScriptedProvider)
    assert isinstance(benchmark.provider_for("mock_reasoning", 7), MockReasoningProvider)
    assert isinstance(benchmark.provider_for("random_valid", 7), RandomValidProvider)
    assert isinstance(benchmark.provider_for("cautious", 7), CautiousProvider)
    assert isinstance(benchmark.provider_for("explorer", 7), ExplorerProvider)


def test_benchmark_provider_factory_rejects_unknown_provider():
    try:
        benchmark.provider_for("unknown", 7)
    except ValueError as error:
        assert "Unknown benchmark provider" in str(error)
    else:
        raise AssertionError("unknown provider silently selected a policy")


def test_generator_config_fingerprint_is_canonical_and_sensitive_to_values():
    assert benchmark.generator_config_fingerprint({"max_rooms": 3, "min_rooms": 2}) == benchmark.generator_config_fingerprint({"min_rooms": 2, "max_rooms": 3})
    assert benchmark.generator_config_fingerprint({"min_rooms": 2}) != benchmark.generator_config_fingerprint({"min_rooms": 3})
    assert benchmark.generator_config_fingerprint(None) is None


def test_benchmark_slices_keep_events_and_decisions_separate():
    events, decisions = benchmark.benchmark_slices([{
        "run_id": "r", "scenario_id": "s", "scenario_version": 1, "seed": 2, "step": 3,
        "provider": "mock", "model": None, "chosen_action": {"type": "wait"}, "decision_summary": "safe",
        "action_valid": True, "latency_ms": 4, "token_usage": None,
        "events": [{"type": "ActionCompleted", "message": "Waited."}],
    }])
    assert events[0]["event_type"] == "ActionCompleted"
    assert decisions[0]["chosen_action"] == '{"type": "wait"}'

def test_bounded_concurrent_benchmark_is_seed_ordered_and_writes_csv(monkeypatch, tmp_path):
    def fake_run(provider, seed, _base_url):
        record={"run_id":f"run-{seed}","scenario_id":"survival_room","scenario_version":3,"seed":seed,"step":1,"provider":provider.name,"model":None,"chosen_action":{"type":"wait"},"decision_summary":"test","action_valid":True,"latency_ms":0,"token_usage":None,"events":[{"type":"ActionCompleted","message":"wait"}],"observation":{},"agent_context":{},"metrics":{"normalized_score":90+seed%2}}
        return RemoteRunResult(f"run-{seed}", "escaped", seed % 3 + 1, [record], initialization_latency_ms=2.5)
    monkeypatch.setattr(benchmark,"run_remote",fake_run)
    class ReplayClient:
        def __init__(self, _base_url): pass
        def replay(self, _run_id): return {"control": {"simulation_latency_us": [100, 300]}}
        def close(self): pass
    monkeypatch.setattr(benchmark, "RustRunClient", ReplayClient)
    report=benchmark.benchmark_remote(4,10,tmp_path,"http://sim","scripted",concurrency=2)
    assert report["concurrency"]==2
    assert [row["seed"] for row in report["seed_results"]]==[10,11,12,13]
    assert report["simulation_steps_per_second"] == 5000.0
    assert report["mean_initialization_latency_ms"] == 2.5
    assert all(row["initialization_latency_ms"] == 2.5 for row in report["seed_results"])
    assert (tmp_path/"runs.csv").exists() and (tmp_path/"decisions.csv").exists()

def test_benchmark_comparison_reports_directional_deltas():
    result=benchmark.compare_benchmarks(
        {"scenario_id":"s","provider":"left","runs":2,"success_rate":0.5,"mean_score":40,"mean_steps":10},
        {"scenario_id":"s","provider":"right","runs":3,"success_rate":1.0,"mean_score":70,"mean_steps":8},
    )
    assert result["success_rate_delta"]==0.5
    assert result["mean_score_delta"]==30
    assert result["mean_steps_delta"]==-2


def test_generalization_comparison_requires_matching_conditions_and_reports_deltas():
    base = {"report_version": 1, "engine_version": "rust-v1", "observation_mode": "normal", "world_distribution": {"train": [0], "validation": [8000], "test": [9000]}, "generator_config": None, "generator_configs_by_partition": None, "generalization_gap": {"train_minus_validation_success_rate": .2, "train_minus_test_success_rate": .3}, "partitions": {name: {"success_rate": .5, "mean_episode_reward": 1} for name in ("train", "validation", "test")}}
    left_partitions = {**base["partitions"], "test": {"success_rate": .5, "mean_episode_reward": 1, "hazard_kind_breakdown": {"fire": {"episodes": 1, "success_rate": 0}}, "room_count_breakdown": {"3": {"episodes": 1, "success_rate": 0}}, "mechanics_breakdown": {"rooms:3|hazards:fire": {"episodes": 1, "success_rate": 0}}}}
    right = {**base, "experiment_id": "right", "generalization_gap": {"train_minus_validation_success_rate": .1, "train_minus_test_success_rate": .4}, "partitions": {**left_partitions, "test": {"success_rate": .75, "mean_episode_reward": 3, "hazard_kind_breakdown": {"fire": {"episodes": 1, "success_rate": 1}}, "room_count_breakdown": {"3": {"episodes": 1, "success_rate": 1}}, "mechanics_breakdown": {"rooms:3|hazards:fire": {"episodes": 1, "success_rate": 1}}}}}
    result = benchmark.compare_generalization_reports({**base, "experiment_id": "left", "partitions": left_partitions}, right)
    assert result["partitions"]["test"]["success_rate_delta"] == .25
    assert round(result["generalization_gap"]["train_minus_test_success_rate_delta"], 6) == .1
    assert result["partitions"]["test"]["hazard_kind_breakdown"]["fire"]["success_rate_delta"] == 1
    assert result["partitions"]["test"]["room_count_breakdown"]["3"]["success_rate_delta"] == 1
    assert result["partitions"]["test"]["mechanics_breakdown"]["rooms:3|hazards:fire"]["success_rate_delta"] == 1
    try:
        benchmark.compare_generalization_reports(base, {**right, "observation_mode": "oracle"})
    except ValueError as error:
        assert "observation_mode" in str(error)
    else:
        raise AssertionError("incompatible sensor reports were compared")
    try:
        benchmark.compare_generalization_reports({key: value for key, value in base.items() if key != "engine_version"}, right)
    except ValueError as error:
        assert "engine_version" in str(error)
    else:
        raise AssertionError("reports with missing provenance were compared")


def test_generalization_comparison_reports_paired_seed_outcomes():
    rows = [
        {"partition": "train", "seed": 1, "outcome": "escaped", "observation_mode": "normal", "generator_config": None},
        {"partition": "validation", "seed": 2, "outcome": "timeout", "observation_mode": "normal", "generator_config": None},
        {"partition": "test", "seed": 3, "outcome": "timeout", "observation_mode": "normal", "generator_config": None},
    ]
    left = benchmark.summarize_generalization(rows) | {
        "engine_version": "rust-v1", "observation_mode": "normal",
        "world_distribution": {"train": [1], "validation": [2], "test": [3]},
        "generator_config": None, "generator_configs_by_partition": None,
        "episode_results": rows,
    }
    right_rows = [{**row, "outcome": "escaped"} if row["partition"] == "test" else row for row in rows]
    right = benchmark.summarize_generalization(right_rows) | {
        **{key: value for key, value in left.items() if key not in {"partitions", "generalization_gap", "episode_results"}},
        "partitions": benchmark.summarize_generalization(right_rows)["partitions"],
        "generalization_gap": benchmark.summarize_generalization(right_rows)["generalization_gap"],
        "episode_results": right_rows,
    }
    result = benchmark.compare_generalization_reports(left, right)
    assert result["paired_episode_comparison"]["test"] == {
        "episodes": 1, "both_success": 0, "left_only_success": 0,
        "right_only_success": 1, "neither_success": 0, "paired_success_rate_delta": 1.0,
        "world_manifests_checked": False,
    }


def test_generalization_paired_comparison_rejects_different_generated_worlds():
    rows = [
        {"partition": "train", "seed": 1, "outcome": "escaped", "observation_mode": "normal", "generator_config": None, "world_manifest": {"world_hash": "a"}},
        {"partition": "validation", "seed": 2, "outcome": "timeout", "observation_mode": "normal", "generator_config": None, "world_manifest": {"world_hash": "b"}},
        {"partition": "test", "seed": 3, "outcome": "timeout", "observation_mode": "normal", "generator_config": None, "world_manifest": {"world_hash": "c"}},
    ]
    left = benchmark.summarize_generalization(rows) | {"engine_version": "rust-v1", "observation_mode": "normal", "world_distribution": {"train": [1], "validation": [2], "test": [3]}, "generator_config": None, "generator_configs_by_partition": None, "episode_results": rows}
    right_rows = [{**row, "world_manifest": {"world_hash": "wrong"}} if row["seed"] == 3 else row for row in rows]
    right_summary = benchmark.summarize_generalization(right_rows)
    right = {**left, "partitions": right_summary["partitions"], "generalization_gap": right_summary["generalization_gap"], "episode_results": right_rows}
    try:
        benchmark.compare_generalization_reports(left, right)
    except ValueError as error:
        assert "world manifests" in str(error)
    else:
        raise AssertionError("paired comparison accepted different generated worlds")


def test_generalization_plan_rejects_overlapping_seed_sets():
    train = benchmark.SeedPartition.from_range("train", 0, 2)
    validation = benchmark.SeedPartition.from_range("validation", 2, 3)
    test = benchmark.SeedPartition.from_range("test", 3, 4)
    plan = benchmark.GeneralizationPlan(train, validation, test)
    assert plan.as_dict() == {"train": [0, 1], "validation": [2], "test": [3]}
    try:
        benchmark.GeneralizationPlan(train, benchmark.SeedPartition.from_range("validation", 1, 3), test)
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("overlapping held-out partitions were accepted")


def test_generalization_summary_reports_held_out_gap_and_uncertainty():
    rows = [
        {"partition": "train", "outcome": "escaped", "steps": 2, "total_reward": 5, "invalid_actions": 0, "exploration_coverage": .7, "resource_efficiency": .9, "control_elapsed_ms": 10, "hazard_kinds": ["electrical"], "room_count": 2, "mechanics_signature": "rooms:2|hazards:electrical"},
        {"partition": "validation", "outcome": "hazard", "steps": 4, "total_reward": -2, "invalid_actions": 1, "exploration_coverage": .4, "resource_efficiency": .2, "control_elapsed_ms": 20, "hazard_kinds": ["fire"], "room_count": 3, "mechanics_signature": "rooms:3|hazards:fire"},
        {"partition": "test", "outcome": "hazard", "steps": 3, "total_reward": -3, "invalid_actions": 1, "exploration_coverage": .3, "resource_efficiency": .1, "control_elapsed_ms": 15, "hazard_kinds": ["fire"], "room_count": 3, "mechanics_signature": "rooms:3|hazards:fire"},
    ]
    report = benchmark.summarize_generalization(rows)
    assert report["partitions"]["train"]["success_rate"] == 1.0
    assert report["partitions"]["test"]["success_rate"] == 0.0
    assert report["generalization_gap"]["train_minus_test_success_rate"] == 1.0
    assert len(report["partitions"]["test"]["success_rate_wilson_95"]) == 2
    assert report["partitions"]["validation"]["failure_reasons"] == {"hazard": 1}
    hazard_slice = report["partitions"]["train"]["hazard_kind_breakdown"]["electrical"]
    assert hazard_slice["episodes"] == 1 and hazard_slice["success_rate"] == 1.0
    assert len(hazard_slice["success_rate_wilson_95"]) == 2
    topology_slice = report["partitions"]["validation"]["room_count_breakdown"]["3"]
    assert topology_slice["episodes"] == 1 and topology_slice["success_rate"] == 0.0
    assert len(topology_slice["success_rate_wilson_95"]) == 2
    assert report["partitions"]["test"]["mechanics_breakdown"]["rooms:3|hazards:fire"]["episodes"] == 1


def test_generalization_provenance_rejects_episode_rows_that_disagree_with_the_report():
    rows = [
        {"partition": "train", "seed": 1, "outcome": "escaped", "observation_mode": "normal", "generator_config": None},
        {"partition": "validation", "seed": 2, "outcome": "timeout", "observation_mode": "normal", "generator_config": None},
        {"partition": "test", "seed": 3, "outcome": "timeout", "observation_mode": "normal", "generator_config": None},
    ]
    report = benchmark.summarize_generalization(rows) | {
        "engine_version": "rust-v1", "observation_mode": "normal",
        "world_distribution": {"train": [1], "validation": [2], "test": [3]},
        "generator_config": None, "generator_configs_by_partition": None,
        "episode_results": rows,
    }
    benchmark.validate_generalization_report(report)
    tampered = {**report, "episode_results": [{**rows[0], "seed": 99}, *rows[1:]]}
    try:
        benchmark.validate_generalization_report(tampered)
    except ValueError as error:
        assert "partition seed" in str(error)
    else:
        raise AssertionError("a report with an undeclared episode seed was accepted")
    mismatched_sensor = {**report, "episode_results": [{**rows[0], "observation_mode": "oracle"}, *rows[1:]]}
    try:
        benchmark.validate_generalization_report(mismatched_sensor)
    except ValueError as error:
        assert "observation_mode" in str(error)
    else:
        raise AssertionError("an episode from another sensor condition was accepted")
    empty_split = {**report, "world_distribution": {"train": [], "validation": [2], "test": [3]}}
    try:
        benchmark.validate_generalization_report(empty_split)
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("an empty evaluation split was accepted")
    mixed_config = {**report, "generator_config": {}, "generator_configs_by_partition": {name: {} for name in ("train", "validation", "test")}}
    try:
        benchmark.validate_generalization_report(mixed_config)
    except ValueError as error:
        assert "without a shared" in str(error)
    else:
        raise AssertionError("mixed generator configuration sources were accepted")


def test_generalization_evaluation_uses_procedural_worlds_and_writes_report(monkeypatch, tmp_path):
    def fake_run(provider, seed, _base_url, **kwargs):
        expected = {"seed": seed, "config": {"min_width": 9, "max_width": 9}}
        if seed == 1:
            expected["partition"] = "train"
        assert kwargs["generated_world"] == expected
        record = {"reward": 4, "metrics": {"invalid_actions": 0, "exploration_coverage": .5, "resource_efficiency": .8}, "control_elapsed_ms": 10}
        return RemoteRunResult(f"run-{seed}", "escaped" if seed == 1 else "timeout", 2, [record], world_manifest={"scenario": {"hazards": [{"kind": "fire"}], "rooms": [{}, {}, {}]}})
    monkeypatch.setattr(benchmark, "run_remote", fake_run)
    plan = benchmark.GeneralizationPlan(
        benchmark.SeedPartition("train", (1,)),
        benchmark.SeedPartition("validation", (2,)),
        benchmark.SeedPartition("test", (3,)),
    )
    report = benchmark.evaluate_generalization_remote(plan, tmp_path, "http://sim", generator_config={"min_width": 9, "max_width": 9})
    assert report["world_distribution"] == {"train": [1], "validation": [2], "test": [3]}
    assert report["generalization_gap"]["train_minus_test_success_rate"] == 1.0
    assert all("world_manifest" in episode for episode in report["episode_results"])
    assert all(episode["hazard_kinds"] == ["fire"] for episode in report["episode_results"])
    assert all(episode["room_count"] == 3 for episode in report["episode_results"])
    assert all(episode["mechanics_signature"] == "rooms:3|hazards:fire" for episode in report["episode_results"])
    assert report["generator_config_fingerprint"] == benchmark.generator_config_fingerprint({"min_width": 9, "max_width": 9})
    assert all(episode["generator_config_fingerprint"] == report["generator_config_fingerprint"] for episode in report["episode_results"])
    assert (tmp_path / "generalization_report.json").exists()
    assert (tmp_path / "generalization_episodes.jsonl").exists()
    manifest = tmp_path / report["experiment_manifest"]
    assert manifest.exists()
    assert json.loads(manifest.read_text())["world_distribution"] == report["world_distribution"]


def test_generalization_asserts_matching_built_in_partition_membership(monkeypatch, tmp_path):
    requested = []
    def fake_run(_provider, seed, _base_url, **kwargs):
        requested.append(kwargs["generated_world"])
        return RemoteRunResult(f"run-{seed}", "timeout", 1, [{"metrics": {}}])
    monkeypatch.setattr(benchmark, "run_remote", fake_run)
    plan = benchmark.GeneralizationPlan(
        benchmark.SeedPartition("train", (0,)),
        benchmark.SeedPartition("validation", (8000,)),
        benchmark.SeedPartition("test", (9000,)),
    )
    benchmark.evaluate_generalization_remote(plan, tmp_path, "http://sim")
    assert requested == [
        {"seed": 0, "partition": "train"},
        {"seed": 8000, "partition": "validation"},
        {"seed": 9000, "partition": "test"},
    ]


def test_generalization_can_use_distinct_generator_configs_per_partition(monkeypatch, tmp_path):
    requested = []
    def fake_run(_provider, seed, _base_url, **kwargs):
        requested.append(kwargs["generated_world"])
        return RemoteRunResult(f"run-{seed}", "timeout", 1, [{"metrics": {}}])
    monkeypatch.setattr(benchmark, "run_remote", fake_run)
    plan = benchmark.GeneralizationPlan(
        benchmark.SeedPartition("train", (0,)),
        benchmark.SeedPartition("validation", (8000,)),
        benchmark.SeedPartition("test", (9000,)),
    )
    configs = {
        "train": {"min_rooms": 2, "max_rooms": 2},
        "validation": {"min_rooms": 3, "max_rooms": 3},
        "test": {"min_rooms": 3, "max_rooms": 3},
    }
    report = benchmark.evaluate_generalization_remote(
        plan, tmp_path, "http://sim", generator_configs_by_partition=configs,
    )
    assert [item["config"] for item in requested] == [configs["train"], configs["validation"], configs["test"]]
    assert report["generator_configs_by_partition"] == configs
    assert {
        episode["partition"]: episode["generator_config"]
        for episode in report["episode_results"]
    } == configs


def test_generalization_rejects_incomplete_partition_generator_configs(tmp_path):
    plan = benchmark.GeneralizationPlan(
        benchmark.SeedPartition("train", (0,)),
        benchmark.SeedPartition("validation", (1,)),
        benchmark.SeedPartition("test", (2,)),
    )
    try:
        benchmark.evaluate_generalization_remote(
            plan, tmp_path, "http://sim", generator_configs_by_partition={"train": {}},
        )
    except ValueError as error:
        assert "train, validation, and test" in str(error)
    else:
        raise AssertionError("incomplete partition configs were accepted")


def test_generalization_records_a_nondefault_sensor_mode(monkeypatch, tmp_path):
    requested = []
    def fake_run(_provider, seed, _base_url, **kwargs):
        requested.append(kwargs)
        return RemoteRunResult(f"run-{seed}", "timeout", 1, [{"metrics": {}}])
    monkeypatch.setattr(benchmark, "run_remote", fake_run)
    plan = benchmark.GeneralizationPlan(
        benchmark.SeedPartition("train", (0,)),
        benchmark.SeedPartition("validation", (8000,)),
        benchmark.SeedPartition("test", (9000,)),
    )
    report = benchmark.evaluate_generalization_remote(plan, tmp_path, "http://sim", observation_mode="noisy")
    assert all(request["observation_mode"] == "noisy" for request in requested)
    assert report["observation_mode"] == "noisy"
    assert all(episode["observation_mode"] == "noisy" for episode in report["episode_results"])
    manifest = json.loads((tmp_path / report["experiment_manifest"]).read_text())
    assert manifest["observation_mode"] == "noisy"


def test_parallel_scaling_records_each_requested_worker_level(monkeypatch, tmp_path):
    calls = []
    def fake_benchmark(runs, seed_start, output, base_url, provider_name, concurrency):
        calls.append((runs, seed_start, output.name, base_url, provider_name, concurrency))
        return {"runs": runs, "simulation_steps_per_second": concurrency * 10, "mean_control_elapsed_ms": 4, "mean_step_latency_ms": 2}
    monkeypatch.setattr(benchmark, "benchmark_remote", fake_benchmark)
    report = benchmark.benchmark_parallel_scaling(3, 20, tmp_path, "http://sim", worker_counts=(1, 8, 64))
    assert [level["workers"] for level in report["levels"]] == [1, 8, 64]
    assert [call[-1] for call in calls] == [1, 8, 64]
    assert (tmp_path / "parallel_scaling.json").exists()
