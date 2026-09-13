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
        return RemoteRunResult(f"run-{seed}", "escaped", seed % 3 + 1, [record])
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
    assert (tmp_path/"runs.csv").exists() and (tmp_path/"decisions.csv").exists()

def test_benchmark_comparison_reports_directional_deltas():
    result=benchmark.compare_benchmarks(
        {"scenario_id":"s","provider":"left","runs":2,"success_rate":0.5,"mean_score":40,"mean_steps":10},
        {"scenario_id":"s","provider":"right","runs":3,"success_rate":1.0,"mean_score":70,"mean_steps":8},
    )
    assert result["success_rate_delta"]==0.5
    assert result["mean_score_delta"]==30
    assert result["mean_steps_delta"]==-2


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
        {"partition": "train", "outcome": "escaped", "steps": 2, "total_reward": 5, "invalid_actions": 0, "exploration_coverage": .7, "resource_efficiency": .9, "control_elapsed_ms": 10},
        {"partition": "validation", "outcome": "hazard", "steps": 4, "total_reward": -2, "invalid_actions": 1, "exploration_coverage": .4, "resource_efficiency": .2, "control_elapsed_ms": 20},
        {"partition": "test", "outcome": "hazard", "steps": 3, "total_reward": -3, "invalid_actions": 1, "exploration_coverage": .3, "resource_efficiency": .1, "control_elapsed_ms": 15},
    ]
    report = benchmark.summarize_generalization(rows)
    assert report["partitions"]["train"]["success_rate"] == 1.0
    assert report["partitions"]["test"]["success_rate"] == 0.0
    assert report["generalization_gap"]["train_minus_test_success_rate"] == 1.0
    assert len(report["partitions"]["test"]["success_rate_wilson_95"]) == 2
    assert report["partitions"]["validation"]["failure_reasons"] == {"hazard": 1}


def test_generalization_evaluation_uses_procedural_worlds_and_writes_report(monkeypatch, tmp_path):
    def fake_run(provider, seed, _base_url, **kwargs):
        expected = {"seed": seed, "config": {"min_width": 9, "max_width": 9}}
        if seed == 1:
            expected["partition"] = "train"
        assert kwargs["generated_world"] == expected
        record = {"reward": 4, "metrics": {"invalid_actions": 0, "exploration_coverage": .5, "resource_efficiency": .8}, "control_elapsed_ms": 10}
        return RemoteRunResult(f"run-{seed}", "escaped" if seed == 1 else "timeout", 2, [record])
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
