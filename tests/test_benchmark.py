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
    report=benchmark.benchmark_remote(4,10,tmp_path,"http://sim","scripted",concurrency=2)
    assert report["concurrency"]==2
    assert [row["seed"] for row in report["seed_results"]]==[10,11,12,13]
    assert (tmp_path/"runs.csv").exists() and (tmp_path/"decisions.csv").exists()

def test_benchmark_comparison_reports_directional_deltas():
    result=benchmark.compare_benchmarks(
        {"scenario_id":"s","provider":"left","runs":2,"success_rate":0.5,"mean_score":40,"mean_steps":10},
        {"scenario_id":"s","provider":"right","runs":3,"success_rate":1.0,"mean_score":70,"mean_steps":8},
    )
    assert result["success_rate_delta"]==0.5
    assert result["mean_score_delta"]==30
    assert result["mean_steps_delta"]==-2
