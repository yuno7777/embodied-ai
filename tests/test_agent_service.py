from pathlib import Path

import pytest

from embodied_ai.agent_service import AgentRunManager, AgentRunRequest
from embodied_ai.experiments import load_experiment_manifest
from embodied_ai.runner import RemoteRunResult


def test_agent_service_validates_browser_requests_strictly():
    assert AgentRunRequest.model_validate({"provider": "scripted", "seed": 4}).provider == "scripted"
    with pytest.raises(Exception):
        AgentRunRequest.model_validate({"provider": "scripted", "unexpected": True})
    with pytest.raises(Exception):
        AgentRunRequest.model_validate({"provider": "not-a-provider"})


def test_agent_service_starts_a_provider_only_after_the_rust_run_exists(monkeypatch, tmp_path: Path):
    captured = {}
    def fake_run_remote(_provider, _seed, _url, _memory, _max_steps, _mode, on_created, _wall, _tokens, memory_window, **kwargs):
        assert memory_window == 5
        captured.update(kwargs)
        on_created("authoritative-run")
        return RemoteRunResult("authoritative-run", "escaped", 3, [])

    monkeypatch.setattr("embodied_ai.agent_service.run_remote", fake_run_remote)
    manager = AgentRunManager("http://127.0.0.1:8080", tmp_path)
    started = manager.start(AgentRunRequest(provider="scripted", seed=7, policy_state_mode="preserve"))
    assert started["run_id"] == "authoritative-run"
    assert started["status"] in {"running", "completed"}
    assert isinstance(captured["experiment_id"], str)
    assert captured["policy_state_mode"] == "preserve"
    assert len(list(tmp_path.glob("*.experiment.json"))) == 1


def test_agent_service_reconciles_manifest_from_rust_run_provenance(monkeypatch, tmp_path: Path):
    def fake_run_remote(_provider, _seed, _url, _memory, _max_steps, _mode, on_created, _wall, _tokens, memory_window, **_kwargs):
        on_created("authoritative-run")
        return RemoteRunResult("authoritative-run", "escaped", 3, [], provenance={"scenario_id": "other_room", "scenario_version": 8, "seed": 91, "observation_mode": "noisy", "world_manifest": {"generator_version": 1}, "reward_config": {"baseline_per_step": -2}})

    monkeypatch.setattr("embodied_ai.agent_service.run_remote", fake_run_remote)
    manager = AgentRunManager("http://127.0.0.1:8080", tmp_path)
    started = manager.start(AgentRunRequest(provider="scripted", seed=7))
    while started["status"] == "running":
        import time
        time.sleep(.01)
        started = manager.status("authoritative-run")
    manifest_path = Path(started["exports"]["experiment_manifest"])
    manifest = load_experiment_manifest(manifest_path)
    assert manifest.scenario_id == "other_room" and manifest.scenario_version == 8
    assert manifest.seed == 91 and manifest.observation_mode == "noisy"


def test_agent_service_retains_manifest_export_without_trajectory_records(tmp_path: Path):
    manifest = tmp_path / "experiment.experiment.json"
    manifest.write_text("{}", encoding="utf-8")
    manager = AgentRunManager("http://127.0.0.1:8080", tmp_path)
    manager._finish(RemoteRunResult("authoritative-run", "provider_error", 0, []), manifest)
    assert manager.status("authoritative-run")["exports"] == {"experiment_manifest": str(manifest)}


def test_agent_service_retains_manifest_alongside_trajectory_exports(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "experiment.experiment.json"
    manifest.write_text("{}", encoding="utf-8")
    jsonl = tmp_path / "authoritative-run.trajectory.jsonl"
    received_paths = []
    monkeypatch.setattr("embodied_ai.agent_service.export_jsonl", lambda _records, path: (received_paths.append(path) or jsonl))
    monkeypatch.setattr("embodied_ai.agent_service.export_parquet", lambda _records, _path: _path)
    record = {"observation": {}, "next_observation": {}, "agent_context": {}, "agent_metadata": None, "events": [], "chosen_action": {"type": "wait"}, "metrics": {}}
    manager = AgentRunManager("http://127.0.0.1:8080", tmp_path)
    manager._finish(RemoteRunResult("authoritative-run", "escaped", 1, [record]), manifest)
    assert manager.status("authoritative-run")["exports"] == {
        "experiment_manifest": str(manifest),
        "jsonl": str(jsonl),
        "parquet": str(tmp_path / "authoritative-run.parquet"),
    }
    assert received_paths == [jsonl]


def test_agent_export_downloads_cannot_escape_the_configured_output_directory(tmp_path: Path):
    output = tmp_path / "exports"
    output.mkdir()
    trajectory = output / "run.jsonl"
    trajectory.write_text("{}\n", encoding="utf-8")
    manager = AgentRunManager("http://127.0.0.1:8080", output)
    manager._runs["safe"] = {"exports": {"jsonl": str(trajectory)}}
    assert manager.export_path("safe", "jsonl") == trajectory.resolve()

    manifest = output / "run.experiment.json"
    manifest.write_text("{}", encoding="utf-8")
    manager._runs["manifest"] = {"exports": {"experiment_manifest": str(manifest)}}
    assert manager.export_path("manifest", "experiment_manifest") == manifest.resolve()

    outside = tmp_path / "secret.jsonl"
    outside.write_text("not an export", encoding="utf-8")
    manager._runs["unsafe"] = {"exports": {"jsonl": str(outside)}}
    with pytest.raises(KeyError):
        manager.export_path("unsafe", "jsonl")


def test_agent_request_validates_lightweight_runtime_budgets():
    request = AgentRunRequest(max_wall_seconds=2.5, max_total_tokens=500)
    assert request.max_wall_seconds == 2.5 and request.max_total_tokens == 500
    with pytest.raises(Exception):
        AgentRunRequest(max_wall_seconds=0)
    with pytest.raises(Exception):
        AgentRunRequest(max_total_tokens=0)


def test_agent_request_accepts_explicit_sensor_ablation_modes():
    assert AgentRunRequest(observation_mode="noisy").observation_mode == "noisy"
    assert AgentRunRequest(observation_mode="oracle").observation_mode == "oracle"


def test_agent_request_makes_policy_state_lifecycle_explicit():
    assert AgentRunRequest().policy_state_mode == "reset"
    assert AgentRunRequest(policy_state_mode="preserve").policy_state_mode == "preserve"
    with pytest.raises(Exception):
        AgentRunRequest(policy_state_mode="implicit")
