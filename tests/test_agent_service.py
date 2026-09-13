from pathlib import Path

import pytest

from embodied_ai.agent_service import AgentRunManager, AgentRunRequest
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
    started = manager.start(AgentRunRequest(provider="scripted", seed=7))
    assert started["run_id"] == "authoritative-run"
    assert started["status"] in {"running", "completed"}
    assert isinstance(captured["experiment_id"], str)
    assert len(list(tmp_path.glob("*.experiment.json"))) == 1


def test_agent_service_retains_manifest_export_without_trajectory_records(tmp_path: Path):
    manifest = tmp_path / "experiment.experiment.json"
    manifest.write_text("{}", encoding="utf-8")
    manager = AgentRunManager("http://127.0.0.1:8080", tmp_path)
    manager._finish(RemoteRunResult("authoritative-run", "provider_error", 0, []), manifest)
    assert manager.status("authoritative-run")["exports"] == {"experiment_manifest": str(manifest)}


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
