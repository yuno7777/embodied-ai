import json
import sys

import pytest

from embodied_ai import cli
from embodied_ai.runner import RemoteRunResult


@pytest.mark.parametrize("command", ["run", "benchmark", "benchmark-scale", "generalize", "train-tabular", "evaluate-tabular", "generalize-tabular"])
def test_operational_cli_requires_the_rust_authority(monkeypatch, command):
    monkeypatch.setattr(sys, "argv", ["embodied-ai", command])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_legacy_python_server_command_is_not_exposed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "serve"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_run_cli_requests_a_generated_world_and_records_its_manifest(monkeypatch, tmp_path):
    captured = {}
    def fake_run(_provider, _seed, *_args, **kwargs):
        captured.update(kwargs)
        return RemoteRunResult("run-1", "timeout", 0, [])
    monkeypatch.setattr(cli, "run_remote", fake_run)
    monkeypatch.setattr(cli, "export_jsonl", lambda _records, path: (captured.setdefault("trajectory_path", path) or path))
    monkeypatch.setattr(cli, "export_parquet", lambda _records, _path: None)
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--generated-world-seed", "99", "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    assert captured["scenario_id"] is None
    assert captured["generated_world"] == {"seed": 99}
    assert captured["trajectory_path"].name == "run-1.trajectory.jsonl"
    assert list(tmp_path.glob("*.experiment.json"))


def test_run_cli_reconciles_continued_manifest_with_authoritative_provenance(monkeypatch, tmp_path):
    provenance = {"scenario_id": "other_room", "scenario_version": 8, "seed": 91, "observation_mode": "noisy", "world_manifest": {"world_hash": "fnv1a64:91", "generator_version": 1}, "reward_config": {"baseline_per_step": -2}}
    monkeypatch.setattr(cli, "run_remote", lambda *_args, **_kwargs: RemoteRunResult("existing", "escaped", 5, [], provenance=provenance))
    monkeypatch.setattr(cli, "export_jsonl", lambda _records, path: path)
    monkeypatch.setattr(cli, "export_parquet", lambda _records, _path: None)
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--resume-run-id", "existing", "--seed", "7", "--observation-mode", "normal", "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    manifest = json.loads(next(tmp_path.glob("*.experiment.json")).read_text(encoding="utf-8"))
    assert manifest["scenario_id"] == "other_room" and manifest["scenario_version"] == 8
    assert manifest["seed"] == 91 and manifest["observation_mode"] == "noisy"
    assert manifest["generated_world"] == provenance["world_manifest"]
    assert manifest["generator_version"] == 1 and manifest["reward_config"] == {"baseline_per_step": -2}


def test_run_cli_forwards_a_generator_config_file(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "run_remote", lambda *_args, **kwargs: (captured.update(kwargs) or RemoteRunResult("run-1", "timeout", 0, [])))
    monkeypatch.setattr(cli, "export_jsonl", lambda _records, path: path)
    monkeypatch.setattr(cli, "export_parquet", lambda _records, _path: None)
    config = tmp_path / "generator.json"; config.write_text('{"min_width":9,"max_width":9}')
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--generated-world-seed", "99", "--generator-config", str(config), "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    assert captured["generated_world"] == {"seed": 99, "config": {"min_width": 9, "max_width": 9}}


def test_run_cli_forwards_and_pins_a_complete_reward_profile(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "run_remote", lambda *_args, **kwargs: (captured.update(kwargs) or RemoteRunResult("run-1", "timeout", 0, [])))
    monkeypatch.setattr(cli, "export_jsonl", lambda _records, path: path)
    monkeypatch.setattr(cli, "export_parquet", lambda _records, _path: None)
    rewards = {"baseline_per_step": -2, "discovery_bonus": 3, "invalid_action_penalty": -4, "terminal_success": 50, "terminal_failure": -60}
    config = tmp_path / "rewards.json"; config.write_text(json.dumps(rewards), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--reward-config", str(config), "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    manifest = json.loads(next(tmp_path.glob("*.experiment.json")).read_text())
    assert captured["reward_config"] == rewards
    assert manifest["reward_config"] == rewards


@pytest.mark.parametrize("content", ['{"baseline_per_step": -1}', 'null', '[]', 'true'])
def test_run_cli_rejects_an_incomplete_reward_profile(monkeypatch, tmp_path, content):
    config = tmp_path / "rewards.json"; config.write_text(content, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--reward-config", str(config), "--server-url", "http://sim"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_run_cli_requires_an_explicit_flag_for_privileged_research_snapshots(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "run_remote", lambda *_args, **kwargs: (captured.update(kwargs) or RemoteRunResult("run-1", "timeout", 0, [])))
    monkeypatch.setattr(cli, "export_jsonl", lambda _records, path: path)
    monkeypatch.setattr(cli, "export_parquet", lambda _records, _path: None)
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--include-research-snapshots", "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    assert captured["include_research_snapshots"] is True


def test_run_cli_records_an_explicit_policy_state_mode(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "run_remote", lambda *_args, **kwargs: (captured.update(kwargs) or RemoteRunResult("run-1", "timeout", 0, [])))
    monkeypatch.setattr(cli, "export_jsonl", lambda _records, path: path)
    monkeypatch.setattr(cli, "export_parquet", lambda _records, _path: None)
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--policy-state-mode", "preserve", "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    manifest = __import__("json").loads(next(tmp_path.glob("*.experiment.json")).read_text())
    assert captured["policy_state_mode"] == "preserve"
    assert manifest["agent_config"]["policy_state_mode"] == "preserve"


def test_audit_dataset_cli_reports_local_transition_coverage(monkeypatch, tmp_path, capsys):
    trajectory = tmp_path / "run.jsonl"
    trajectory.write_text('{"run_id":"r","step":1,"chosen_action":{"type":"wait"},"next_observation":{}}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "audit-dataset", "--trajectory", str(trajectory)])
    cli.main()
    assert '"transitions": 1' in capsys.readouterr().out


def test_audit_dataset_cli_reports_a_missing_trajectory_as_a_usage_error(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "audit-dataset", "--trajectory", str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_audit_experiment_cli_verifies_manifest_and_trajectory(monkeypatch, tmp_path, capsys):
    manifest = cli.ExperimentManifest(
        experiment_id="experiment", scenario_id="survival_room", seed=42, provider="scripted",
        observation_mode="normal", memory_mode="none", memory_window=1,
    )
    manifest_path = manifest.persist(tmp_path)
    trajectory = tmp_path / "run.jsonl"
    trajectory.write_text('{"experiment_id":"experiment","run_id":"run-1","observation_mode":"normal","world_manifest":null}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "audit-experiment", "--manifest", str(manifest_path), "--trajectory", str(trajectory)])
    cli.main()
    receipt = capsys.readouterr().out
    assert '"valid": true' in receipt and '"trajectory_provenance_checked": true' in receipt


def test_audit_generalization_cli_validates_a_single_report(monkeypatch, tmp_path, capsys):
    rows = [
        {"partition": "train", "seed": 1, "outcome": "escaped", "observation_mode": "normal", "generator_config": None},
        {"partition": "validation", "seed": 2, "outcome": "timeout", "observation_mode": "normal", "generator_config": None},
        {"partition": "test", "seed": 3, "outcome": "timeout", "observation_mode": "normal", "generator_config": None},
    ]
    report = cli.summarize_generalization(rows) | {
        "engine_version": "rust-v1", "observation_mode": "normal",
        "world_distribution": {"train": [1], "validation": [2], "test": [3]},
        "generator_config": None, "generator_configs_by_partition": None,
        "episode_results": rows,
    }
    path = tmp_path / "generalization_report.json"
    path.write_text(__import__("json").dumps(report), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "audit-generalization", "--report", str(path)])
    cli.main()
    receipt = capsys.readouterr().out
    assert '"valid": true' in receipt and '"episode_provenance_checked": true' in receipt


def test_audit_generalization_cli_verifies_a_tabular_checkpoint(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / 'policy.json'; checkpoint.write_text('checkpoint', encoding='utf-8')
    rows = [
        {"partition": "train", "seed": 1, "outcome": "escaped", "observation_mode": "normal", "generator_config": None},
        {"partition": "validation", "seed": 2, "outcome": "timeout", "observation_mode": "normal", "generator_config": None},
        {"partition": "test", "seed": 3, "outcome": "timeout", "observation_mode": "normal", "generator_config": None},
    ]
    source = cli.summarize_generalization(rows) | {"engine_version": "rust-v1", "observation_mode": "normal", "world_distribution": {"train": [1], "validation": [2], "test": [3]}, "generator_config": None, "generator_configs_by_partition": None, "episode_results": rows, "checkpoint_fingerprint": cli.checkpoint_fingerprint(checkpoint)}
    report_path = tmp_path / 'report.json'; report_path.write_text(json.dumps(source), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['embodied-ai', 'audit-generalization', '--report', str(report_path), '--checkpoint', str(checkpoint)])
    cli.main()
    assert '"checkpoint_fingerprint_checked": true' in capsys.readouterr().out
    checkpoint.write_text('changed', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['embodied-ai', 'audit-generalization', '--report', str(report_path), '--checkpoint', str(checkpoint)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_generalize_cli_defaults_match_authoritative_seed_partitions(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        cli,
        "evaluate_generalization_remote",
        lambda plan, *_args, **_kwargs: (captured.update(plan.as_dict()) or {"ok": True}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodied-ai", "generalize", "--server-url", "http://sim", "--output", str(tmp_path)],
    )
    cli.main()
    assert captured == {"train": [0, 1, 2, 3, 4], "validation": [8000, 8001], "test": [9000, 9001]}


def test_generalize_cli_forwards_provider_model(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "evaluate_generalization_remote", lambda *_args, **kwargs: (captured.update(kwargs) or {"ok": True}))
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "generalize", "--provider", "gemini", "--model", "gemini-test-model", "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    assert captured["model"] == "gemini-test-model"


def test_generalize_cli_rejects_a_model_for_non_model_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "generalize", "--provider", "scripted", "--model", "not-applicable", "--server-url", "http://sim", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_generalize_cli_forwards_per_partition_generator_configs(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        cli,
        "evaluate_generalization_remote",
        lambda _plan, *_args, **kwargs: (captured.update(kwargs) or {"ok": True}),
    )
    paths = {}
    for name, rooms in {"train": 2, "validation": 3, "test": 3}.items():
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"min_rooms": {rooms}, "max_rooms": {rooms}}}')
        paths[name] = path
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodied-ai", "generalize", "--server-url", "http://sim", "--output", str(tmp_path),
         "--train-generator-config", str(paths["train"]),
         "--validation-generator-config", str(paths["validation"]),
         "--test-generator-config", str(paths["test"])],
    )
    cli.main()
    assert captured["generator_configs_by_partition"] == {
        "train": {"min_rooms": 2, "max_rooms": 2},
        "validation": {"min_rooms": 3, "max_rooms": 3},
        "test": {"min_rooms": 3, "max_rooms": 3},
    }


def test_generalize_cli_forwards_a_nondefault_observation_mode(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        cli,
        "evaluate_generalization_remote",
        lambda _plan, *_args, **kwargs: (captured.update(kwargs) or {"ok": True}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodied-ai", "generalize", "--observation-mode", "noisy", "--server-url", "http://sim", "--output", str(tmp_path)],
    )
    cli.main()
    assert captured["observation_mode"] == "noisy"


def test_tabular_cli_forwards_sensor_mode_to_the_authoritative_environment(monkeypatch, tmp_path):
    captured = {}
    def fake_train(factory, _policy, _seeds, _max_steps, options):
        captured["train"] = factory().config.observation_mode
        captured["train_world"] = options(5)["generated_world"]
        return [{"total_reward": 1, "terminal_reason": "escaped"}]
    monkeypatch.setattr(cli, "train_tabular_q", fake_train)
    checkpoint = tmp_path / "tabular.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodied-ai", "train-tabular", "--observation-mode", "noisy", "--generator-config", str(tmp_path / "config.json"), "--checkpoint", str(checkpoint), "--server-url", "http://sim"],
    )
    (tmp_path / "config.json").write_text('{"min_rooms": 2, "max_rooms": 2}')
    cli.main()
    assert captured["train"] == "noisy"
    assert captured["train_world"] == {"seed": 5, "config": {"min_rooms": 2, "max_rooms": 2}}
    assert len(list(tmp_path.glob("*.experiment.json"))) == 1

    def fake_evaluate(factory, _policy, _seeds, _max_steps, options):
        captured["evaluate"] = factory().config.observation_mode
        captured["evaluate_world"] = options(9000)["generated_world"]
        return [{"total_reward": 1, "terminal_reason": "escaped"}]
    monkeypatch.setattr(cli, "evaluate_tabular_q", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodied-ai", "evaluate-tabular", "--observation-mode", "noisy", "--generator-config", str(tmp_path / "config.json"), "--checkpoint", str(checkpoint), "--server-url", "http://sim"],
    )
    cli.main()
    assert captured["evaluate"] == "noisy"
    assert captured["evaluate_world"] == {"seed": 9000, "config": {"min_rooms": 2, "max_rooms": 2}}
    assert len(list(tmp_path.glob("*.experiment.json"))) == 2


def test_tabular_generalization_cli_trains_once_and_reports_disjoint_partitions(monkeypatch, tmp_path):
    captured = {}
    def fake_train(_factory, _policy, seeds, _steps, options): captured['train'] = (seeds, options(seeds[0])); return []
    def fake_evaluate(_factory, _policy, partitions, _steps, options):
        captured['partitions'] = partitions; captured['test_options'] = options('test', 9000)
        return {name: [{'seed': seeds[0], 'steps': 1, 'total_reward': 1, 'terminal_reason': 'escaped'}] for name, seeds in partitions.items()}
    monkeypatch.setattr(cli, 'train_tabular_q', fake_train); monkeypatch.setattr(cli, 'evaluate_tabular_partitions', fake_evaluate)
    config = tmp_path / 'generator.json'; config.write_text('{"min_rooms":2,"max_rooms":2}')
    monkeypatch.setattr(sys, 'argv', ['embodied-ai','generalize-tabular','--train-count','1','--validation-count','1','--test-count','1','--generator-config',str(config),'--checkpoint',str(tmp_path/'policy.json'),'--output',str(tmp_path/'report'),'--server-url','http://sim'])
    cli.main()
    assert captured['train'] == ([0], {'generated_world': {'seed': 0, 'config': {'min_rooms': 2, 'max_rooms': 2}}})
    assert captured['partitions'] == {'train': [0], 'validation': [8000], 'test': [9000]}
    assert captured['test_options']['generated_world']['seed'] == 9000
    assert (tmp_path/'report'/'tabular_generalization_report.json').exists()
    assert '"success_rate": 1.0' in (tmp_path/'report'/'tabular_generalization_report.json').read_text()
    report = __import__('json').loads((tmp_path/'report'/'tabular_generalization_report.json').read_text())
    assert report['engine_version'] == 'rust-v1' and report['generator_config'] == {'min_rooms': 2, 'max_rooms': 2}
    assert report['generator_config_fingerprint'] == cli.generator_config_fingerprint({'min_rooms': 2, 'max_rooms': 2})
    assert report['checkpoint_fingerprint'].startswith('sha256:')
    assert cli.audit_generalization_report(report)['valid'] is True


def test_tabular_generalization_cli_exports_actual_generated_world_evidence(monkeypatch, tmp_path):
    validation = {"geometry_valid": True, "spawn_valid": True, "required_key_reachable": True, "exit_reachable": True, "solvable": True}
    world = {"scenario": {"hazards": [{"kind": "fire"}], "rooms": [{}, {}, {}]}, "validation": validation}
    monkeypatch.setattr(cli, 'train_tabular_q', lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, 'evaluate_tabular_partitions', lambda *_args, **_kwargs: {name: [{'seed': seed, 'steps': 1, 'total_reward': 1, 'terminal_reason': 'escaped', 'world_manifest': world}] for name, seed in {'train': 0, 'validation': 8000, 'test': 9000}.items()})
    monkeypatch.setattr(sys, 'argv', ['embodied-ai', 'generalize-tabular', '--train-count', '1', '--validation-count', '1', '--test-count', '1', '--checkpoint', str(tmp_path / 'policy.json'), '--output', str(tmp_path / 'report'), '--server-url', 'http://sim'])
    cli.main()
    report = json.loads((tmp_path / 'report' / 'tabular_generalization_report.json').read_text())
    episode = report['episode_results'][0]
    assert episode['hazard_kinds'] == ['fire'] and episode['room_count'] == 3
    assert episode['world_validation'] == validation
    assert report['partitions']['train']['world_validation']['all_manifested_solvable'] is True
