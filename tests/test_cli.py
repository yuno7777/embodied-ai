import sys

import pytest

from embodied_ai import cli
from embodied_ai.runner import RemoteRunResult


@pytest.mark.parametrize("command", ["run", "benchmark", "benchmark-scale", "generalize", "train-tabular", "evaluate-tabular"])
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
    monkeypatch.setattr(cli, "export_jsonl", lambda _records, path: path)
    monkeypatch.setattr(cli, "export_parquet", lambda _records, _path: None)
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--generated-world-seed", "99", "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    assert captured["scenario_id"] is None
    assert captured["generated_world"] == {"seed": 99}
    assert list(tmp_path.glob("*.experiment.json"))


def test_run_cli_forwards_a_generator_config_file(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "run_remote", lambda *_args, **kwargs: (captured.update(kwargs) or RemoteRunResult("run-1", "timeout", 0, [])))
    monkeypatch.setattr(cli, "export_jsonl", lambda _records, path: path)
    monkeypatch.setattr(cli, "export_parquet", lambda _records, _path: None)
    config = tmp_path / "generator.json"; config.write_text('{"min_width":9,"max_width":9}')
    monkeypatch.setattr(sys, "argv", ["embodied-ai", "run", "--generated-world-seed", "99", "--generator-config", str(config), "--server-url", "http://sim", "--output", str(tmp_path)])
    cli.main()
    assert captured["generated_world"] == {"seed": 99, "config": {"min_width": 9, "max_width": 9}}


def test_generalize_cli_defaults_match_authoritative_seed_partitions(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        cli,
        "evaluate_generalization_remote",
        lambda plan, *_args: (captured.update(plan.as_dict()) or {"ok": True}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodied-ai", "generalize", "--server-url", "http://sim", "--output", str(tmp_path)],
    )
    cli.main()
    assert captured == {"train": [0, 1, 2, 3, 4], "validation": [8000, 8001], "test": [9000, 9001]}


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
    def fake_train(factory, _policy, _seeds, _max_steps, _options):
        captured["train"] = factory().config.observation_mode
        return [{"total_reward": 1, "terminal_reason": "escaped"}]
    monkeypatch.setattr(cli, "train_tabular_q", fake_train)
    checkpoint = tmp_path / "tabular.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodied-ai", "train-tabular", "--observation-mode", "noisy", "--checkpoint", str(checkpoint), "--server-url", "http://sim"],
    )
    cli.main()
    assert captured["train"] == "noisy"

    def fake_evaluate(factory, _policy, _seeds, _max_steps, _options):
        captured["evaluate"] = factory().config.observation_mode
        return [{"total_reward": 1, "terminal_reason": "escaped"}]
    monkeypatch.setattr(cli, "evaluate_tabular_q", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodied-ai", "evaluate-tabular", "--observation-mode", "noisy", "--checkpoint", str(checkpoint), "--server-url", "http://sim"],
    )
    cli.main()
    assert captured["evaluate"] == "noisy"
