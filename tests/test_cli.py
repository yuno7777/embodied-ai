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
