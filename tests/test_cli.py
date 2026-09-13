import sys

import pytest

from embodied_ai import cli


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
