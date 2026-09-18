import sys

import pytest

from embodied_ai import cli, runner


@pytest.mark.parametrize("option", ["resume_run_id", "restore_replay_id"])
def test_resume_rejects_reward_override_before_connecting(monkeypatch, option):
    def unexpected_client(*args, **kwargs):
        pytest.fail("invalid reward override must not connect or mutate a run")
    monkeypatch.setattr(runner, "RustRunClient", unexpected_client)
    with pytest.raises(ValueError, match="only to new runs"):
        runner.run_remote(object(), 42, reward_config={}, **{option: "existing"})


@pytest.mark.parametrize("option", ["--resume-run-id", "--restore-replay-id"])
def test_cli_rejects_reward_override_before_loading_provider(monkeypatch, option, capsys):
    def unexpected_provider(*args, **kwargs):
        pytest.fail("invalid arguments must not initialize a provider")
    monkeypatch.setattr(cli, "provider_for", unexpected_provider)
    monkeypatch.setattr(sys, "argv", [
        "embodied-ai", "run", "--server-url", "http://sim", option, "existing",
        "--reward-config", "unread-profile.json",
    ])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "only to new runs" in capsys.readouterr().err
