"""A declared unsupported format must never fall back to legacy handling."""
import pytest

from embodied_ai.analysis import filter_trajectory, summarize_trajectory
from embodied_ai.datasets import export_jsonl, world_model_transitions
from embodied_ai.experiments import ExperimentManifest, audit_experiment_manifest


@pytest.mark.parametrize("version", [2, 0, None, True, "1", 1.0])
def test_all_trajectory_boundaries_reject_unsupported_versions(version, tmp_path):
    manifest = ExperimentManifest(
        experiment_id="test", scenario_id="survival_room", seed=1,
        provider="scripted", observation_mode="normal", memory_mode="none", memory_window=1,
    )
    manifest_path = manifest.persist(tmp_path)
    row = {
        "trajectory_schema_version": version, "experiment_id": "test",
        "observation_mode": "normal", "world_manifest": None,
        "run_id": "r", "step": 1, "chosen_action": {"type": "wait"},
        "events": [], "done": False,
    }
    output = tmp_path / "trajectory.jsonl"
    output.write_text("existing data", encoding="utf-8")
    operations = [
        lambda: export_jsonl([row], output),
        lambda: world_model_transitions([row]),
        lambda: summarize_trajectory([row]),
        # Even a row excluded by the filter must be validated.
        lambda: filter_trajectory([row], action_type="move"),
        lambda: audit_experiment_manifest(manifest_path, [row]),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="unsupported trajectory_schema_version"):
            operation()
    assert output.read_text(encoding="utf-8") == "existing data"


def test_filter_rejects_malformed_v1_before_discarding_row():
    with pytest.raises(ValueError):
        filter_trajectory([{"trajectory_schema_version": 1}], action_type="move")


def test_filter_accepts_empty_and_legacy_subsets():
    assert filter_trajectory([]) == []
    rows = [{"step": 7, "chosen_action": {"type": "wait"}}]
    assert filter_trajectory(rows, action_type="wait") == rows
