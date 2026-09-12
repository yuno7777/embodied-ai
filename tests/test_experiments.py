import json

from embodied_ai.experiments import ExperimentManifest


def test_manifest_fingerprint_excludes_run_identity_and_timestamp():
    fields = dict(scenario_id="survival_room", seed=42, provider="random_valid", observation_mode="minimal", memory_mode="none", memory_window=1)
    first = ExperimentManifest(experiment_id="first", created_at="2026-01-01T00:00:00+00:00", **fields)
    second = ExperimentManifest(experiment_id="second", created_at="2026-01-02T00:00:00+00:00", **fields)
    assert first.fingerprint() == second.fingerprint()


def test_manifest_persists_reconstructible_metadata(tmp_path):
    manifest = ExperimentManifest(experiment_id="experiment", scenario_id="generated_grid", seed=9, provider="scripted", observation_mode="normal", memory_mode="recent", memory_window=4)
    path = manifest.persist(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["experiment_id"] == "experiment"
    assert saved["fingerprint"] == manifest.fingerprint()
