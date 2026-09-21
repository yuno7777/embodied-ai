import json

import pytest

from embodied_ai.experiments import ExperimentManifest, audit_experiment_manifest, load_experiment_manifest, reconcile_manifest_provenance


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


def test_manifest_captures_a_procedural_evaluation_distribution():
    manifest = ExperimentManifest(
        scenario_id="procedural", seed=0, provider="scripted", observation_mode="normal",
        memory_mode="none", memory_window=1, generator_version=1,
        generated_world={"seed": 0, "config": {"min_width": 9}},
        world_distribution={"train": (0, 1), "validation": (2,), "test": (3,)},
        reward_config={"baseline_per_step": -1}, agent_config={"policy": "scripted"},
    )
    assert manifest.manifest_version == 2
    assert manifest.world_distribution["test"] == (3,)


def test_manifest_rejects_overlapping_evaluation_distribution():
    try:
        ExperimentManifest(
            scenario_id="procedural", seed=0, provider="scripted", observation_mode="normal",
            memory_mode="none", memory_window=1,
            world_distribution={"train": (0,), "validation": (0,), "test": (2,)},
        )
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("overlapping partitions were accepted")


def test_experiment_audit_verifies_persisted_manifest_and_trajectory_provenance(tmp_path):
    world = {"seed": 9, "world_hash": "fnv1a64:verified"}
    manifest = ExperimentManifest(
        experiment_id="experiment", scenario_id="procedural", seed=9, provider="scripted",
        observation_mode="normal", memory_mode="none", memory_window=1,
        generated_world=world, agent_config={"policy_state_mode": "reset"},
    )
    path = manifest.persist(tmp_path)
    record = {
        "experiment_id": "experiment", "run_id": "run-1", "observation_mode": "normal",
        "world_manifest": world, "policy_state_mode": "reset",
    }
    assert load_experiment_manifest(path) == manifest
    receipt = audit_experiment_manifest(path, [record])
    assert receipt["valid"] is True
    assert receipt["trajectory_run_id"] == "run-1"
    assert receipt["world_manifest_checked"] is True


def test_experiment_audit_binds_versioned_rows_to_scenario_and_seed(monkeypatch, tmp_path):
    manifest = ExperimentManifest(
        experiment_id="experiment", scenario_id="survival_room", scenario_version=3, seed=9,
        provider="scripted", observation_mode="normal", memory_mode="none", memory_window=1,
    )
    path = manifest.persist(tmp_path)
    # Versioned-row structural validation has its own schema tests. Stub it
    # here so this focused test can exercise the audit's manifest binding.
    monkeypatch.setattr("embodied_ai.experiments.validate_trajectory_record", lambda _record: None)
    record = {"trajectory_schema_version": 1, "experiment_id": "experiment", "run_id": "run-1", "scenario_id": "survival_room", "scenario_version": 3, "seed": 9, "provider": "scripted", "model": None, "observation_mode": "normal", "world_manifest": None}
    receipt = audit_experiment_manifest(path, [record])
    assert receipt["scenario_provenance_checked"] is True and receipt["seed_provenance_checked"] is True
    assert receipt["provider_provenance_checked"] is True and receipt["model_provenance_checked"] is True
    record["seed"] = 10
    with pytest.raises(ValueError, match="seed"):
        audit_experiment_manifest(path, [record])
    record["seed"] = 9
    record["provider"] = "other"
    with pytest.raises(ValueError, match="provider"):
        audit_experiment_manifest(path, [record])


def test_manifest_reconciliation_revalidates_authoritative_metadata():
    manifest = ExperimentManifest(scenario_id="survival_room", seed=7, provider="scripted", observation_mode="normal", memory_mode="none", memory_window=1)
    provenance = {"scenario_id": "other_room", "scenario_version": 8, "seed": 91, "observation_mode": "noisy", "world_manifest": {"generator_version": 1}, "reward_config": {"baseline_per_step": -2}}
    reconciled = reconcile_manifest_provenance(manifest, provenance)
    assert reconciled.scenario_id == "other_room" and reconciled.seed == 91
    assert reconciled.generator_version == 1 and reconciled.reward_config == {"baseline_per_step": -2}
    with pytest.raises(ValueError, match="unexpected"):
        reconcile_manifest_provenance(manifest, {"seed": 91})


def test_experiment_audit_rejects_tampering_and_mismatched_trajectory_provenance(tmp_path):
    manifest = ExperimentManifest(
        experiment_id="experiment", scenario_id="survival_room", seed=9, provider="scripted",
        observation_mode="normal", memory_mode="none", memory_window=1,
    )
    path = manifest.persist(tmp_path)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["provider"] = "random_valid"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        audit_experiment_manifest(path)
    path = manifest.persist(tmp_path)
    with pytest.raises(ValueError, match="experiment_id"):
        audit_experiment_manifest(path, [{"experiment_id": "other", "observation_mode": "normal", "world_manifest": None}])


def test_experiment_audit_binds_a_pinned_reward_profile_to_trajectory_rows(tmp_path):
    rewards = {"baseline_per_step": -1, "discovery_bonus": 5, "invalid_action_penalty": -2, "terminal_success": 100, "terminal_failure": -100}
    manifest = ExperimentManifest(
        experiment_id="experiment", scenario_id="survival_room", seed=9, provider="scripted",
        observation_mode="normal", memory_mode="none", memory_window=1, reward_config=rewards,
    )
    path = manifest.persist(tmp_path)
    record = {"experiment_id": "experiment", "run_id": "run-1", "observation_mode": "normal", "world_manifest": None, "reward_config": rewards}
    receipt = audit_experiment_manifest(path, [record])
    assert receipt["reward_config_checked"] is True
    record["reward_config"] = {**rewards, "terminal_success": 1}
    with pytest.raises(ValueError, match="reward_config"):
        audit_experiment_manifest(path, [record])


def test_experiment_audit_revalidates_versioned_trajectory_rows(tmp_path):
    manifest = ExperimentManifest(
        experiment_id="experiment", scenario_id="survival_room", seed=9, provider="scripted",
        observation_mode="normal", memory_mode="none", memory_window=1,
    )
    path = manifest.persist(tmp_path)
    with pytest.raises(ValueError):
        audit_experiment_manifest(path, [{"trajectory_schema_version": 1, "experiment_id": "experiment"}])
