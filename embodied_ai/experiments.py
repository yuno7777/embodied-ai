"""Immutable, portable metadata for reproducible embodied experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .paths import ROOT
from .schemas import validate_trajectory_record


EXPERIMENT_MANIFEST_VERSION = 2


def source_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: int = EXPERIMENT_MANIFEST_VERSION
    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    engine_version: str = "rust-v1"
    action_mode: str = "typed_action_v1"
    source_revision: str | None = Field(default_factory=source_revision)
    scenario_id: str
    scenario_version: int | None = None
    seed: int = Field(ge=0)
    provider: str
    model: str | None = None
    observation_mode: str
    memory_mode: str
    memory_window: int = Field(ge=1, le=100)
    max_steps: int | None = Field(default=None, ge=1)
    max_wall_seconds: float | None = Field(default=None, gt=0)
    max_total_tokens: int | None = Field(default=None, ge=1)
    generator_version: int | None = Field(default=None, ge=1)
    generated_world: dict[str, object] | None = None
    world_distribution: dict[str, tuple[int, ...]] | None = None
    reward_config: dict[str, int] | None = None
    dataset_version: str | None = None
    agent_config: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_world_distribution(self) -> "ExperimentManifest":
        if self.world_distribution is None:
            return self
        expected = {"train", "validation", "test"}
        if set(self.world_distribution) != expected:
            raise ValueError("world_distribution must contain train, validation, and test")
        seeds = [seed for partition in self.world_distribution.values() for seed in partition]
        if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("world_distribution seeds must be non-negative and disjoint")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at", "experiment_id"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def persist(self, output_directory: Path) -> Path:
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / f"{self.experiment_id}.experiment.json"
        body = self.model_dump(mode="json") | {"fingerprint": self.fingerprint()}
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def reconcile_manifest_provenance(
    manifest: ExperimentManifest, provenance: dict[str, Any] | None,
) -> ExperimentManifest:
    """Pin an experiment manifest to identity returned by the Rust authority.

    A caller may supply defaults in order to start a run, but replay/create
    metadata is the evidence for the episode that actually exists.  Revalidate
    the complete result rather than using ``model_copy`` so malformed metadata
    cannot become a persisted manifest.
    """
    if provenance is None:
        return manifest
    required = {
        "scenario_id", "scenario_version", "seed", "observation_mode",
        "world_manifest", "reward_config",
    }
    if set(provenance) != required:
        raise ValueError("authoritative run provenance has unexpected fields")
    world_manifest = provenance["world_manifest"]
    generator_version = (
        world_manifest.get("generator_version")
        if isinstance(world_manifest, dict) else None
    )
    payload = manifest.model_dump(mode="json") | {
        "scenario_id": provenance["scenario_id"],
        "scenario_version": provenance["scenario_version"],
        "seed": provenance["seed"],
        "observation_mode": provenance["observation_mode"],
        "generated_world": world_manifest,
        "generator_version": generator_version if type(generator_version) is int else None,
        "reward_config": provenance["reward_config"],
    }
    return ExperimentManifest.model_validate(payload)


def load_experiment_manifest(path: Path) -> ExperimentManifest:
    """Load a persisted manifest only when its immutable fingerprint verifies."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment manifest must be a JSON object")
    fingerprint = payload.pop("fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("experiment manifest requires a fingerprint")
    manifest = ExperimentManifest.model_validate(payload)
    if manifest.fingerprint() != fingerprint:
        raise ValueError("experiment manifest fingerprint does not match its contents")
    return manifest


def audit_experiment_manifest(
    manifest_path: Path, records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify manifest integrity and, optionally, trajectory provenance.

    This is intentionally local and read-only.  A trajectory is accepted only
    when every row identifies the exact manifest experiment, sensor condition,
    and generated world recorded for the rollout.
    """
    manifest = load_experiment_manifest(manifest_path)
    receipt: dict[str, Any] = {
        "valid": True,
        "experiment_id": manifest.experiment_id,
        "manifest_version": manifest.manifest_version,
        "fingerprint": manifest.fingerprint(),
        "engine_version": manifest.engine_version,
        "scenario_id": manifest.scenario_id,
        "observation_mode": manifest.observation_mode,
        "provider": manifest.provider,
        "trajectory_provenance_checked": records is not None,
    }
    if records is None:
        return receipt
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("trajectory records must be JSON objects")
    for record in records:
        validate_trajectory_record(record)
        if record.get("experiment_id") != manifest.experiment_id:
            raise ValueError("trajectory experiment_id does not match its manifest")
        if record.get("observation_mode") != manifest.observation_mode:
            raise ValueError("trajectory observation_mode does not match its manifest")
        if record.get("world_manifest") != manifest.generated_world:
            raise ValueError("trajectory world_manifest does not match its manifest")
        if record.get("trajectory_schema_version") == 1:
            if record["scenario_id"] != manifest.scenario_id:
                raise ValueError("trajectory scenario_id does not match its manifest")
            if record["scenario_version"] != manifest.scenario_version:
                raise ValueError("trajectory scenario_version does not match its manifest")
            if record["seed"] != manifest.seed:
                raise ValueError("trajectory seed does not match its manifest")
            if record["provider"] != manifest.provider:
                raise ValueError("trajectory provider does not match its manifest")
            if record["model"] != manifest.model:
                raise ValueError("trajectory model does not match its manifest")
    policy_state_mode = (
        manifest.agent_config.get("policy_state_mode")
        if isinstance(manifest.agent_config, dict) else None
    )
    if policy_state_mode is not None:
        if any(record.get("policy_state_mode") != policy_state_mode for record in records):
            raise ValueError("trajectory policy_state_mode does not match its manifest")
    if manifest.reward_config is not None:
        if any(record.get("reward_config") != manifest.reward_config for record in records):
            raise ValueError("trajectory reward_config does not match its manifest")
    run_ids = {record.get("run_id") for record in records}
    if len(run_ids) > 1:
        raise ValueError("one experiment trajectory audit requires exactly one run_id")
    receipt.update({
        "trajectory_records": len(records),
        "trajectory_run_id": next(iter(run_ids), None),
        "world_manifest_checked": True,
        "scenario_provenance_checked": all(record.get("trajectory_schema_version") == 1 for record in records),
        "seed_provenance_checked": all(record.get("trajectory_schema_version") == 1 for record in records),
        "provider_provenance_checked": all(record.get("trajectory_schema_version") == 1 for record in records),
        "model_provenance_checked": all(record.get("trajectory_schema_version") == 1 for record in records),
        "policy_state_mode_checked": policy_state_mode is not None,
        "reward_config_checked": manifest.reward_config is not None,
    })
    return receipt
