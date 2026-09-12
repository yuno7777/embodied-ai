"""Immutable, portable metadata for reproducible embodied experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .paths import ROOT


EXPERIMENT_MANIFEST_VERSION = 1


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
