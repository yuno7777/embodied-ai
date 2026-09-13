from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd

def export_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, default=str) for row in records) + ("\n" if records else ""), encoding="utf-8")
    return path

def export_parquet(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)
    return path

def export_csv(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return path

def world_model_transitions(records: list[dict[str, Any]], *, include_privileged_state: bool = False) -> list[dict[str, Any]]:
    """Build ordered `(observation, action, next_observation)` examples.

    The default is safe for ordinary policy trajectories: it exports only exact
    policy observations. Privileged snapshots are included solely when records
    explicitly contain them and a researcher opts in.
    """
    ordered = sorted(records, key=lambda record: (record.get("run_id", ""), record.get("step", 0)))
    provenance_by_run: dict[str, str] = {}
    for record in ordered:
        run_id = str(record.get("run_id", ""))
        provenance = json.dumps(
            {"experiment_id": record.get("experiment_id"), "world_manifest": record.get("world_manifest")},
            sort_keys=True,
            default=str,
        )
        previous = provenance_by_run.setdefault(run_id, provenance)
        if previous != provenance:
            raise ValueError("one run cannot contain mixed experiment or world-manifest provenance")
    transitions: list[dict[str, Any]] = []
    for index, record in enumerate(ordered):
        next_record = ordered[index + 1] if index + 1 < len(ordered) and ordered[index + 1].get("run_id") == record.get("run_id") else None
        transition = {
            "dataset_schema_version": record.get("dataset_schema_version", 1),
            "experiment_id": record.get("experiment_id"),
            "world_manifest": record.get("world_manifest"),
            "run_id": record.get("run_id"),
            "step": record.get("step"),
            "observation_t": record.get("observation"),
            "action_t": record.get("chosen_action"),
            "observation_t_plus_1": (next_record or record).get("observation"),
            "reward_t": record.get("reward"),
            "terminated_t": bool(record.get("done")) and record.get("terminal_reason") not in {"timeout", "time_limit", "client_timeout", "token_budget_exhausted"},
            "truncated_t": bool(record.get("done")) and record.get("terminal_reason") in {"timeout", "time_limit", "client_timeout", "token_budget_exhausted"},
        }
        if include_privileged_state:
            transition["privileged_state_t"] = record.get("research_snapshot")
            transition["privileged_state_t_plus_1"] = (next_record or record).get("research_snapshot")
        transitions.append(transition)
    return transitions
