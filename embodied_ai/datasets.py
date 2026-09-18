from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
from .schemas import TrajectoryStep, validate_trajectory_record

def export_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    for record in records:
        validate_trajectory_record(record)
    records = [
        TrajectoryStep.model_validate(record).model_dump(mode="json")
        if record.get("trajectory_schema_version") == 1 else record
        for record in records
    ]
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

    Explicit post-action observations take precedence. Legacy rows may infer a
    target only from an adjacent nonterminal transition in the same run; unknown
    targets remain None. Privileged snapshots require researcher opt-in.
    """
    for record in records:
        validate_trajectory_record(record)
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
    step_counts = Counter((record.get("run_id"), record.get("step")) for record in ordered)
    for index, record in enumerate(ordered):
        next_record = ordered[index + 1] if index + 1 < len(ordered) and ordered[index + 1].get("run_id") == record.get("run_id") else None
        step = record.get("step")
        if not (
            next_record is not None
            and isinstance(record.get("run_id"), str) and record["run_id"]
            and not record.get("done") and record.get("terminal_reason") is None
            and type(step) is int and step > 0
            and type(next_record.get("step")) is int and next_record["step"] == step + 1
            and step_counts[(record["run_id"], step)] == 1
            and step_counts[(record["run_id"], step + 1)] == 1
        ):
            next_record = None
        successor = next_record or {}
        transition = {
            "dataset_schema_version": record.get("dataset_schema_version", 1),
            "experiment_id": record.get("experiment_id"),
            "world_manifest": record.get("world_manifest"),
            "run_id": record.get("run_id"),
            "step": record.get("step"),
            "observation_t": record.get("observation"),
            "action_t": record.get("chosen_action"),
            "observation_t_plus_1": record.get("next_observation", successor.get("observation")),
            "reward_t": record.get("reward"),
            "reward_breakdown_t": record.get("reward_breakdown"),
            "terminated_t": bool(record.get("done")) and record.get("terminal_reason") not in {"timeout", "time_limit", "client_timeout", "token_budget_exhausted"},
            "truncated_t": bool(record.get("done")) and record.get("terminal_reason") in {"timeout", "time_limit", "client_timeout", "token_budget_exhausted"},
        }
        if include_privileged_state:
            transition["privileged_state_t"] = record.get("research_snapshot")
            transition["privileged_state_t_plus_1"] = record.get(
                "next_research_snapshot", successor.get("research_snapshot")
            )
        transitions.append(transition)
    return transitions


def summarize_world_model_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report lightweight completeness and coverage facts for exported transitions."""
    transitions = world_model_transitions(records)
    action_counts = Counter(
        str(record.get("chosen_action", {}).get("type", "unknown"))
        for record in records
        if isinstance(record.get("chosen_action"), dict)
    )
    terminal_reasons = Counter(
        str(record.get("terminal_reason"))
        for record in records
        if record.get("done") and record.get("terminal_reason") is not None
    )
    observation_modes = Counter(str(record.get("observation_mode", "unknown")) for record in records)
    policy_state_modes = Counter(str(record.get("policy_state_mode", "unknown")) for record in records)
    metadata_records = [record.get("agent_metadata") for record in records if isinstance(record.get("agent_metadata"), dict)]
    metadata_field_counts = Counter(
        field for metadata in metadata_records for field in metadata if isinstance(field, str)
    )
    planner_metadata_records = sum(
        isinstance(metadata.get("planner"), dict) for metadata in metadata_records
    )
    policy_state_by_run: dict[str, set[str]] = {}
    for record in records:
        run_id = record.get("run_id")
        if run_id is not None:
            policy_state_by_run.setdefault(str(run_id), set()).add(str(record.get("policy_state_mode", "unknown")))
    paired_snapshots = sum(
        "research_snapshot" in record and "next_research_snapshot" in record for record in records
    )
    partial_snapshots = sum(
        ("research_snapshot" in record) != ("next_research_snapshot" in record) for record in records
    )
    reward_breakdowns = [record["reward_breakdown"] for record in records if isinstance(record.get("reward_breakdown"), dict)]
    reward_component_totals = {
        component: sum(
            breakdown.get(component, 0)
            for breakdown in reward_breakdowns
            if isinstance(breakdown.get(component, 0), int) and not isinstance(breakdown.get(component, 0), bool)
        )
        for component in ("baseline", "progress", "invalid_action_penalty", "hazard_penalty", "terminal")
    }
    return {
        "records": len(records),
        "transitions": len(transitions),
        "transitions_missing_next_observation": sum(row["observation_t_plus_1"] is None for row in transitions),
        "runs": len({record.get("run_id") for record in records if record.get("run_id") is not None}),
        "action_counts": dict(sorted(action_counts.items())),
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "observation_modes": dict(sorted(observation_modes.items())),
        "policy_state_modes": dict(sorted(policy_state_modes.items())),
        "runs_with_mixed_policy_state_mode": sum(len(modes) > 1 for modes in policy_state_by_run.values()),
        "records_with_agent_metadata": len(metadata_records),
        "agent_metadata_field_counts": dict(sorted(metadata_field_counts.items())),
        "planner_metadata_records": planner_metadata_records,
        "records_with_exact_next_observation": sum("next_observation" in record for record in records),
        "records_with_reward_breakdown": len(reward_breakdowns),
        "reward_component_totals": reward_component_totals,
        "privileged_snapshot_pairs": paired_snapshots,
        "partial_privileged_snapshot_records": partial_snapshots,
    }
