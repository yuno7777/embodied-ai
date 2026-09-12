"""Local, deterministic trajectory quality checks and compact research summaries."""
from __future__ import annotations

from collections import Counter
from statistics import fmean
from typing import Any
import json
import hashlib
from pathlib import Path


class TrajectoryValidationError(ValueError):
    """Raised when an exported trajectory is incomplete or internally inconsistent."""


def validate_trajectory(records: list[dict[str, Any]]) -> None:
    if not records:
        raise TrajectoryValidationError("Trajectory has no decision records.")
    run_ids = {record.get("run_id") for record in records}
    if len(run_ids) != 1 or None in run_ids:
        raise TrajectoryValidationError("Trajectory must contain one non-empty run_id.")
    steps = [record.get("step") for record in records]
    if any(not isinstance(step, int) or step < 1 for step in steps):
        raise TrajectoryValidationError("Each record needs a positive integer step.")
    if steps != list(range(1, len(records) + 1)):
        raise TrajectoryValidationError("Trajectory steps must be contiguous and start at 1.")
    if any(not isinstance(record.get("chosen_action"), dict) for record in records):
        raise TrajectoryValidationError("Each record needs a structured chosen_action.")
    if any(not isinstance(record["chosen_action"].get("type"), str) for record in records):
        raise TrajectoryValidationError("Each chosen_action needs a string type.")
    if any(not isinstance(record.get("events"), list) for record in records):
        raise TrajectoryValidationError("Each record needs an events list.")
    if any(bool(record.get("done")) for record in records[:-1]):
        raise TrajectoryValidationError("Only the final trajectory record may be terminal.")
    for record in records:
        terminal_reason = record.get("terminal_reason")
        if bool(record.get("done")) and not isinstance(terminal_reason, str):
            raise TrajectoryValidationError("Terminal records need a terminal_reason.")
        if not bool(record.get("done")) and terminal_reason is not None:
            raise TrajectoryValidationError("Non-terminal records cannot have a terminal_reason.")


def summarize_trajectory(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce an analysis-ready, non-speculative summary from exact exported records."""
    validate_trajectory(records)
    final = records[-1]
    action_counts = Counter(record["chosen_action"].get("type", "unknown") for record in records)
    event_counts = Counter(
        event.get("type", "unknown")
        for record in records
        for event in record["events"]
        if isinstance(event, dict)
    )
    latencies = [record["latency_ms"] for record in records if isinstance(record.get("latency_ms"), (int, float))]
    step_latencies = [record["step_latency_ms"] for record in records if isinstance(record.get("step_latency_ms"), (int, float))]
    return {
        "run_id": final["run_id"],
        "scenario_id": final.get("scenario_id"),
        "scenario_version": final.get("scenario_version"),
        "provider": final.get("provider"),
        "model": final.get("model"),
        "steps": len(records),
        "outcome": final.get("terminal_reason"),
        "terminal": bool(final.get("done")),
        "valid_action_rate": sum(bool(record.get("action_valid")) for record in records) / len(records),
        "action_counts": dict(sorted(action_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "mean_provider_latency_ms": fmean(latencies) if latencies else None,
        "mean_step_latency_ms": fmean(step_latencies) if step_latencies else None,
        "control_elapsed_ms": final.get("control_elapsed_ms"),
        "final_metrics": final.get("metrics", {}),
    }

def filter_trajectory(records: list[dict[str, Any]], *, action_type: str | None = None, valid_only: bool = False, event_type: str | None = None) -> list[dict[str, Any]]:
    return [
        record for record in records
        if (action_type is None or record.get("chosen_action",{}).get("type") == action_type)
        and (not valid_only or record.get("action_valid") is True)
        and (event_type is None or any(event.get("type") == event_type for event in record.get("events",[]) if isinstance(event,dict)))
    ]

def verify_replay(replay: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(replay, dict):
        raise TrajectoryValidationError("Replay must be an object.")
    timeline=replay.get("timeline")
    observations=replay.get("observations")
    snapshot=replay.get("snapshot")
    if not isinstance(timeline,list) or not timeline: raise TrajectoryValidationError("Replay timeline is missing or empty.")
    if not isinstance(observations,list) or len(observations) != len(timeline): raise TrajectoryValidationError("Replay timeline and observations must have equal lengths.")
    if any(not isinstance(frame, dict) for frame in timeline + observations):
        raise TrajectoryValidationError("Replay frames and observations must be objects.")
    if not isinstance(snapshot,dict) or not isinstance(snapshot.get("run_id"), str) or not snapshot["run_id"]:
        raise TrajectoryValidationError("Replay snapshot requires a non-empty run ID.")
    if any(frame.get("run_id") != snapshot["run_id"] for frame in timeline):
        raise TrajectoryValidationError("Replay timeline contains mismatched run IDs.")
    steps=[frame.get("step") for frame in timeline]
    if any(type(step) is not int or step < 0 for step in steps) or steps[0] != 0:
        raise TrajectoryValidationError("Replay steps must be non-negative integers starting at zero.")
    for index, (before, after) in enumerate(zip(steps, steps[1:]), 1):
        # An external interruption records a terminal snapshot without advancing simulation time.
        interrupted = after == before and index == len(steps) - 1 and timeline[index].get("done") is True and timeline[index - 1].get("done") is not True
        if after != before + 1 and not interrupted:
            raise TrajectoryValidationError("Replay timeline steps must be contiguous.")
    if snapshot != timeline[-1]:
        raise TrajectoryValidationError("Replay snapshot does not match the final frame.")
    for frame, observation in zip(timeline, observations):
        if "step" in observation and observation["step"] != frame["step"]:
            raise TrajectoryValidationError("Replay observation step does not match its frame.")
        if "run_id" in observation and observation["run_id"] != snapshot["run_id"]:
            raise TrajectoryValidationError("Replay observation run ID does not match.")
    for field in ("events", "decisions"):
        if not isinstance(replay.get(field, []), list) or any(not isinstance(item, dict) for item in replay.get(field, [])):
            raise TrajectoryValidationError(f"Replay {field} must be an array of objects.")
    return {"valid":True,"run_id":snapshot.get("run_id"),"frames":len(timeline),"events":len(replay.get("events",[])),"decisions":len(replay.get("decisions",[]))}

def replay_fingerprint(replay: dict[str, Any]) -> str:
    """Hash deterministic replay content while excluding generated IDs and wall-clock timestamps."""
    verify_replay(replay)
    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: stable(item) for key, item in sorted(value.items()) if key not in {"run_id", "event_id", "timestamp"}}
        if isinstance(value, list): return [stable(item) for item in value]
        return value
    # Provider timing, token usage and control durations are not deterministic world state.
    world = {key: value for key, value in replay.items() if key not in {"control", "decisions"}}
    payload=json.dumps(stable(world),sort_keys=True,separators=(",",":"),ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def check_reproducibility(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if (left.get("scenario_id"),left.get("scenario_version"),left.get("seed")) != (right.get("scenario_id"),right.get("scenario_version"),right.get("seed")):
        raise TrajectoryValidationError("Replays must use the same scenario version and seed.")
    left_hash,replay_hash=replay_fingerprint(left),replay_fingerprint(right)
    return {"reproducible":left_hash==replay_hash,"left_fingerprint":left_hash,"right_fingerprint":replay_hash,"seed":left.get("seed"),"scenario_id":left.get("scenario_id")}

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
