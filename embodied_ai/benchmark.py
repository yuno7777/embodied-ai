from __future__ import annotations
import hashlib
import json
import os
from math import comb
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping
from .datasets import export_csv, export_jsonl, export_parquet
from .experiments import ExperimentManifest
from .providers import CautiousProvider, ExplorerProvider, GeminiProvider, MockReasoningProvider, RandomValidProvider, ScriptedProvider
from .runner import RustRunClient, run_remote


@dataclass(frozen=True)
class SeedPartition:
    """Named, explicit world seeds used by one evaluation split."""

    name: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("partition name must be non-empty alphanumeric text")
        if not self.seeds or any(not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("partitions require one or more non-negative integer seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("partition seeds must be unique")

    @classmethod
    def from_range(cls, name: str, start: int, stop: int) -> "SeedPartition":
        """Create an explicit partition from a half-open deterministic seed range."""
        return cls(name=name, seeds=tuple(range(start, stop)))


@dataclass(frozen=True)
class GeneralizationPlan:
    """Disjoint seed splits for comparable train/validation/test evaluation."""

    train: SeedPartition
    validation: SeedPartition
    test: SeedPartition

    def __post_init__(self) -> None:
        expected = {"train", "validation", "test"}
        partitions = (self.train, self.validation, self.test)
        if {partition.name for partition in partitions} != expected:
            raise ValueError("plan partitions must be named train, validation, and test")
        all_seeds = [seed for partition in partitions for seed in partition.seeds]
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("train, validation, and test seed sets must be disjoint")

    def as_dict(self) -> dict[str, list[int]]:
        return {partition.name: list(partition.seeds) for partition in (self.train, self.validation, self.test)}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Deterministic 95% Wilson interval for a binary episode success rate."""
    if total < 1:
        return None
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return (center - margin, center + margin)


def paired_success_exact_p_value(left_only_success: int, right_only_success: int) -> float:
    """Two-sided exact binomial p-value for paired binary outcomes.

    Only discordant pairs contribute. Under the null, either policy is equally
    likely to be the successful one for each such world. This is an exact
    descriptive test for a fixed paired-seed evaluation, not a replacement for
    repeated independent studies.
    """
    if any(type(value) is not int or value < 0 for value in (left_only_success, right_only_success)):
        raise ValueError("paired success counts must be non-negative integers")
    discordant = left_only_success + right_only_success
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, successes) for successes in range(min(left_only_success, right_only_success) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def built_in_world_partition(seed: int) -> str | None:
    """Return Rust's optional default-distribution assertion for a seed."""
    if 0 <= seed <= 7_999:
        return "train"
    if 8_000 <= seed <= 8_999:
        return "validation"
    if 9_000 <= seed <= 9_999:
        return "test"
    return None


def generated_hazard_kinds(world_manifest: dict[str, Any] | None) -> list[str]:
    """Extract actual generated mechanics for lightweight report stratification."""
    scenario = world_manifest.get("scenario") if isinstance(world_manifest, dict) else None
    hazards = scenario.get("hazards") if isinstance(scenario, dict) else None
    return sorted({str(hazard.get("kind")) for hazard in hazards if isinstance(hazard, dict) and isinstance(hazard.get("kind"), str)}) if isinstance(hazards, list) else []


def generated_room_count(world_manifest: dict[str, Any] | None) -> int | None:
    """Extract the actual generated room topology without trusting requested config."""
    scenario = world_manifest.get("scenario") if isinstance(world_manifest, dict) else None
    rooms = scenario.get("rooms") if isinstance(scenario, dict) else None
    return len(rooms) if isinstance(rooms, list) else None


def generated_mechanics_signature(world_manifest: dict[str, Any] | None) -> str | None:
    """Name the joint generated mechanics actually present in a world manifest."""
    room_count = generated_room_count(world_manifest)
    hazard_kinds = generated_hazard_kinds(world_manifest)
    if room_count is None and not hazard_kinds:
        return None
    rooms = str(room_count) if room_count is not None else "unknown"
    hazards = ",".join(hazard_kinds) if hazard_kinds else "none"
    return f"rooms:{rooms}|hazards:{hazards}"


def generated_world_validation(world_manifest: dict[str, Any] | None) -> dict[str, bool] | None:
    """Extract Rust's procedural-world constraint result without inferring it."""
    validation = world_manifest.get("validation") if isinstance(world_manifest, dict) else None
    fields = ("geometry_valid", "spawn_valid", "required_key_reachable", "exit_reachable", "solvable")
    if not isinstance(validation, dict) or any(type(validation.get(field)) is not bool for field in fields):
        return None
    return {field: validation[field] for field in fields}


def world_validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only explicit Rust validation evidence from generated worlds."""
    validations = [row_validation for row in rows if (row_validation := row.get("world_validation")) is not None]
    validation_fields = ("geometry_valid", "spawn_valid", "required_key_reachable", "exit_reachable", "solvable")
    return {
        "manifested_episodes": len(validations),
        **{
            f"{field}_episodes": sum(validation.get(field) is True for validation in validations)
            for field in validation_fields
        },
        "all_manifested_solvable": all(validation["solvable"] for validation in validations) if validations else None,
    }


def generator_config_fingerprint(config: Mapping[str, Any] | None) -> str | None:
    """Produce a stable identity for a complete JSON generator configuration."""
    if config is None:
        return None
    try:
        encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("generator configuration must be JSON-serializable") from error
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def effective_provider_model(provider_name: str, model: str | None) -> str | None:
    """Resolve the model identity that will actually be used by a provider."""
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be non-empty text when supplied")
    if provider_name != "gemini":
        if model is not None:
            raise ValueError("--model is supported only for the gemini provider")
        return None
    return model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def stratified_success_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an auditable binary-success slice with the same interval as a split."""
    successes = sum(row.get("outcome") == "escaped" for row in rows)
    return {
        "episodes": len(rows),
        "success_rate": successes / len(rows),
        "success_rate_wilson_95": wilson_interval(successes, len(rows)),
    }


def summarize_generalization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize train/validation/test episodes and their held-out gap."""
    expected = {"train", "validation", "test"}
    grouped = {name: [row for row in rows if row.get("partition") == name] for name in expected}
    if {row.get("partition") for row in rows} != expected or any(not group for group in grouped.values()):
        raise ValueError("rows must contain at least one train, validation, and test episode")

    def summarize(rows_for_partition: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(row.get("outcome") == "escaped" for row in rows_for_partition)
        numeric = lambda key: [float(row[key]) for row in rows_for_partition if isinstance(row.get(key), (int, float))]
        step_counts = numeric("steps")
        rewards = numeric("total_reward")
        invalid = numeric("invalid_actions")
        coverage = numeric("exploration_coverage")
        efficiency = numeric("resource_efficiency")
        elapsed = numeric("control_elapsed_ms")
        failure_reasons = Counter(
            str(row.get("outcome") or "unknown") for row in rows_for_partition if row.get("outcome") != "escaped"
        )
        hazard_groups: dict[str, list[dict[str, Any]]] = {}
        room_groups: dict[int, list[dict[str, Any]]] = {}
        mechanics_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows_for_partition:
            for kind in row.get("hazard_kinds", []):
                if isinstance(kind, str):
                    hazard_groups.setdefault(kind, []).append(row)
            room_count = row.get("room_count")
            if isinstance(room_count, int) and room_count > 0:
                room_groups.setdefault(room_count, []).append(row)
            mechanics_signature = row.get("mechanics_signature")
            if isinstance(mechanics_signature, str) and mechanics_signature:
                mechanics_groups.setdefault(mechanics_signature, []).append(row)
        total_steps = sum(step_counts)
        return {
            "episodes": len(rows_for_partition),
            "success_rate": successes / len(rows_for_partition),
            "success_rate_wilson_95": wilson_interval(successes, len(rows_for_partition)),
            "mean_episode_reward": fmean(rewards) if rewards else None,
            "mean_episode_length": fmean(step_counts) if step_counts else None,
            "invalid_action_rate": sum(invalid) / total_steps if invalid and total_steps else None,
            "mean_exploration_coverage": fmean(coverage) if coverage else None,
            "mean_resource_efficiency": fmean(efficiency) if efficiency else None,
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "steps_per_second": total_steps / (sum(elapsed) / 1000) if elapsed and sum(elapsed) > 0 else None,
            "hazard_kind_breakdown": {
                kind: stratified_success_summary(items)
                for kind, items in sorted(hazard_groups.items())
            },
            "room_count_breakdown": {
                str(room_count): stratified_success_summary(items)
                for room_count, items in sorted(room_groups.items())
            },
            "mechanics_breakdown": {
                signature: stratified_success_summary(items)
                for signature, items in sorted(mechanics_groups.items())
            },
            "world_validation": world_validation_summary(rows_for_partition),
        }

    partitions = {name: summarize(grouped[name]) for name in sorted(expected)}
    return {
        "report_version": 1,
        "partitions": partitions,
        "generalization_gap": {
            "train_minus_validation_success_rate": partitions["train"]["success_rate"] - partitions["validation"]["success_rate"],
            "train_minus_test_success_rate": partitions["train"]["success_rate"] - partitions["test"]["success_rate"],
        },
    }


def evaluate_generalization_remote(
    plan: GeneralizationPlan,
    output: Path,
    base_url: str,
    provider_name: str = "scripted",
    concurrency: int = 1,
    model: str | None = None,
    generator_config: Mapping[str, Any] | None = None,
    generator_configs_by_partition: Mapping[str, Mapping[str, Any]] | None = None,
    observation_mode: str = "normal",
) -> dict[str, Any]:
    """Evaluate a policy over disjoint procedural worlds and persist an exact report."""
    if concurrency < 1 or concurrency > 32:
        raise ValueError("concurrency must be between 1 and 32")
    if observation_mode not in {"minimal", "normal", "rich", "noisy", "oracle"}:
        raise ValueError("unsupported observation mode")
    if generator_config is not None and generator_configs_by_partition is not None:
        raise ValueError("choose either one generator config or configs by partition")
    effective_model = effective_provider_model(provider_name, model)
    expected_partitions = {"train", "validation", "test"}
    if generator_configs_by_partition is not None:
        if set(generator_configs_by_partition) != expected_partitions:
            raise ValueError("generator configs must contain train, validation, and test")
        if any(not isinstance(config, Mapping) for config in generator_configs_by_partition.values()):
            raise ValueError("each partition generator config must be an object")
    output.mkdir(parents=True, exist_ok=True)
    partition_configs = (
        {name: dict(config) for name, config in generator_configs_by_partition.items()}
        if generator_configs_by_partition is not None
        else None
    )
    shared_config_fingerprint = generator_config_fingerprint(generator_config)
    partition_config_fingerprints = (
        {name: generator_config_fingerprint(config) for name, config in partition_configs.items()}
        if partition_configs is not None
        else None
    )
    manifest = ExperimentManifest(
        scenario_id="procedural",
        seed=plan.train.seeds[0],
        provider=provider_name,
        model=effective_model,
        observation_mode=observation_mode,
        memory_mode="none",
        memory_window=1,
        generator_version=1,
        generated_world=(
            {"config_by_partition": partition_configs}
            if partition_configs is not None
            else {"config": dict(generator_config)} if generator_config is not None else {}
        ),
        world_distribution={name: tuple(seeds) for name, seeds in plan.as_dict().items()},
        agent_config={"policy": provider_name},
    )
    manifest_path = manifest.persist(output)
    jobs = [(partition.name, seed) for partition in (plan.train, plan.validation, plan.test) for seed in partition.seeds]

    def execute(job: tuple[str, int]) -> dict[str, Any]:
        partition, seed = job
        generated_world: dict[str, Any] = {"seed": seed}
        selected_config = partition_configs.get(partition) if partition_configs is not None else generator_config
        if selected_config is not None:
            generated_world["config"] = dict(selected_config)
        if built_in_world_partition(seed) == partition:
            generated_world["partition"] = partition
        run_kwargs = {"generated_world": generated_world}
        if observation_mode != "normal":
            run_kwargs["observation_mode"] = observation_mode
        result = run_remote(provider_for(provider_name, seed, effective_model), seed, base_url, **run_kwargs)
        final = result.records[-1] if result.records else {}
        metrics = final.get("metrics", {}) if isinstance(final.get("metrics"), dict) else {}
        return {
            "partition": partition,
            "seed": seed,
            "run_id": result.run_id,
            "world_manifest": result.world_manifest,
            "hazard_kinds": generated_hazard_kinds(result.world_manifest),
            "room_count": generated_room_count(result.world_manifest),
            "mechanics_signature": generated_mechanics_signature(result.world_manifest),
            "world_validation": generated_world_validation(result.world_manifest),
            "provider": provider_name,
            "model": effective_model,
            "outcome": result.terminal_reason,
            "steps": result.steps,
            "total_reward": sum(record.get("reward", 0) for record in result.records if isinstance(record.get("reward", 0), (int, float))),
            "invalid_actions": metrics.get("invalid_actions"),
            "exploration_coverage": metrics.get("exploration_coverage"),
            "resource_efficiency": metrics.get("resource_efficiency"),
            "control_elapsed_ms": final.get("control_elapsed_ms"),
            "generator_config": dict(selected_config) if selected_config is not None else None,
            "generator_config_fingerprint": generator_config_fingerprint(selected_config),
            "observation_mode": observation_mode,
        }

    with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as executor:
        rows = list(executor.map(execute, jobs))
    rows.sort(key=lambda row: (row["partition"], row["seed"]))
    report = summarize_generalization(rows) | {
        "experiment_id": manifest.experiment_id,
        "experiment_manifest": manifest_path.name,
        "experiment_fingerprint": manifest.fingerprint(),
        "provider": provider_name,
        "model": effective_model,
        "engine_version": "rust-v1",
        "observation_mode": observation_mode,
        "world_distribution": plan.as_dict(),
        "generator_config": dict(generator_config) if generator_config is not None else None,
        "generator_configs_by_partition": partition_configs,
        "generator_config_fingerprint": shared_config_fingerprint,
        "generator_config_fingerprints_by_partition": partition_config_fingerprints,
        "episode_results": rows,
    }
    (output / "generalization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export_jsonl(rows, output / "generalization_episodes.jsonl")
    return report

def provider_for(name: str, seed: int, model: str | None = None):
    providers = {
        "scripted": ScriptedProvider,
        "mock_reasoning": MockReasoningProvider,
        "random_valid": lambda: RandomValidProvider(seed),
        "cautious": CautiousProvider,
        "explorer": ExplorerProvider,
        "gemini": lambda: GeminiProvider(model=model),
    }
    try:
        return providers[name]()
    except KeyError as error:
        raise ValueError(f"Unknown benchmark provider: {name}") from error

def benchmark_slices(trajectories: list[dict]) -> tuple[list[dict], list[dict]]:
    events, decisions = [], []
    for record in trajectories:
        common = {key: record[key] for key in ("run_id", "scenario_id", "scenario_version", "seed", "step", "provider", "model")}
        decisions.append({**common, "chosen_action": json.dumps(record["chosen_action"]), "decision_summary": record["decision_summary"], "action_valid": record["action_valid"], "latency_ms": record["latency_ms"], "step_latency_ms": record.get("step_latency_ms"), "control_elapsed_ms": record.get("control_elapsed_ms"), "provider_attempts": record.get("provider_attempts",1), "token_usage": json.dumps(record["token_usage"])})
        decisions[-1]["provider_backoff_ms"] = json.dumps(record.get("provider_backoff_ms", []))
        events.extend({**common, "event_type": event["type"], "event_message": event["message"], "event": json.dumps(event)} for event in record["events"])
    return events, decisions

def benchmark_remote(runs: int, seed_start: int, output: Path, base_url: str, provider_name: str="scripted", concurrency: int=1) -> dict:
    if runs < 1: raise ValueError("runs must be positive")
    if concurrency < 1 or concurrency > 64: raise ValueError("concurrency must be between 1 and 64")
    output.mkdir(parents=True, exist_ok=True)
    def execute(index: int):
        provider=provider_for(provider_name, seed_start+index); result=run_remote(provider, seed_start+index, base_url)
        client = RustRunClient(base_url)
        try:
            control = client.replay(result.run_id).get("control", {})
        finally:
            client.close()
        simulation_latency_us = control.get("simulation_latency_us", []) if isinstance(control, dict) else []
        simulation_latency_us = [latency for latency in simulation_latency_us if isinstance(latency, (int, float)) and latency >= 0]
        final_metrics=result.records[-1].get("metrics",{}) if result.records else {}
        step_latencies=[record.get("step_latency_ms") for record in result.records if isinstance(record.get("step_latency_ms"),(int,float))]
        elapsed=result.records[-1].get("control_elapsed_ms") if result.records else None
        row={"run_id":result.run_id,"seed":seed_start+index,"scenario_id":"survival_room","provider":provider.name,"outcome":result.terminal_reason,"steps":result.steps,"score":final_metrics.get("normalized_score",100 if result.terminal_reason=="escaped" else 0),"initialization_latency_ms":result.initialization_latency_ms,"mean_step_latency_ms":sum(step_latencies)/len(step_latencies) if step_latencies else None,"control_elapsed_ms":elapsed,"simulation_latency_us":sum(simulation_latency_us),"simulation_steps_per_second":result.steps/(sum(simulation_latency_us)/1_000_000) if simulation_latency_us and sum(simulation_latency_us)>0 else None}
        return row,result.records
    with ThreadPoolExecutor(max_workers=min(concurrency,runs)) as executor:
        completed=list(executor.map(execute,range(runs)))
    completed.sort(key=lambda pair: pair[0]["seed"])
    rows=[pair[0] for pair in completed]
    trajectories=[record for _,records in completed for record in records]
    event_rows, decision_rows = benchmark_slices(trajectories)
    export_parquet(rows,output/"runs.parquet")
    export_csv(rows,output/"runs.csv")
    export_jsonl(trajectories,output/"trajectories.jsonl")
    export_parquet([{**record,"observation":json.dumps(record["observation"]),"next_observation":json.dumps(record.get("next_observation")),"agent_context":json.dumps(record["agent_context"]),"agent_metadata":json.dumps(record.get("agent_metadata")),"events":json.dumps(record["events"]),"chosen_action":json.dumps(record["chosen_action"]),"metrics":json.dumps(record["metrics"])} for record in trajectories],output/"trajectories.parquet")
    export_parquet(event_rows,output/"events.parquet")
    export_csv(event_rows,output/"events.csv")
    export_parquet(decision_rows,output/"decisions.parquet")
    export_csv(decision_rows,output/"decisions.csv")
    step_latencies=[r["mean_step_latency_ms"] for r in rows if isinstance(r["mean_step_latency_ms"],(int,float))]
    control_times=[r["control_elapsed_ms"] for r in rows if isinstance(r["control_elapsed_ms"],(int,float))]
    initialization_times=[r["initialization_latency_ms"] for r in rows if isinstance(r["initialization_latency_ms"],(int,float))]
    simulation_times=[r["simulation_latency_us"] for r in rows if isinstance(r["simulation_latency_us"],(int,float))]
    total_simulation_us=sum(simulation_times)
    summary={"scenario_id":"survival_room","provider":provider_name,"runs":runs,"seed_start":seed_start,"concurrency":concurrency,"success_rate":sum(r["outcome"]=="escaped" for r in rows)/runs,"mean_score":sum(r["score"] for r in rows)/runs,"mean_steps":sum(r["steps"] for r in rows)/runs,"mean_initialization_latency_ms":sum(initialization_times)/len(initialization_times) if initialization_times else None,"mean_step_latency_ms":sum(step_latencies)/len(step_latencies) if step_latencies else None,"mean_control_elapsed_ms":sum(control_times)/len(control_times) if control_times else None,"simulation_steps_per_second":sum(r["steps"] for r in rows)/(total_simulation_us/1_000_000) if total_simulation_us>0 else None,"engine_version":"rust-v1","base_url":base_url,"seed_results":[{"seed":r["seed"],"outcome":r["outcome"],"score":r["score"],"steps":r["steps"],"initialization_latency_ms":r["initialization_latency_ms"],"simulation_latency_us":r["simulation_latency_us"]} for r in rows]}
    (output/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def benchmark_parallel_scaling(
    runs: int, seed_start: int, output: Path, base_url: str, provider_name: str = "scripted",
    worker_counts: tuple[int, ...] = (1, 8, 32, 64),
) -> dict[str, Any]:
    """Measure bounded local parallel episode execution at explicit worker counts."""
    if not worker_counts or any(count < 1 or count > 64 for count in worker_counts):
        raise ValueError("worker counts must be between 1 and 64")
    if len(set(worker_counts)) != len(worker_counts):
        raise ValueError("worker counts must be unique")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for count in worker_counts:
        report = benchmark_remote(runs, seed_start, output / f"workers-{count}", base_url, provider_name, count)
        reports.append({
            "workers": count,
            "runs": report["runs"],
            "simulation_steps_per_second": report["simulation_steps_per_second"],
            "mean_initialization_latency_ms": report.get("mean_initialization_latency_ms"),
            "mean_control_elapsed_ms": report["mean_control_elapsed_ms"],
            "mean_step_latency_ms": report["mean_step_latency_ms"],
        })
    result = {"report_version": 1, "provider": provider_name, "runs_per_level": runs, "seed_start": seed_start, "levels": reports}
    (output / "parallel_scaling.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

def compare_benchmarks(left: dict, right: dict) -> dict:
    if left.get("scenario_id") != right.get("scenario_id"):
        raise ValueError("benchmark scenarios must match")
    return {
        "scenario_id": left.get("scenario_id"),
        "left_provider": left.get("provider"),
        "right_provider": right.get("provider"),
        "success_rate_delta": right.get("success_rate",0)-left.get("success_rate",0),
        "mean_score_delta": right.get("mean_score",0)-left.get("mean_score",0),
        "mean_steps_delta": right.get("mean_steps",0)-left.get("mean_steps",0),
        "mean_initialization_latency_ms_delta": None if left.get("mean_initialization_latency_ms") is None or right.get("mean_initialization_latency_ms") is None else right["mean_initialization_latency_ms"]-left["mean_initialization_latency_ms"],
        "mean_step_latency_ms_delta": None if left.get("mean_step_latency_ms") is None or right.get("mean_step_latency_ms") is None else right["mean_step_latency_ms"]-left["mean_step_latency_ms"],
        "mean_control_elapsed_ms_delta": None if left.get("mean_control_elapsed_ms") is None or right.get("mean_control_elapsed_ms") is None else right["mean_control_elapsed_ms"]-left["mean_control_elapsed_ms"],
        "left_runs": left.get("runs",0),
        "right_runs": right.get("runs",0),
    }


def validate_generalization_report(report: dict[str, Any]) -> None:
    """Reject incomplete provenance before a research comparison is attempted."""
    if not isinstance(report, dict):
        raise ValueError("generalization report must be an object")
    if report.get("report_version") != 1 or not isinstance(report.get("engine_version"), str) or not report["engine_version"]:
        raise ValueError("generalization report requires report_version and engine_version")
    if report.get("observation_mode") not in {"minimal", "normal", "rich", "noisy", "oracle"}:
        raise ValueError("generalization report requires a supported observation_mode")
    report_provider = report.get("provider")
    if report_provider is not None and (not isinstance(report_provider, str) or not report_provider):
        raise ValueError("generalization report provider must be non-empty text when supplied")
    if "model" in report and report.get("model") is not None and not isinstance(report.get("model"), str):
        raise ValueError("generalization report model must be text or null when supplied")
    distribution = report.get("world_distribution")
    if not isinstance(distribution, dict) or set(distribution) != {"train", "validation", "test"}:
        raise ValueError("generalization report requires train, validation, and test world_distribution")
    if any(not isinstance(values, list) or not values for values in distribution.values()):
        raise ValueError("generalization world_distribution partitions must be non-empty seed lists")
    seeds = [seed for values in distribution.values() for seed in values]
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("generalization world_distribution seeds must be non-negative and disjoint")
    shared_config = report.get("generator_config")
    configs_by_partition = report.get("generator_configs_by_partition")
    if shared_config is not None and not isinstance(shared_config, dict):
        raise ValueError("generalization generator_config must be an object or null")
    if configs_by_partition is not None:
        if shared_config is not None or not isinstance(configs_by_partition, dict) or set(configs_by_partition) != {"train", "validation", "test"} or any(not isinstance(config, dict) for config in configs_by_partition.values()):
            raise ValueError("generalization generator configs by partition must be complete objects without a shared config")
    expected_shared_fingerprint = generator_config_fingerprint(shared_config)
    expected_partition_fingerprints = (
        {name: generator_config_fingerprint(config) for name, config in configs_by_partition.items()}
        if configs_by_partition is not None
        else None
    )
    if "generator_config_fingerprint" in report and report.get("generator_config_fingerprint") != expected_shared_fingerprint:
        raise ValueError("generalization generator_config_fingerprint does not match generator_config")
    if "generator_config_fingerprints_by_partition" in report and report.get("generator_config_fingerprints_by_partition") != expected_partition_fingerprints:
        raise ValueError("generalization generator config fingerprints do not match partition configs")
    partitions = report.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise ValueError("generalization report requires train, validation, and test summaries")
    episodes = report.get("episode_results")
    if episodes is None:
        return
    if not isinstance(episodes, list):
        raise ValueError("generalization episode_results must be a list when supplied")
    expected_episodes = {(name, seed) for name, values in distribution.items() for seed in values}
    observed_episodes = {(row.get("partition"), row.get("seed")) for row in episodes if isinstance(row, dict)}
    if len(episodes) != len(observed_episodes) or observed_episodes != expected_episodes:
        raise ValueError("generalization episode_results must contain each declared partition seed exactly once")
    for row in episodes:
        if not isinstance(row, dict) or row.get("observation_mode") != report["observation_mode"]:
            raise ValueError("generalization episodes must match the report observation_mode")
        if report_provider is not None and row.get("provider") != report_provider:
            raise ValueError("generalization episodes must match the report provider")
        if "model" in report and row.get("model") != report.get("model"):
            raise ValueError("generalization episodes must match the report model")
        if row.get("world_manifest") is not None and not isinstance(row.get("world_manifest"), dict):
            raise ValueError("generalization episode world_manifest must be an object or null")
        validation = row.get("world_validation")
        if validation is not None and (
            not isinstance(validation, dict)
            or any(type(validation.get(field)) is not bool for field in ("geometry_valid", "spawn_valid", "required_key_reachable", "exit_reachable", "solvable"))
        ):
            raise ValueError("generalization episode world_validation must be complete boolean evidence or null")
        manifest_validation = generated_world_validation(row.get("world_manifest"))
        if manifest_validation is not None and validation != manifest_validation:
            raise ValueError("generalization episode world_validation must match its Rust world_manifest")
        expected_config = configs_by_partition.get(row["partition"]) if isinstance(configs_by_partition, dict) else shared_config
        if row.get("generator_config") != expected_config:
            raise ValueError("generalization episodes must match the report generator configuration")
        world_manifest = row.get("world_manifest")
        if isinstance(world_manifest, dict):
            manifest_seed = world_manifest.get("seed")
            if manifest_seed is not None and (type(manifest_seed) is not int or manifest_seed != row["seed"]):
                raise ValueError("generalization episode world_manifest seed must match its declared seed")
            manifest_config = world_manifest.get("generator_config")
            if manifest_config is not None:
                if not isinstance(manifest_config, dict):
                    raise ValueError("generalization episode world_manifest generator_config must be an object")
                if expected_config is not None and any(manifest_config.get(key) != value for key, value in expected_config.items()):
                    raise ValueError("generalization episode world_manifest generator_config must match its selected config")
        if "generator_config_fingerprint" in row and row.get("generator_config_fingerprint") != generator_config_fingerprint(expected_config):
            raise ValueError("generalization episode generator configuration fingerprint does not match")
    for name, summary in partitions.items():
        if not isinstance(summary, dict):
            raise ValueError("generalization partition summaries must be objects")
        rows = [row for row in episodes if row["partition"] == name]
        successes = sum(row.get("outcome") == "escaped" for row in rows)
        if summary.get("episodes") != len(rows) or summary.get("success_rate") != successes / len(rows):
            raise ValueError("generalization partition summaries must match episode_results")
        expected_validation = world_validation_summary(rows)
        if "world_validation" in summary and summary.get("world_validation") != expected_validation:
            raise ValueError("generalization world validation summary must match episode_results")


def audit_generalization_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate one report and return a compact, shareable audit receipt."""
    validate_generalization_report(report)
    episodes = report.get("episode_results")
    return {
        "valid": True,
        "report_version": report["report_version"],
        "engine_version": report["engine_version"],
        "observation_mode": report["observation_mode"],
        "experiment_id": report.get("experiment_id"),
        "episode_provenance_checked": isinstance(episodes, list),
        "episode_count": len(episodes) if isinstance(episodes, list) else None,
        "world_distribution": report["world_distribution"],
        "generator_config_fingerprint": report.get("generator_config_fingerprint"),
        "generator_config_fingerprints_by_partition": report.get("generator_config_fingerprints_by_partition"),
        "partition_episode_counts": {
            name: report["partitions"][name].get("episodes")
            for name in ("train", "validation", "test")
        },
    }


def compare_generalization_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare only reports produced under identical experimental conditions."""
    validate_generalization_report(left)
    validate_generalization_report(right)
    for field in ("report_version", "engine_version", "observation_mode", "world_distribution", "generator_config", "generator_configs_by_partition"):
        if left.get(field) != right.get(field):
            raise ValueError(f"generalization reports must match {field}")
    expected = {"train", "validation", "test"}
    left_partitions, right_partitions = left.get("partitions"), right.get("partitions")
    metrics = ("success_rate", "mean_episode_reward", "mean_episode_length", "invalid_action_rate", "mean_exploration_coverage", "mean_resource_efficiency", "steps_per_second")
    def delta(left_value: Any, right_value: Any) -> float | None:
        return right_value - left_value if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)) else None
    def hazard_deltas(name: str) -> dict[str, dict[str, float | None]]:
        left_hazards = left_partitions[name].get("hazard_kind_breakdown", {})
        right_hazards = right_partitions[name].get("hazard_kind_breakdown", {})
        if not isinstance(left_hazards, dict) or not isinstance(right_hazards, dict):
            return {}
        return {
            kind: {
                "episodes_delta": delta(left_hazards.get(kind, {}).get("episodes"), right_hazards.get(kind, {}).get("episodes")),
                "success_rate_delta": delta(left_hazards.get(kind, {}).get("success_rate"), right_hazards.get(kind, {}).get("success_rate")),
            }
            for kind in sorted(set(left_hazards) | set(right_hazards))
        }
    def room_count_deltas(name: str) -> dict[str, dict[str, float | None]]:
        left_rooms = left_partitions[name].get("room_count_breakdown", {})
        right_rooms = right_partitions[name].get("room_count_breakdown", {})
        if not isinstance(left_rooms, dict) or not isinstance(right_rooms, dict):
            return {}
        return {
            room_count: {
                "episodes_delta": delta(left_rooms.get(room_count, {}).get("episodes"), right_rooms.get(room_count, {}).get("episodes")),
                "success_rate_delta": delta(left_rooms.get(room_count, {}).get("success_rate"), right_rooms.get(room_count, {}).get("success_rate")),
            }
            for room_count in sorted(set(left_rooms) | set(right_rooms))
        }
    def mechanics_deltas(name: str) -> dict[str, dict[str, float | None]]:
        left_mechanics = left_partitions[name].get("mechanics_breakdown", {})
        right_mechanics = right_partitions[name].get("mechanics_breakdown", {})
        if not isinstance(left_mechanics, dict) or not isinstance(right_mechanics, dict):
            return {}
        return {
            signature: {
                "episodes_delta": delta(left_mechanics.get(signature, {}).get("episodes"), right_mechanics.get(signature, {}).get("episodes")),
                "success_rate_delta": delta(left_mechanics.get(signature, {}).get("success_rate"), right_mechanics.get(signature, {}).get("success_rate")),
            }
            for signature in sorted(set(left_mechanics) | set(right_mechanics))
        }
    def paired_episode_comparison() -> dict[str, Any] | None:
        """Show paired success changes when both reports retain episode evidence."""
        left_episodes, right_episodes = left.get("episode_results"), right.get("episode_results")
        if not isinstance(left_episodes, list) or not isinstance(right_episodes, list):
            return None
        left_by_seed = {(row["partition"], row["seed"]): row for row in left_episodes}
        right_by_seed = {(row["partition"], row["seed"]): row for row in right_episodes}
        if set(left_by_seed) != set(right_by_seed):
            raise ValueError("generalization reports must retain the same episode seeds for paired comparison")
        comparisons: dict[str, Any] = {}
        for name in sorted(expected):
            pairs = [
                (left_by_seed[(name, seed)], right_by_seed[(name, seed)])
                for seed in left["world_distribution"][name]
            ]
            both_success = sum(first.get("outcome") == "escaped" and second.get("outcome") == "escaped" for first, second in pairs)
            left_only_success = sum(first.get("outcome") == "escaped" and second.get("outcome") != "escaped" for first, second in pairs)
            right_only_success = sum(first.get("outcome") != "escaped" and second.get("outcome") == "escaped" for first, second in pairs)
            neither_success = len(pairs) - both_success - left_only_success - right_only_success
            manifests_available = all(
                isinstance(first.get("world_manifest"), dict) and isinstance(second.get("world_manifest"), dict)
                for first, second in pairs
            )
            if any(first.get("world_manifest") != second.get("world_manifest") for first, second in pairs):
                raise ValueError("generalization paired episodes must use identical world manifests")
            comparisons[name] = {
                "episodes": len(pairs),
                "both_success": both_success,
                "left_only_success": left_only_success,
                "right_only_success": right_only_success,
                "neither_success": neither_success,
                "paired_success_rate_delta": (right_only_success - left_only_success) / len(pairs),
                "discordant_pairs": left_only_success + right_only_success,
                "paired_success_exact_p_value": paired_success_exact_p_value(left_only_success, right_only_success),
                "world_manifests_checked": manifests_available,
            }
        return comparisons
    return {
        "engine_version": left.get("engine_version"), "observation_mode": left.get("observation_mode"),
        "left_experiment_id": left.get("experiment_id"), "right_experiment_id": right.get("experiment_id"),
        "left_provider": left.get("provider"), "left_model": left.get("model"),
        "right_provider": right.get("provider"), "right_model": right.get("model"),
        "partitions": {name: {**{f"{metric}_delta": delta(left_partitions[name].get(metric), right_partitions[name].get(metric)) for metric in metrics}, "hazard_kind_breakdown": hazard_deltas(name), "room_count_breakdown": room_count_deltas(name), "mechanics_breakdown": mechanics_deltas(name)} for name in sorted(expected)},
        "generalization_gap": {key + "_delta": delta(left.get("generalization_gap", {}).get(key), right.get("generalization_gap", {}).get(key)) for key in ("train_minus_validation_success_rate", "train_minus_test_success_rate")},
        "paired_episode_comparison": paired_episode_comparison(),
    }
