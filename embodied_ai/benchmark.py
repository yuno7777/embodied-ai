from __future__ import annotations
import json
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
    generator_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a policy over disjoint procedural worlds and persist an exact report."""
    if concurrency < 1 or concurrency > 32:
        raise ValueError("concurrency must be between 1 and 32")
    output.mkdir(parents=True, exist_ok=True)
    manifest = ExperimentManifest(
        scenario_id="procedural",
        seed=plan.train.seeds[0],
        provider=provider_name,
        observation_mode="normal",
        memory_mode="none",
        memory_window=1,
        generator_version=1,
        generated_world={"config": dict(generator_config)} if generator_config is not None else {},
        world_distribution={name: tuple(seeds) for name, seeds in plan.as_dict().items()},
        agent_config={"policy": provider_name},
    )
    manifest_path = manifest.persist(output)
    jobs = [(partition.name, seed) for partition in (plan.train, plan.validation, plan.test) for seed in partition.seeds]

    def execute(job: tuple[str, int]) -> dict[str, Any]:
        partition, seed = job
        generated_world: dict[str, Any] = {"seed": seed}
        if generator_config is not None:
            generated_world["config"] = dict(generator_config)
        result = run_remote(provider_for(provider_name, seed), seed, base_url, generated_world=generated_world)
        final = result.records[-1] if result.records else {}
        metrics = final.get("metrics", {}) if isinstance(final.get("metrics"), dict) else {}
        return {
            "partition": partition,
            "seed": seed,
            "run_id": result.run_id,
            "outcome": result.terminal_reason,
            "steps": result.steps,
            "total_reward": sum(record.get("reward", 0) for record in result.records if isinstance(record.get("reward", 0), (int, float))),
            "invalid_actions": metrics.get("invalid_actions"),
            "exploration_coverage": metrics.get("exploration_coverage"),
            "resource_efficiency": metrics.get("resource_efficiency"),
            "control_elapsed_ms": final.get("control_elapsed_ms"),
        }

    with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as executor:
        rows = list(executor.map(execute, jobs))
    rows.sort(key=lambda row: (row["partition"], row["seed"]))
    report = summarize_generalization(rows) | {
        "experiment_id": manifest.experiment_id,
        "experiment_manifest": manifest_path.name,
        "experiment_fingerprint": manifest.fingerprint(),
        "provider": provider_name,
        "engine_version": "rust-v1",
        "world_distribution": plan.as_dict(),
        "generator_config": dict(generator_config) if generator_config is not None else None,
        "episode_results": rows,
    }
    (output / "generalization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export_jsonl(rows, output / "generalization_episodes.jsonl")
    return report

def provider_for(name: str, seed: int):
    providers = {
        "scripted": ScriptedProvider,
        "mock_reasoning": MockReasoningProvider,
        "random_valid": lambda: RandomValidProvider(seed),
        "cautious": CautiousProvider,
        "explorer": ExplorerProvider,
        "gemini": GeminiProvider,
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
        row={"run_id":result.run_id,"seed":seed_start+index,"scenario_id":"survival_room","provider":provider.name,"outcome":result.terminal_reason,"steps":result.steps,"score":final_metrics.get("normalized_score",100 if result.terminal_reason=="escaped" else 0),"mean_step_latency_ms":sum(step_latencies)/len(step_latencies) if step_latencies else None,"control_elapsed_ms":elapsed,"simulation_latency_us":sum(simulation_latency_us),"simulation_steps_per_second":result.steps/(sum(simulation_latency_us)/1_000_000) if simulation_latency_us and sum(simulation_latency_us)>0 else None}
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
    export_parquet([{**record,"observation":json.dumps(record["observation"]),"agent_context":json.dumps(record["agent_context"]),"events":json.dumps(record["events"]),"chosen_action":json.dumps(record["chosen_action"]),"metrics":json.dumps(record["metrics"])} for record in trajectories],output/"trajectories.parquet")
    export_parquet(event_rows,output/"events.parquet")
    export_csv(event_rows,output/"events.csv")
    export_parquet(decision_rows,output/"decisions.parquet")
    export_csv(decision_rows,output/"decisions.csv")
    step_latencies=[r["mean_step_latency_ms"] for r in rows if isinstance(r["mean_step_latency_ms"],(int,float))]
    control_times=[r["control_elapsed_ms"] for r in rows if isinstance(r["control_elapsed_ms"],(int,float))]
    simulation_times=[r["simulation_latency_us"] for r in rows if isinstance(r["simulation_latency_us"],(int,float))]
    total_simulation_us=sum(simulation_times)
    summary={"scenario_id":"survival_room","provider":provider_name,"runs":runs,"seed_start":seed_start,"concurrency":concurrency,"success_rate":sum(r["outcome"]=="escaped" for r in rows)/runs,"mean_score":sum(r["score"] for r in rows)/runs,"mean_steps":sum(r["steps"] for r in rows)/runs,"mean_step_latency_ms":sum(step_latencies)/len(step_latencies) if step_latencies else None,"mean_control_elapsed_ms":sum(control_times)/len(control_times) if control_times else None,"simulation_steps_per_second":sum(r["steps"] for r in rows)/(total_simulation_us/1_000_000) if total_simulation_us>0 else None,"engine_version":"rust-v1","base_url":base_url,"seed_results":[{"seed":r["seed"],"outcome":r["outcome"],"score":r["score"],"steps":r["steps"],"simulation_latency_us":r["simulation_latency_us"]} for r in rows]}
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
        "mean_step_latency_ms_delta": None if left.get("mean_step_latency_ms") is None or right.get("mean_step_latency_ms") is None else right["mean_step_latency_ms"]-left["mean_step_latency_ms"],
        "mean_control_elapsed_ms_delta": None if left.get("mean_control_elapsed_ms") is None or right.get("mean_control_elapsed_ms") is None else right["mean_control_elapsed_ms"]-left["mean_control_elapsed_ms"],
        "left_runs": left.get("runs",0),
        "right_runs": right.get("runs",0),
    }
