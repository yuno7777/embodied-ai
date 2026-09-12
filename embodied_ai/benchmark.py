from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .datasets import export_csv, export_jsonl, export_parquet
from .engine import Environment
from .providers import CautiousProvider, ExplorerProvider, GeminiProvider, MockReasoningProvider, RandomValidProvider, ScriptedProvider
from .runner import run_remote

def benchmark(runs: int, seed_start: int, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True); rows=[]
    for index in range(runs):
        env=Environment(seed=seed_start+index); provider=ScriptedProvider()
        while not env.done: env.step(provider.choose_action(env.observe()))
        rows.append({"run_id":env.run_id,"seed":env.seed,"scenario_id":env.scenario.raw["id"],"scenario_version":env.scenario.raw["version"],"provider":"scripted","outcome":env.terminal_reason,"steps":env.step_number,"final_health":env.agent.health,"invalid_actions":env.invalid_actions,"hazard_damage":env.hazard_damage,"score":100 if env.terminal_reason=="escaped" else 0})
    export_parquet(rows, output / "runs.parquet")
    summary={"scenario_id":"survival_room","provider":"scripted","runs":runs,"success_rate":sum(r["outcome"]=="escaped" for r in rows)/runs,"mean_score":sum(r["score"] for r in rows)/runs,"engine_version":"python-prototype-v1"}
    (output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2)); return summary

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
    if concurrency < 1 or concurrency > 32: raise ValueError("concurrency must be between 1 and 32")
    output.mkdir(parents=True, exist_ok=True)
    def execute(index: int):
        provider=provider_for(provider_name, seed_start+index); result=run_remote(provider, seed_start+index, base_url)
        final_metrics=result.records[-1].get("metrics",{}) if result.records else {}
        step_latencies=[record.get("step_latency_ms") for record in result.records if isinstance(record.get("step_latency_ms"),(int,float))]
        elapsed=result.records[-1].get("control_elapsed_ms") if result.records else None
        row={"run_id":result.run_id,"seed":seed_start+index,"scenario_id":"survival_room","provider":provider.name,"outcome":result.terminal_reason,"steps":result.steps,"score":final_metrics.get("normalized_score",100 if result.terminal_reason=="escaped" else 0),"mean_step_latency_ms":sum(step_latencies)/len(step_latencies) if step_latencies else None,"control_elapsed_ms":elapsed}
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
    summary={"scenario_id":"survival_room","provider":provider_name,"runs":runs,"seed_start":seed_start,"concurrency":concurrency,"success_rate":sum(r["outcome"]=="escaped" for r in rows)/runs,"mean_score":sum(r["score"] for r in rows)/runs,"mean_steps":sum(r["steps"] for r in rows)/runs,"mean_step_latency_ms":sum(step_latencies)/len(step_latencies) if step_latencies else None,"mean_control_elapsed_ms":sum(control_times)/len(control_times) if control_times else None,"engine_version":"rust-v1","base_url":base_url,"seed_results":[{"seed":r["seed"],"outcome":r["outcome"],"score":r["score"],"steps":r["steps"]} for r in rows]}
    (output/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)); return summary

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
