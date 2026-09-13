"""HTTP orchestration client. It can only request observations and submit actions."""
from __future__ import annotations
import asyncio, time
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field
import httpx
from .context import AgentContext
from .schemas import AgentDecision, ActionRequest

@dataclass
class RemoteRunResult:
    run_id: str
    terminal_reason: str | None
    steps: int
    records: list[dict] = field(default_factory=list)
    stop_detail: str | None = None
    world_manifest: dict | None = None
    initialization_latency_ms: float | None = None

class RustRunClient:
    def __init__(self, base_url: str="http://127.0.0.1:8080", timeout: float=15): self.client=httpx.Client(base_url=base_url, timeout=timeout)
    def create(self, seed: int, max_steps: int | None = None, observation_mode: str = "normal", scenario_id: str | None = None, generated_world: dict[str, Any] | None = None, reward_config: dict[str, int] | None = None) -> dict:
        if scenario_id is not None and generated_world is not None:
            raise ValueError("choose either scenario_id or generated_world")
        request = {"seed": seed}
        if scenario_id is not None:
            request["scenario_id"] = scenario_id
        if generated_world is not None:
            request["generated_world"] = generated_world
        if reward_config is not None:
            request["reward_config"] = reward_config
        if max_steps is not None: request["max_steps"] = max_steps
        request["observation_mode"] = observation_mode
        return self.client.post("/api/runs", json=request).raise_for_status().json()
    def scenarios(self) -> list[dict]: return self.client.get("/api/scenarios").raise_for_status().json()
    def world_partition(self, seed: int) -> str: return self.client.get(f"/api/worlds/partition/{seed}").raise_for_status().json()
    def snapshot(self, run_id: str) -> dict: return self.client.get(f"/api/runs/{run_id}").raise_for_status().json()
    def status(self, run_id: str) -> dict: return self.client.get(f"/api/runs/{run_id}/status").raise_for_status().json()
    def observation(self, run_id: str) -> dict: return self.client.get(f"/api/runs/{run_id}/observation").raise_for_status().json()
    def replay(self, run_id: str) -> dict: return self.client.get(f"/api/runs/{run_id}/replay").raise_for_status().json()
    def restore(self, replay_id: str) -> dict: return self.client.post(f"/api/replays/{replay_id}/resume").raise_for_status().json()
    def record_decision(self, run_id: str, action: ActionRequest, decision_summary: str, provider: str, model: str | None, latency_ms: int, token_usage: dict | None, agent_metadata: dict | None = None) -> dict:
        return self.client.post(f"/api/runs/{run_id}/decision", json={"action": action.model_dump(exclude_none=True), "decision_summary": decision_summary, "provider": provider, "model": model, "latency_ms": max(0, round(latency_ms)), "token_usage": token_usage, "agent_metadata": agent_metadata}).raise_for_status().json()
    def step(self, run_id: str, action: ActionRequest) -> dict: return self.client.post(f"/api/runs/{run_id}/step",json=action.model_dump(exclude_none=True)).raise_for_status().json()
    def provider_error(self, run_id: str) -> dict: return self.client.post(f"/api/runs/{run_id}/provider-error").raise_for_status().json()
    def stop(self, run_id: str, reason: str) -> dict: return self.client.post(f"/api/runs/{run_id}/stop", json={"reason": reason}).raise_for_status().json()
    def close(self) -> None: self.client.close()

def run_remote(provider, seed: int, base_url: str="http://127.0.0.1:8080", memory_mode: str="recent", max_steps: int | None = None, observation_mode: str = "normal", on_created: Callable[[str], None] | None = None, max_wall_seconds: float | None = None, max_total_tokens: int | None = None, resume_run_id: str | None = None, restore_replay_id: str | None = None, memory_window: int = 5, scenario_id: str | None = None, generated_world: dict[str, Any] | None = None, reward_config: dict[str, int] | None = None, experiment_id: str | None = None, include_research_snapshots: bool = False, policy_state_mode: str = "reset") -> RemoteRunResult:
    context=AgentContext(memory_mode=memory_mode, memory_window=memory_window); client=RustRunClient(base_url)
    run_id: str | None = None
    initialization_latency_ms: float | None = None
    try:
        if policy_state_mode not in {"reset", "preserve"}:
            raise ValueError("policy_state_mode must be reset or preserve")
        if policy_state_mode == "reset" and hasattr(provider, "reset"):
            provider.reset(seed)
        if max_wall_seconds is not None and max_wall_seconds <= 0: raise ValueError("max_wall_seconds must be positive")
        if max_total_tokens is not None and max_total_tokens <= 0: raise ValueError("max_total_tokens must be positive")
        if resume_run_id is not None and restore_replay_id is not None: raise ValueError("choose either a live run or a persisted replay to resume")
        scenario: dict[str, object] | None = None
        world_manifest: dict | None = None
        if restore_replay_id is not None:
            initialization_started=time.perf_counter(); restored=client.restore(restore_replay_id); initialization_latency_ms=(time.perf_counter()-initialization_started)*1000; run_id=restored["run_id"]; observation=restored["observation"]; steps=restored["snapshot"]["step"]
        elif resume_run_id is None:
            initialization_started=time.perf_counter(); created=client.create(seed, max_steps, observation_mode, scenario_id, generated_world, reward_config); initialization_latency_ms=(time.perf_counter()-initialization_started)*1000; run_id=created["run_id"]; observation=created["observation"]; steps=0
            manifest = created.get("world_manifest")
            world_manifest = manifest if isinstance(manifest, dict) else None
            if isinstance(manifest, dict) and isinstance(manifest.get("scenario"), dict):
                scenario = manifest["scenario"]
            else:
                scenario = client.scenarios()[0]
        else:
            run_id=resume_run_id; status=client.status(run_id); steps=status["step"]
            if status["done"]: return RemoteRunResult(run_id,status.get("terminal_reason"),steps,[],"run was already terminal",initialization_latency_ms=initialization_latency_ms)
            observation=client.observation(run_id)
        records=[]; total_tokens=0; started=time.monotonic()
        # Snapshots are deliberately captured into export-only records, never
        # into ``observation`` or ``AgentContext`` passed to the provider.
        research_snapshot = client.snapshot(run_id) if include_research_snapshots else None
        if on_created is not None:
            on_created(run_id)
        try:
            while True:
                if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
                    client.stop(run_id, "client_timeout")
                    return RemoteRunResult(run_id,"client_timeout",steps,records,"wall-clock budget exhausted",world_manifest,initialization_latency_ms)
                status = client.status(run_id)
                if status["done"]:
                    return RemoteRunResult(run_id, status.get("terminal_reason"), status["step"], records, world_manifest=world_manifest, initialization_latency_ms=initialization_latency_ms)
                if status["paused"]:
                    time.sleep(0.1)
                    continue
                provider_context = context.payload()
                if hasattr(provider, "choose"):
                    decision, provider_latency_ms=asyncio.run(provider.choose(observation,context)); action=decision.action; decision_summary=decision.decision_summary; agent_metadata=decision.agent_metadata.model_dump(exclude_none=True) if decision.agent_metadata is not None else None
                elif hasattr(provider, "decide"):
                    decision = provider.decide(observation)
                    if not isinstance(decision, AgentDecision):
                        decision = AgentDecision.model_validate(decision)
                    action=decision.action; decision_summary=decision.decision_summary; agent_metadata=decision.agent_metadata.model_dump(exclude_none=True) if decision.agent_metadata is not None else None; provider_latency_ms=0
                else:
                    action=(provider.act(observation) if hasattr(provider, "act") else provider.choose_action(observation))
                    if not isinstance(action, ActionRequest):
                        action = ActionRequest.model_validate(action)
                    provider_latency_ms=0; decision_summary="provider action submitted to Rust authority"; agent_metadata=None
                if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
                    client.stop(run_id, "client_timeout")
                    return RemoteRunResult(run_id,"client_timeout",steps,records,"wall-clock budget exhausted during provider decision",world_manifest,initialization_latency_ms)
                token_usage = getattr(provider, "last_token_usage", None)
                decision_tokens = (token_usage or {}).get("total_tokens", 0)
                if max_total_tokens is not None and total_tokens + decision_tokens > max_total_tokens:
                    client.stop(run_id, "token_budget_exhausted")
                    return RemoteRunResult(run_id,"token_budget_exhausted",steps,records,f"token budget {max_total_tokens} exceeded",world_manifest,initialization_latency_ms)
                total_tokens += decision_tokens
                client.record_decision(run_id, action, decision_summary, provider.name, getattr(provider, "model", None), provider_latency_ms, token_usage, agent_metadata)
                step_started=time.perf_counter(); result=client.step(run_id,action); step_latency_ms=(time.perf_counter()-step_started)*1000; steps=result["step_number"]
                if scenario is None:
                    scenario = client.scenarios()[0]
                next_research_snapshot = client.snapshot(run_id) if include_research_snapshots else None
                records.append({"dataset_schema_version":1,"experiment_id":experiment_id,"world_manifest":world_manifest,"run_id":run_id,"scenario_id":scenario["id"],"scenario_version":scenario["version"],"seed":seed,"step":steps,"observation_mode":observation_mode,"policy_state_mode":policy_state_mode,"initialization_latency_ms":initialization_latency_ms,"observation":observation,"next_observation":result["observation"],"agent_context":provider_context,"allowed_actions":observation["allowed_action_types"],"chosen_action":action.model_dump(exclude_none=True),"action_valid":not any(event["type"]=="InvalidAction" for event in result["events"]),"decision_summary":decision_summary,"agent_metadata":agent_metadata,"events":result["events"],"reward":result["reward"],"done":result["done"],"terminal_reason":result["terminal_reason"],"metrics":result["metrics"],"provider":provider.name,"model":getattr(provider,"model",None),"latency_ms":provider_latency_ms,"step_latency_ms":step_latency_ms,"token_usage":token_usage,"provider_attempts":getattr(provider,"last_attempts",1),"cumulative_tokens":total_tokens,"control_elapsed_ms":round((time.monotonic()-started)*1000)})
                if include_research_snapshots:
                    records[-1]["research_snapshot"] = research_snapshot
                    records[-1]["next_research_snapshot"] = next_research_snapshot
                records[-1]["provider_backoff_ms"] = list(getattr(provider, "last_backoff_ms", []))
                context.record(observation,action.model_dump(exclude_none=True),result["events"]); observation=result["observation"]; research_snapshot = next_research_snapshot
                if result["done"]: return RemoteRunResult(run_id,result["terminal_reason"],steps,records,world_manifest=world_manifest,initialization_latency_ms=initialization_latency_ms)
        except Exception:
            if run_id is None:
                raise
            client.provider_error(run_id)
            return RemoteRunResult(run_id,"provider_error",steps,records,world_manifest=world_manifest,initialization_latency_ms=initialization_latency_ms)
    finally: client.close()
