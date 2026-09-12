"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Snapshot = {
  run_id: string; step: number; done: boolean; terminal_reason?: string;
  agent: { health: number };
  metrics: { invalid_actions: number; normalized_score: number; hazard_damage_taken?: number };
};
type Replay = { snapshot: Snapshot };
type BenchmarkSummary = {
  total_runs: number; completed_runs: number; escaped_runs: number; success_rate: number;
  mean_steps: number; mean_simulated_time: number; mean_normalized_score: number;
  mean_final_health: number; mean_invalid_actions: number; mean_hazard_damage: number;
  total_provider_calls: number; mean_provider_latency_ms: number | null;
  total_input_tokens: number | null; total_output_tokens: number | null;
};
const API = process.env.NEXT_PUBLIC_SIM_SERVER_URL ?? "http://localhost:8080";

export default function BenchmarksPage() {
  const [runs, setRuns] = useState<Snapshot[]>([]);
  const [summary, setSummary] = useState<BenchmarkSummary>();
  const [error, setError] = useState<string>();
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");

  async function load() {
    try {
      const [current, archived, aggregate] = await Promise.all([fetch(API + "/api/runs"), fetch(API + "/api/replays"), fetch(API + "/api/benchmarks")]);
      if (!current.ok || !archived.ok || !aggregate.ok) throw new Error("Simulation server is unavailable.");
      const snapshots = (await current.json()) as Snapshot[];
      const replays = (await archived.json()) as Replay[];
      const unique = new Map(snapshots.map((snapshot) => [snapshot.run_id, snapshot]));
      replays.forEach((replay) => unique.set(replay.snapshot.run_id, replay.snapshot));
      const loaded = [...unique.values()];
      setRuns(loaded);
      setLeftId((currentId) => currentId || loaded[0]?.run_id || "");
      setRightId((currentId) => currentId || loaded[1]?.run_id || loaded[0]?.run_id || "");
      setSummary(await aggregate.json() as BenchmarkSummary);
      setError(undefined);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to load benchmarks."); }
  }
  useEffect(() => { const timer = window.setTimeout(() => { void load(); }, 0); return () => window.clearTimeout(timer); }, []);

  const completed = runs.filter((run) => run.done);
  const left = runs.find((run) => run.run_id === leftId);
  const right = runs.find((run) => run.run_id === rightId);
  const delta = (rightValue?: number, leftValue?: number) => rightValue === undefined || leftValue === undefined ? "—" : (rightValue - leftValue).toFixed(1);

  return <main className="benchmark-page">
    <header><div><p>RESEARCH BENCHMARKS / V1</p><h1>Run comparison</h1></div><Link href="/">Observer</Link></header>
    <section>
      <div className="panel"><span>AUTHORITATIVE AGGREGATE</span><div className="metrics">
        <b>RUNS {summary?.total_runs ?? 0}</b><b>COMPLETED {summary?.completed_runs ?? 0}</b><b>SUCCESS {summary ? (summary.success_rate * 100).toFixed(0) : 0}%</b><b>MEAN STEPS {summary?.mean_steps.toFixed(1) ?? "0.0"}</b><b>MEAN SCORE {summary?.mean_normalized_score.toFixed(1) ?? "0.0"}</b><b>MEAN HEALTH {summary?.mean_final_health.toFixed(1) ?? "0.0"}</b><b>INVALID {summary?.mean_invalid_actions.toFixed(1) ?? "0.0"}</b><b>HAZARD {summary?.mean_hazard_damage.toFixed(1) ?? "0.0"}</b><b>PROVIDER CALLS {summary?.total_provider_calls ?? 0}</b><b>MEAN LATENCY {summary?.mean_provider_latency_ms == null ? "unavailable" : summary.mean_provider_latency_ms.toFixed(1) + " ms"}</b><b>INPUT TOKENS {summary?.total_input_tokens ?? "unavailable"}</b><b>OUTPUT TOKENS {summary?.total_output_tokens ?? "unavailable"}</b>
      </div><button onClick={() => void load()}>Refresh</button></div>
      <div className="panel"><span>PAIRWISE DELTA (RIGHT − LEFT)</span><div className="compare-controls">
        <label>Left run<select aria-label="Left run" value={leftId} onChange={(event) => setLeftId(event.target.value)}>{completed.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id.slice(0, 8)} · {run.terminal_reason}</option>)}</select></label>
        <label>Right run<select aria-label="Right run" value={rightId} onChange={(event) => setRightId(event.target.value)}>{completed.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id.slice(0, 8)} · {run.terminal_reason}</option>)}</select></label>
      </div><div className="metrics"><b>SCORE Δ {delta(right?.metrics.normalized_score, left?.metrics.normalized_score)}</b><b>STEPS Δ {delta(right?.step, left?.step)}</b><b>HEALTH Δ {delta(right?.agent.health, left?.agent.health)}</b><b>INVALID Δ {delta(right?.metrics.invalid_actions, left?.metrics.invalid_actions)}</b><b>HAZARD Δ {delta(right?.metrics.hazard_damage_taken, left?.metrics.hazard_damage_taken)}</b></div></div>
      <div className="panel"><span>COMPLETED RUNS</span><ol>{completed.map((run) => <li key={run.run_id}><Link href={'/replay/' + run.run_id}>{run.terminal_reason ?? "completed"}</Link> · {run.step} steps · score {run.metrics.normalized_score.toFixed(1)}</li>)}</ol>{!completed.length && <p className="empty">No completed local runs yet.</p>}</div>
      {error && <p className="error">{error}</p>}
    </section>
  </main>;
}
