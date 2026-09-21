# Benchmarking

Run a repeatable local benchmark without a model key:

```powershell
python -m embodied_ai.cli benchmark --scenario survival_room --provider scripted --runs 20 --seed-start 1000 --concurrency 4 --server-url http://127.0.0.1:8080
```

The command writes `benchmark_summary.json`, Parquet datasets, and CSV mirrors to `data/exports/`. `--concurrency` is bounded to 1–32 and results remain ordered by seed. Authoritative remote benchmarks also write step trajectories, events, and decisions. A live Gemini benchmark records provider attempts, input/output/cached/total token counts, cumulative token use, and latency when available; unavailable values remain null rather than guessed.

Compare two summaries and export the directional deltas:

```powershell
python -m embodied_ai.cli compare --left data/baseline/benchmark_summary.json --right data/candidate/benchmark_summary.json --output data/comparison.json
```

Run the Rust API as a separate process for every provider and target it explicitly:

```powershell
.\scripts\dev.ps1
python -m embodied_ai.cli benchmark --provider mock_reasoning --runs 20 --server-url http://127.0.0.1:8080
```

The benchmark report records the selected provider. Python providers only propose actions; the Rust engine remains the authority for every benchmark.

Inspect a single exported JSONL trajectory locally without a server or model key:

```powershell
python -m embodied_ai.cli analyze --trajectory data/runs/<run-id>.trajectory.jsonl
python -m embodied_ai.cli filter --trajectory data/runs/<run-id>.trajectory.jsonl --valid-only --action-type move --output data/moves.csv --csv
python -m embodied_ai.cli verify-replay --replay data/runs/<run-id>.replay.json
python -m embodied_ai.cli check-reproducibility --left data/run-a.replay.json --right data/run-b.replay.json
```

The report validates contiguous decisions and reports action/event distributions, valid-action rate, available provider latency, and the final authoritative metric and score breakdown. The normalized score is transparent: task success contributes 100, health contributes up to 10, invalid actions cost 2 each, and every step costs 0.1; the final value is clamped to 0–100.

## Rust core performance

Replay `control` statistics measure non-overlapping wall-clock intervals. New runs default to provider control; the observer creates manual runs with `controller: "manual"`. Pausing starts paused time; the first paused manual step starts manual-control time until resume returns to the run's original controller. Terminal runs freeze all counters, and restoring a replay excludes offline downtime. These values include waiting/thinking time, unlike the earlier execution-only counters, so old exports are not directly comparable. `simulation_latency_us` stores engine execution latency for each submitted step, separately from the Python HTTP round-trip latency.

Agent context defaults to the five most recent actions. Set `--memory-window 1..100` on the run command, or use the observer's Memory window field, to change this bound. Observations retained by Python use the same bound; `memory_mode=none` still supplies no memory. Known facts remain independently capped at 12 in the provider payload. Policy-local state has a separate lifecycle: `--policy-state-mode reset` is the reproducible default, while `preserve` is an explicitly persisted option for a reused policy instance in a continual-memory experiment.

Benchmark exports separately record episode initialization latency: the local Python-to-Rust `create` or replay-restore request, including local HTTP/serialization overhead. It is not engine execution time and is never folded into `simulation_steps_per_second`. The simulation core is benchmarked separately from HTTP, rendering and provider latency:

```powershell
.\scripts\use-rust-env.ps1
Push-Location rust
cargo bench -p sim-core
Pop-Location
```

Criterion measures both Survival Room observation generation and a representative movement step. Results are machine-specific and are written under `rust/target/criterion/`.

## Local baseline

The latest checked local Windows baseline was captured with the command above on 2026-09-11:

| Benchmark | Criterion estimate |
| --- | --- |
| `survival_room/observation` | 1.5080–1.5582 µs |
| `survival_room/step_move` | 2.7601–2.9022 µs |

These measurements exclude HTTP, rendering, disk persistence, and provider latency. They are a regression reference for this machine, not a portable performance claim.
# Benchmarking protocol

Generalization reports evaluate a fixed provider over explicit train, validation, and test seed lists. Treat the generator configuration and observation mode as experimental inputs, not presentation settings: both are persisted in the immutable experiment manifest and report. Episode exports capture the actual Rust-generated room count and hazard families from each world manifest; split summaries and compatible-report comparisons stratify success rates by each mechanic and by their joint signature. Do not compare a restricted-perception result against an `oracle` result as if they measured the same task.

Before interpreting a report, run `audit-generalization`. It rejects empty or
overlapping seed splits, ambiguous shared versus per-split generator
configuration, and any supplied episode evidence that does not reconstruct the
claimed split summaries.

Generated reports include a SHA-256 fingerprint for their canonical shared
generator configuration, or one fingerprint per split when layouts differ.
The audit receipt exposes these identities without requiring a researcher to
visually compare JSON key order.

When two compatible reports both retain their per-episode rows,
`compare-generalization` additionally groups the exact shared seeds into paired
outcomes: both policies succeed, only the left succeeds, only the right
succeeds, or neither succeeds. This is descriptive paired evidence for the
fixed seed study, not a claim of statistical significance. When both reports
retain Rust-generated world manifests, comparison also requires the manifest to
match for every paired seed; it will not call different generated worlds a
policy difference.
