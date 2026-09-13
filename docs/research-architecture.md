# Embodied Intelligence Research Environment

This repository is a headless-first symbolic embodied research harness. It is not an AGI system. Its tabular-Q learner is an intentionally small lifecycle baseline, not a claim of general-purpose embodied intelligence.

```text
Policy / provider / future learner
        │ AgentObservation + typed ActionRequest
        ▼
Python control plane ───────────────┐
        │ HTTP                       │ experiment, trajectory, benchmark exports
        ▼                             ▼
Rust sim-server ───────────► persisted replay + immutable world manifest
        │
        ▼
Rust sim-core (authoritative transition, evaluator, seeded RNG)
        │
        ├── catalog scenario
        └── deterministic procedural generator
              ├── rooms + corridor
              ├── key, exit, resources, hazard, NPC
              └── solvability validation
```

## Implemented

- Rust is the only simulation authority. Python submits typed actions and does not simulate world state.
- Normal agent observations exclude researcher coordinates, hidden contents, and NPC trust. Local perception propagates only through reachable floor/open-door cells, so walls and closed doors block entities behind them; global ambient and perturbation events remain researcher-only. Research snapshots and evaluator output are separate.
- Sensor ablations are explicit and replayable: `minimal`, `normal` local symbolic, `rich` extended local, `noisy` local symbolic with deterministic 20% cell dropout, and opt-in `oracle` full-map symbolic baseline. Oracle is not researcher state and must be reported separately from restricted-perception results.
- Catalog scenarios and generated worlds are deterministic, versioned, and replayable.
- Generated worlds contain a deterministic two- or three-room corridor graph, interior dividers, traversable corridors, a deterministic hazard-cooldown perturbation, and constraint-validated key-to-exit routes. Within that family, seeded generation varies dimensions, corridor row, spawn, key, water, container, hazard, NPC, and perturbation timing. Generator configuration can pin either topology family and electrical, fire, or toxic-gas hazard families for reproducible distribution-shift experiments.
- Train, validation, and test seed partitions are explicit and required to be disjoint in both Rust and Python evaluation planning.
- The Python `EmbodiedEnv` implements reset/step semantics over the Rust service, including generated-world and reward-profile requests.
- Replays persist generated manifests and reward configurations; replay verification checks these inputs before declaring two executions comparable.
- Generalization evaluation reports train/validation/test success, Wilson intervals, reward, invalid-action rate, exploration, efficiency, failures, and held-out gaps. It records actual generated room counts and hazard kinds from the Rust manifest, then reports uncertainty-bearing mechanics slices without inferring mechanics from a requested config.
- Benchmark reporting separates authoritative simulation throughput from control and provider latency.
- A lightweight tabular Q-learning baseline trains through the Rust environment, persists a JSON checkpoint, reloads it, and evaluates greedily on explicit disjoint train/validation/test seed partitions. The locally visible-object action path was smoke-tested with two train and two held-out episodes against a local Rust process; this was lifecycle verification, not a performance claim.
- `CallablePolicy` provides a narrow adapter for custom planners or future learned policies: it receives only a public observation and returns a validated typed action or `AgentDecision` through the same Rust authority boundary. The latter can carry only bounded operational metadata.
- Decision records may persist bounded confidence, value, entropy, and planner telemetry for later ablations. This metadata is observational only: it cannot alter the Rust transition, evaluator, or agent observation, and it does not collect chain-of-thought.
- `public_action_candidates` is the shared public-observation helper for lightweight policies. It proposes only permitted basic actions, adjacent openable targets, co-located visible items, and carried items; Rust still validates every submitted action.

## Partial

- The procedural generator is deterministic and constraint checked, but currently produces compact two- and three-room corridor graphs rather than arbitrary room graphs or broad task compositions.
- Trajectories export observations, actions, events, reward, and optional privileged transition data. Per-step agent metadata remains limited to provider-safe operational metadata.
- Parallel episode execution is supported by bounded local threads for benchmarks. A first 1/8-worker local measurement is recorded in [performance notes](performance-notes.md); 32/64-worker levels remain unmeasured.
- The observer UI remains an observer/researcher console; it is not required for headless execution. It displays generated-world/reward metadata in its privileged researcher panel and bounded decision telemetry in its read-only agent-decision panel.

## Not implemented

- The tabular Q baseline can propose open/pickup actions only for reachable locally observed object IDs and use public carried-item IDs, but it remains deliberately small and is an infrastructure baseline rather than a strong task-solving agent.
- Vision/depth/audio sensors and multimodal perception.
- Automatic curriculum generation, distributed infrastructure, unrestricted external tools, or 3D physics.

## Reproducibility contract

World determinism requires the engine version, scenario or full generated-world manifest, observation mode, reward configuration, seed, maximum steps, and ordered action stream. Agent decision reproducibility is a separate question: an external provider can make different decisions even when the world replays identically.
