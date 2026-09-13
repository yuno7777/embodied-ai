# Embodied Intelligence Research Environment

This repository is a headless-first symbolic embodied research harness. It is not an AGI system and it does not yet include a learning baseline.

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
- Catalog scenarios and generated worlds are deterministic, versioned, and replayable.
- Generated worlds contain a deterministic two- or three-room corridor graph, interior dividers, traversable corridors, a deterministic hazard-cooldown perturbation, and constraint-validated key-to-exit routes.
- Train, validation, and test seed partitions are explicit and required to be disjoint in both Rust and Python evaluation planning.
- The Python `EmbodiedEnv` implements reset/step semantics over the Rust service, including generated-world and reward-profile requests.
- Replays persist generated manifests and reward configurations; replay verification checks these inputs before declaring two executions comparable.
- Generalization evaluation reports train/validation/test success, Wilson intervals, reward, invalid-action rate, exploration, efficiency, failures, and held-out gaps.
- Benchmark reporting separates authoritative simulation throughput from control and provider latency.
- A lightweight tabular Q-learning baseline trains through the Rust environment, persists a JSON checkpoint, reloads it, and evaluates greedily on held-out generated seeds. The locally visible-object action path was smoke-tested with two train and two held-out episodes against a local Rust process; this was lifecycle verification, not a performance claim.

## Partial

- The procedural generator is deterministic and constraint checked, but currently produces compact two- and three-room corridor graphs rather than arbitrary room graphs or broad task compositions.
- Trajectories export observations, actions, events, reward, and optional privileged transition data. Per-step agent metadata remains limited to provider-safe operational metadata.
- Parallel episode execution is supported by bounded local threads for benchmarks. A first 1/8-worker local measurement is recorded in [performance notes](performance-notes.md); 32/64-worker levels remain unmeasured.
- The observer UI remains an observer/researcher console; it is not required for headless execution. It displays generated-world and reward metadata only in its privileged researcher panel.

## Not implemented

- The tabular Q baseline can propose open/pickup actions for locally visible object IDs, but it remains deliberately small and is an infrastructure baseline rather than a strong task-solving agent.
- Noisy/vision/depth/audio sensors.
- Automatic curriculum generation, distributed infrastructure, unrestricted external tools, or 3D physics.

## Reproducibility contract

World determinism requires the engine version, scenario or full generated-world manifest, observation mode, reward configuration, seed, maximum steps, and ordered action stream. Agent decision reproducibility is a separate question: an external provider can make different decisions even when the world replays identically.
