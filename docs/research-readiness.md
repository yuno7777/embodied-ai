# Research readiness scorecard

Scores are deliberately conservative: 0 means absent, 5 means ready for sustained research use. They are not claims of agent intelligence.

| Area | Score | Evidence and limitation |
| --- | ---: | --- |
| Procedural worlds | 3/5 | Seeded two- and three-room corridor graphs, items, hazards, NPC, deterministic perturbation, manifest hashing and solvability checks. Arbitrary room graphs and broad task composition are not yet implemented. |
| RL interface | 3/5 | Rust-authoritative reset/step adapter, generated-world configuration, reward profiles, and a tabular-Q smoke baseline that can act on locally visible doors, containers, and items. No high-capacity policy baseline. |
| Generalization | 3/5 | Explicit disjoint seed plans, held-out evaluator, manifest-backed per-episode reports, and Wilson intervals at both split and generated-mechanics slice level. No published multi-run learning comparison yet. |
| World-model datasets | 3/5 | Observation/action/next-observation transitions, optional privileged state, and strict experiment/world provenance. No large corpus or predictive-model training run yet. |
| Custom architecture integration | 3/5 | Stable policy/action/observation boundary, memory controls, and headless API. Neural architecture adapters remain future work. |
| Parallel simulation | 2/5 | Bounded local parallel benchmark and a measured 1/8-worker smoke run. 32/64-worker behavior and memory-per-environment are not characterized. |
| Reproducibility | 4/5 | Versioned Rust simulation, deterministic generator, manifest persistence, replay reconstruction, input fingerprints, and provenance checks. Cross-machine reproducibility has not yet been tested. |

## Next milestone

Run and document a repeated tabular-Q train/validation/test study across fixed procedural seeds and registered topology/hazard shifts. This will turn the existing lifecycle workflow into a more meaningful learning experiment without overstating its capability.
