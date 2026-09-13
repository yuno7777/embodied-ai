# Research readiness scorecard

Scores are deliberately conservative: 0 means absent, 5 means ready for sustained research use. They are not claims of agent intelligence.

| Area | Score | Evidence and limitation |
| --- | ---: | --- |
| Procedural worlds | 3/5 | Seeded two- and three-room corridor graphs, items, hazards, NPC, deterministic perturbation, manifest hashing and solvability checks. Arbitrary room graphs and broad task composition are not yet implemented. |
| RL interface | 3/5 | Rust-authoritative reset/step adapter, generated-world configuration, reward profiles, and a tabular-Q smoke baseline that can act on locally visible doors, containers, and items. No high-capacity policy baseline. |
| Generalization | 3/5 | Explicit disjoint seed plans, held-out evaluator, confidence intervals, and manifest-backed per-episode reports. No published multi-run learning comparison yet. |
| World-model datasets | 3/5 | Observation/action/next-observation transitions, optional privileged state, and strict experiment/world provenance. No large corpus or predictive-model training run yet. |
| Custom architecture integration | 3/5 | Stable policy/action/observation boundary, memory controls, and headless API. Neural architecture adapters remain future work. |
| Parallel simulation | 2/5 | Bounded local parallel benchmark and a measured 1/8-worker smoke run. 32/64-worker behavior and memory-per-environment are not characterized. |
| Reproducibility | 4/5 | Versioned Rust simulation, deterministic generator, manifest persistence, replay reconstruction, input fingerprints, and provenance checks. Cross-machine reproducibility has not yet been tested. |

## Next milestone

Improve the tabular baseline's action proposal mechanism so it can safely act on locally observed object IDs, then run a documented train/validation/test comparison across procedural seeds. This is the smallest next step that would turn the existing infrastructure-validation baseline into a meaningful learning experiment.
