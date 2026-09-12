# Contributing

Keep the authority boundary intact: changes to world state belong in `rust/crates/sim-core`; provider and research orchestration belongs in `embodied_ai`; the observer must consume snapshots and observations rather than encode rules. Add deterministic tests for new rules and never require a live model key in the normal suite.
