# ADR 0001: Rust is the simulation authority

## Status

Accepted.

## Decision

`rust/crates/sim-core` owns world transition, seeded randomness, validation,
reward inputs, and replay reconstruction. Python policies and the web observer
communicate through typed Rust API contracts only.

## Consequences

Python remains responsible for provider orchestration, analysis, and exports;
it must not gain a second production simulator. The historical Python engine is
quarantined pending migration-fixture replacement.
