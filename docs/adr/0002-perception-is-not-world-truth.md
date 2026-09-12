# ADR 0002: Agent perception is not world truth

## Status

Accepted.

## Decision

The Rust engine models complete authoritative state, while `Observation` is a
sensor payload. `AgentObservation` excludes absolute agent position and
identity. Local NPC perception excludes hidden trust values. Researcher
snapshots remain privileged and are never used as provider input.

## Consequences

New sensors must preserve this boundary. Evaluator and researcher data may be
recorded for experiments, but it must be labelled separately from the exact
observation supplied to a policy.
