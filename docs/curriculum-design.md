# Deterministic Curriculum Proposal Interface (Design Only)

## Status and scope

This document specifies a future, evaluator-driven curriculum interface for the
Embodied AI harness.  It is deliberately a design, not a runtime feature: it
does not start training jobs, mutate configurations, generate worlds, call an
LLM, or add a service.

The purpose is to let a researcher use completed, provenance-checked
generalization results to select the *next* fixed generator configuration near
a policy's measured capability boundary.  Rust remains the authority that
validates every resulting generator configuration and constructs its worlds.

Non-goals:

- free-form natural-language scenario authoring or LLM-created curricula;
- adapting the held-out test distribution to a policy;
- silently overwriting generator files, reports, checkpoints, or manifests;
- automatic retraining or promotion of a policy;
- distributed scheduling, databases, Docker, or external orchestration.

## Required evidence

A proposal consumes immutable artifacts already produced by the harness:

- a validated generalization report, including its engine version, observation
  mode, generator configuration, partition seeds, and episode results;
- the experiment manifest and checkpoint identity that produced the report;
- world manifests or recorded `hazard_kinds` for episode-level environment
  mechanics; and
- explicit researcher constraints: permitted generator axes, a target success
  interval, maximum candidates, and a deterministic proposal seed.

It must reject a report that fails the same provenance requirements used by
`compare-generalization`: unsupported report version or observation mode,
missing engine version, overlapping partitions, or malformed distribution.
It must also reject any requested axis that is not declared as configurable.

## Candidate record

The eventual API should return immutable candidate records shaped like this:

```text
CurriculumCandidate
  candidate_id                 deterministic identifier
  parent_experiment_id         source experiment
  report_fingerprint           hash/identity of validated source report
  proposal_seed                explicit seed for candidate enumeration
  generator_config             complete proposed configuration, never a patch
  generator_config_fingerprint stable canonical fingerprint
  varied_axes                  only the allowed, recorded changes
  target_partition             train or validation; never held-out test
  rationale_metrics            aggregate measurements, not hidden reasoning
  acceptance_status            proposed | accepted | rejected | superseded
  rejection_reason             optional, researcher-supplied or validator code
```

`rationale_metrics` should be compact and auditable: episode count, success
rate with uncertainty, invalid-action rate, resource efficiency, hazard-family
breakdown, and the capability interval used.  It should not record private
chain-of-thought or unbounded narrative text.

The proposal result should also record rejected candidates and their mechanical
validation code, so a later run can reproduce both what was selected and what
was ruled out.

## Deterministic selection procedure

1. Validate the source report and bind the proposal to its experiment,
   distribution, observation mode, and engine version.
2. Aggregate validation (or explicitly designated curriculum-train) episodes
   by declared mechanics such as topology and `hazard_kinds`.  Held-out test
   episodes are measurement-only and are never used to choose candidates.
3. Find buckets that are challenging but not demonstrably unsolved.  A default
   researcher policy can target a conservative measured success interval such
   as 0.40--0.70, adjusted only through explicit constraints and uncertainty.
   Buckets below the interval inform diagnostics; they do not automatically
   cause a difficulty decrease.
4. Enumerate changes only across named generator axes.  Today those axes can
   include room bounds and allowed hazard families; future axes must be added
   to the Rust generator schema before they can appear here.  Sort axes and
   values canonically, then derive candidate seeds from the explicit proposal
   seed and candidate ordinal.
5. Materialize each candidate as a complete generator configuration, validate
   it with the Rust generator's normal configuration checks, and compute a
   stable fingerprint.  Invalid or duplicate candidates are retained as
   rejections, not repaired heuristically.
6. Rank valid candidates with a documented scalar made from public aggregate
   metrics (for example, distance to the target interval plus a penalty for
   poor invalid-action or resource behavior).  Ties resolve by canonical
   fingerprint, never wall-clock time or hash-map iteration order.
7. Emit proposals only.  A researcher must explicitly accept one, create a new
   experiment manifest, and invoke a normal training/evaluation command.

This makes two proposals identical when their source artifacts, constraints,
and proposal seed are identical.

## Safety and scientific boundaries

Every accepted candidate becomes a new immutable experiment lineage.  The
parent report, source checkpoint, candidate configuration, seed derivation,
and acceptance decision are recorded together.  Existing configuration files
remain unchanged.

The curriculum may inspect research snapshots and manifests as offline
evidence, but it never exposes privileged state to an acting policy.  Its
candidate generation is constrained to the existing Rust schema; it cannot
inject scripts, arbitrary maps, external assets, or untrusted scenario text.

The test partition stays fixed until an evaluation protocol explicitly creates
a new benchmark version.  Progress is therefore measured by frozen train /
validation choices and a separately protected held-out distribution, rather
than by moving the test set in response to results.

## Future API boundary

The intended pure, offline entry point is:

```text
propose_curriculum(validated_report, experiment_manifest, constraints)
  -> CurriculumProposal { candidates, rejections, proposal_manifest }
```

The `proposal_manifest` includes the versioned algorithm identifier and all
inputs necessary to reproduce the result.  It has no side effects.  A separate
future command may serialize this manifest, but acceptance, training, and
evaluation remain explicit user actions.

## Current foundation

The project already supplies the components this design relies on: deterministic
Rust generation, generator manifests, bounded hazard-family configurations,
provenance-checked generalization reports, hazard-stratified summaries, and
frozen tabular evaluation partitions.  It intentionally does not yet supply a
curriculum proposer or executor.
