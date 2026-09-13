# Protocol v1

The canonical protocol version is `1`. A provider receives an `Observation` and returns one `ActionRequest`; it never receives a `WorldSnapshot` or a mutation channel. `Observation.agent` is an `AgentObservation`, deliberately excluding agent identity and absolute position. Local NPC observations exclude hidden trust values, and room identifiers are researcher-only. `StepResult` contains the new filtered observation, reward, terminal information and structured events. `WorldSnapshot` is researcher-only.

The committed [WorldManifest JSON Schema](../schemas/world-manifest.v1.json), [Observation JSON Schema](../schemas/observation.v1.json), [ActionResult JSON Schema](../schemas/action-result.v1.json), [ActionRequest JSON Schema](../schemas/action-request.v1.json), and [AgentDecision JSON Schema](../schemas/agent-decision.v1.json) are the versioned wire artifacts. Python validates every Rust HTTP observation, action result, and generated-world manifest before they can reach orchestration or a provider; the models reject extra researcher/debug fields and unknown generator settings. Rust remains the final authority for world generation, observation construction, evaluation, and action validation.

Action values are validated at the Python provider boundary and again by the Rust engine. Wire schemas are defined in `embodied_ai/schemas.py`; the Rust equivalents live in `rust/crates/sim-core`.

## Run control

`POST /api/runs/{id}/pause` blocks the normal `POST /api/runs/{id}/step` endpoint so an automated provider cannot advance the run. While paused, a researcher may submit exactly one validated action through `POST /api/runs/{id}/manual-step`; the run remains paused after that action. `manual-step` returns `409 Conflict` if the run is not paused. This keeps pause/resume control separate from human baseline interaction.

`POST /api/runs/{id}/provider-error` is reserved for orchestration when a provider call fails after a run has been created. It terminates the run with `terminal_reason = "provider_error"`, records a `RunInterrupted` event, and persists the partial replay.

`POST /api/runs/{id}/stop` accepts only `client_timeout` or `token_budget_exhausted`. This lets the Python runner end a budget-limited run through the Rust authority instead of reporting a local-only outcome. `GET /api/runs/{id}/observation` returns the current filtered observation and enables a provider to continue an existing nonterminal run without creating a replacement.

Each replay pairs its initial and per-step researcher `timeline` snapshots with an equally indexed `observations` array and exact structured `actions` stream. Those observations are the exact filtered payloads visible to the agent at the corresponding moment, so playback never exposes hidden-state information as agent perception. Active persisted replays can be reconstructed under a new run ID through `POST /api/replays/{id}/resume`; restore refuses terminal, incomplete, version-mismatched, or state-divergent replays.

Each state-changing server endpoint persists its event log and replay before returning success. A filesystem or serialization failure is logged and returned as an HTTP `500`; the server does not silently claim that an unreplayable transition was saved.

`GET /api/scenarios/{id}` returns the validated scenario definition for an available scenario. `GET /api/benchmarks` returns authoritative aggregates over persisted and live run snapshots; it reports completed-run outcome, resource, score, invalid-action, hazard, simulated-time, provider-call, available decision-latency, and supplied input/output-token summaries without fabricating token or cost data.

`POST /api/worlds/generate` and `generated_world` run requests may include a `partition` of `train`, `validation`, or `test`. When supplied, Rust verifies that the generated-world seed belongs to the named built-in distribution before generation. This is an optional assertion for standard experiments, not a replacement for explicitly recorded custom seed-list distributions.

Before the runner submits a model action, it records `POST /api/runs/{id}/decision` with the strictly validated action, concise decision summary, provider/model identity, measured provider latency, and optional SDK token usage. A decision may additionally carry bounded operational `agent_metadata`: confidence in `[0, 1]`, a finite value estimate, non-negative policy entropy, and concise planner telemetry (name, expanded-node count, and planning time). It never accepts free-form reasoning, hidden state, or a world mutation request. The server broadcasts an `agent_decision` message and persists the record in the replay; it remains non-authoritative until the normal step endpoint validates and executes the action.

Custom synchronous policies may implement `decide(observation) -> AgentDecision`; `CallablePolicy` supports either that complete result or an action-only result. This is equivalent to the asynchronous provider decision boundary after validation, and it remains limited to the same filtered observation and Rust action endpoint.

`GET /api/runs/{id}/status` exposes only orchestration control state (`paused`, `done`, terminal reason, and step). Provider workers check it before every model request, so a paused run waits without consuming another provider turn and an aborted run exits cleanly.

Python trajectory exports record `agent_context` beside each observation. It is the compact context payload supplied to the provider immediately before that decision, never an authoritative snapshot or hidden-state projection.

Policy-local state is explicitly separate from that runner-owned context. `run_remote` and the `run` CLI record `policy_state_mode`: `reset` (the default) invokes the policy reset hook with the run seed; `preserve` leaves a reused policy instance intact for a deliberately labeled continual-memory experiment. Neither mode changes Rust world reset semantics, and each trajectory row plus experiment manifest records the selected mode.

Persisted experiment manifests include a canonical fingerprint. `audit-experiment` verifies that fingerprint locally and can bind a JSONL trajectory to the exact experiment ID, observation mode, generated-world manifest, and policy-state mode. This is provenance validation, not a replay substitute: Rust replay reconstruction remains the authority for world-transition determinism.

For offline world-model research, the headless CLI may be invoked with `--include-research-snapshots`. That explicit local opt-in records a before/after `WorldSnapshot` pair beside each transition as `research_snapshot` and `next_research_snapshot`; it does not alter either the observation or provider context. Dataset conversion excludes the privileged pair unless its caller explicitly requests privileged state.

`embodied-ai audit-dataset --trajectory <run.jsonl>` provides a local coverage report for exported transitions, action/outcome mix, sensor modes, policy-state lifecycle modes, metadata-field presence, exact next-observation availability, and complete versus partial privileged snapshot pairs. It reports metadata coverage only, never metadata values or privileged snapshots.
