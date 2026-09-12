# Local preview release notes

This is an unreleased local preview, not a production or full-goal completion claim. Docker, PostgreSQL, GPU models, and CI remain outside this release's setup.

## Recent changes

- Event filtering combines event type with case-insensitive text search. Saved replay searches include older events; live history retains the latest 1,000 events and deduplicates authoritative event IDs.
- Loaded replays are read-only and detached from the live WebSocket. Both normal and manual step broadcasts include current researcher snapshots.
- Manual runs identify their controller at creation. Replay control metrics now measure wall-clock ownership intervals rather than engine execution duration.
- Engine execution latency is stored separately as `control.simulation_latency_us`, ordered by submitted action. Python step latency remains the HTTP round-trip measurement.
- Terminal control clocks freeze, repeated stop/abort/error requests preserve the existing terminal replay, and completed runs reject pause/resume.

## Compatibility

Existing create-run clients default to provider control. New manual clients should send `controller: "manual"` when creating a run. No action-schema changes are required.

Old replays remain readable, but their earlier execution-only control times are not comparable with new wall-clock values. Restored runs retain saved totals and start a new provider-control interval; offline downtime is excluded. A pause counts as paused time until the first manual step, after which time belongs to manual control until resume.

Restart the Rust server and refresh the observer together to use the updated live snapshot messages and controller field.

## Verification and outstanding work

Run `scripts/test_all.ps1` for Python, Rust, observer tests, formatting, lint, type checking, and the production build. Run `scripts/smoke_test.ps1` for the individual-process Rust/Python integration check.

The automated `scripts/browser-smoke.ps1` check has passed against local Chrome: manual steps, live updates, event filters, replay navigation, read-only controls, theme switching, mobile overflow, and page errors. It saves screenshots under `data/browser-smoke/`. The full requirement-by-requirement completion audit remains outstanding; this smoke test does not cover every UI workflow.
