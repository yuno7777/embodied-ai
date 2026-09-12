# Changelog

## Unreleased

- Fixed observations and discovery timestamps omitting additional items on the same cell; closed-container contents remain hidden until opened.
- Made item/door/container progress rewards one-time per entity, added entity-specific milestone steps, and prevented repeated pickup from inflating useful-item counts.
- Added and ran a real-Chrome browser smoke script covering manual control, live updates, event filtering, replay navigation, read-only controls, themes, mobile overflow, and page errors. Uses an isolated browser session without downloading a browser binary.
- Fixed closed freestanding doors allowing movement through them, and unlocked doors incorrectly demanding their configured key. Added a regression covering blocked movement, key-free opening, and traversal.
- Fixed reusable consumables, explicit pickup of colocated items, and inactive hazards incorrectly blocking rest; added regression tests for each case.
- Isolated smoke-test Cargo artifacts from the running development server to avoid Windows executable locks. The smoke command now restores its port environment and working directory and checks control metrics and terminal replay stability.
- Corrected control-time accounting to track wall-clock provider/manual/paused intervals, added per-step engine latency in microseconds, and froze counters at terminal state. Manual observer runs now explicitly identify their controller.
- Made repeated terminal abort/provider-error/stop requests idempotent and rejected pause/resume on completed runs.
- Preserved event types in live and replay views, added combined type/text filtering, retained older replay events, and deduplicated live events by ID.
- Isolated replay playback from live updates and disabled replay mutation controls. Live step broadcasts now include the researcher snapshot so map and vitals update with each action.
- Added typed and toggleable hazards, weighted inventory, flashlight and bandage effects, locked containers, item giving, richer NPC dialogue/trust/patrol behavior, scenario time limits, and explicit reward breakdowns.
- Added coverage, action diversity/repetition, resource efficiency, discovery timing, milestone, carried-weight, and score-component metrics.
- Added cautious and explorer baselines, bounded benchmark concurrency, CSV exports, benchmark comparison, trajectory filtering, replay verification, live-run continuation, provider retry telemetry, and wall-clock/token budgets.
- Expanded the observer with run configuration, service health, richer maps, resource charts, event/replay filtering, a timeline scrubber, export downloads, keyboard controls, responsive layout, and light/dark themes.
- Added safe export serving, strict scenario/action validation, deterministic edge-case tests, and a one-command launcher for three independent local processes.
