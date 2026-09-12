# Legacy systems quarantine

Rust `sim-core` is the sole supported simulation authority. The Python package,
the agent-control service, benchmarks, and the Next.js observer communicate with
it through the Rust HTTP action boundary.

The following are retained only as migration fixtures and are not production
entry points:

- `embodied_ai.engine`: the original Python grid simulator and its
  `scenarios/survival_room/scenario.json` input.
- `frontend/index.html`, `frontend/app.js`, and `frontend/style.css`: the
  original static observer, superseded by `frontend/observer`.

No production module may import the Python engine. `tests/test_engine.py`
preserves characterization coverage while the canonical Rust scenario and
contracts are expanded. Removing the quarantined code requires replacing those
fixtures with Rust compatibility/replay fixtures first.
