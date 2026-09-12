# Scenario authoring

V1 scenarios are JSON data consumed by the Rust `Scenario` type. Keep world rules in `sim-core`; files declare geometry and values rather than executable logic. The bundled example is [scenario.rust.json](../scenarios/survival_room/scenario.rust.json).

Required fields are `id`, `name`, `version`, `width`, `height`, `spawn`, `max_steps`, `vision_radius`, and `goal`. Optional collections include `walls`, `items`, `doors`, `hazards`, `containers`, `npc`, and `perturbations`.

Items have an ID, name, type, position, weight, optional vision bonus, optional status cures, health/energy/hydration restoration, and `consumable`. Doors and containers can declare an open/locked state and `key_id`; containers list the IDs they conceal. Hazards declare a kind, active state, health/resource drains, and optional status effect. NPCs support disposition, trust, sequential dialogue, inventory, and deterministic patrol positions. Every entity ID must be unique, references must resolve, and all coordinates must be inside the grid.

Perturbations are deterministic state changes selected by the run seed. Effects can close a door, activate/deactivate a hazard, or move the NPC (with a `destination`):

```json
{
  "id": "service_door_auto_close",
  "step": 18,
  "effect": "close_door",
  "target_id": "service_door",
  "probability_per_mille": 500
}
```

`probability_per_mille` is 0–1000 and defaults to 1000. Its seeded outcome is preserved in the event log and replay. `time_limit` optionally caps simulated action time independently of `max_steps`. Add a scenario through a Rust loader/registry change, then run `cargo test --workspace` to validate it. Future folders such as `scenarios/zombie_apocalypse/`, `scenarios/europa_station/`, and `scenarios/disaster_rescue/` can reuse these generic grid entities and extend the core only when new rules are genuinely required.
