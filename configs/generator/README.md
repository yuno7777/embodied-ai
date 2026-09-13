# Generator configuration examples

`compact-v1.json` is a complete WorldGeneratorConfig for generator version 1. It permits either supported compact topology: two or three rooms connected by open corridor doors.

`two-room-v1.json` and `three-room-v1.json` pin the room count. They make a layout-family experiment reproducible: use one as a training distribution and the other as a held-out topology, while retaining deterministic seed variation within each family.

`fire-hazard-v1.json` pins the generated hazard mechanic to fire. Use it as a rule-distribution shift against an electrical-hazard configuration while keeping the same compact map and key-to-exit task family.

For a generalization report, pass all three of `--train-generator-config`, `--validation-generator-config`, and `--test-generator-config`. The report and immutable experiment manifest preserve the selected config for each partition.

Supply every field when providing a config. The server accepts an omitted `config` as defaults, but treats a supplied configuration as explicit and versioned. `min_rooms` and `max_rooms` must be between two and three inclusive; use equal values to select one topology family. Three-room configurations require `min_width` of at least seven so every generated room has a walkable cell. `hazard_kinds` must be a non-empty subset of `electrical`, `fire`, and `toxic_gas`; its seeded selection and exact rule values are recorded in the world manifest.

Within a selected topology family, the seeded generator independently varies map dimensions, corridor row, spawn, key, water, container, hazard, NPC, and perturbation timing. It keeps the current V1 object vocabulary and key-to-exit task structure fixed; those are intentionally separate future composition axes.
