# Generator configuration examples

`compact-v1.json` is a complete WorldGeneratorConfig for generator version 1. It was accepted by the authoritative Rust `POST /api/worlds/generate` endpoint with seed 42 on 2026-09-13, producing a solvable 9×8 world.

Supply every field when providing a config: the server accepts an omitted `config` as defaults, but treats a supplied configuration as explicit and versioned.
