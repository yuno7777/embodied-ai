# Canonical wire schemas

These versioned JSON Schemas define the transport contract shared by providers, the Rust HTTP boundary, and the observer. `action-request.v1.json` is generated in structure from Python's `ActionRequest` model; its semantic per-action field rules are additionally enforced by both Python and Rust, because JSON Schema alone cannot conveniently express the discriminator-dependent field set without making the schema substantially harder to consume. `agent-decision.v1.json` composes that action contract with an optional, bounded operational-metadata envelope; it never permits free-form reasoning fields.

Update the schema, Python model, Rust strict decoder, and compatibility tests together when changing the protocol version.
