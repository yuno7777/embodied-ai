# Security

This is a local research system. Secrets are read only from environment variables and must never be written to runs, events, logs or exports. Provider responses are structured action proposals, not commands: no shell, tool, code-evaluation or arbitrary network access is exposed to the model. The engine validates action types, targets and world constraints before every state mutation.

The local Rust server accepts JSON bodies up to 64 KiB and allows browser requests only from `localhost:3000` and `127.0.0.1:3000`. Production follow-ups: add authentication/authorization where the server is not local-only, replace in-memory active runs with transactional persistence, and use an audited secret store.
