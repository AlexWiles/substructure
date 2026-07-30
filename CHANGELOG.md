# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/runtime`, `@substructure.ai/cli`) and the
`substructure-core` crate release together at the same version.

## [Unreleased]

### Fixed

- Failed decisions now retry or end the run with the error.
- Decisions and LLM calls have default timeouts and retries.
- An LLM call that times out on its last attempt now settles the turn instead of stalling it.
- A turn now completes when its finalizer decision settles without a `done` action.
- The push transport no longer cuts long streams.
- A failed sub-agent and an unconfigured session now report errors.
- A tool result that arrives after its deadline no longer reports an error.

### Added

- A client submit can queue a message while a turn is active.
- Add a Slack adapter and a channel abstraction.
- Add MCP support.
- substructure.toml as a config file.
- `subs init <local|remote> [path]` writes a starter environment file.

### Changed

- `substructure.toml` now declares `target = "local"` or `target = "remote"`, and
  describes one environment: one file, one engine. Engine settings moved into
  `[worker]`, `[llm]`, `[run]`, and `[server]`; `org`/`app`/`url` are the remote
  half. There is no migration — a file without `target` is a parse error.

  ```toml
  # before
  worker_url = "http://localhost:4444"
  llm_provider = "anthropic"
  output = "pretty"
  port = 8080

  # after
  target = "local"
  [worker]
  url = "http://localhost:4444"
  [llm]
  provider = "anthropic"
  [run]
  output = "pretty"
  [server]
  port = 8080
  ```

- `subs mcp login` stores credentials in the environment's `db` instead of
  `credentials.toml`, so a login belongs to the environment that uses it. Run
  `subs mcp login <id>` once per environment; **gitignore `*.db*`**, which now
  holds credentials. `credentials.toml` keeps only `subs login` tokens, and the
  `credentials` config key and `--credentials` flag on `serve` are gone.
- `subs mcp add` is gone: declare `[mcp.<id>]` in the file. Ids and URLs are
  checked when the file is read.
- `subs webhook set` with no URL pushes the `[worker].url` the file declares.
- An `output` the file does not recognize is a parse error rather than a silent
  fall back to `ag-ui`.
- `subs link` writes to the file commands actually read (the discovered one, or
  `-c`) instead of always the working directory, and keeps a `url` it did not
  set.
- Rename the default database file to `substructure.db`.
- Rename the `--provider` option to `--llm-provider`.
- The `tool.call` action has no `handler` field; the engine finds where a call runs from its name.
- LLM calls and tool routing now use the agent config from the same decision.
- Effects and decisions queue in arrival order and dispatch when their prerequisites settle.
- Rename the decision events to the effect lifecycle names: `decision.queued`, `decision.dispatched`, `decision.completed`, `decision.errored`, `decision.dropped`.
- An LLM call waits for the connector fetches of its own decision and offers the fetched tools.
- One scheduler now decides all work: an effect past its deadline and a due retry settle at the next command, not only at a wake.
- A sub-agent effect is named by its child session. In `calls[]`, `id` is the child session and the new `tool_call_id` field gives the call the delegation answers.
- `calls[]` reports a connector fetch with the `connector_sync` kind.
- Each event of an effect names that effect with `id`, in place of `call_id`, `tool_call_id`, `session_id`, `connection_id` and `decision_id`.
- A client submit during an active turn now gives a conflict error and does not start a second turn.
- An AG-UI run that resumes an interrupt and sends messages now continues the resumed turn.
- An AG-UI run with a missing or partial `resume` for its open interrupts now ends with `RUN_ERROR`.
- A turn that starts while the previous turn finishes now completes the previous turn first.
- The session state has a new incompatible shape; delete existing database files.

## [0.2.3] - 2026-07-22

### Added

- Workers had no hook to run side effects (e.g. post a result) after a turn. Added a `turn.finished` engine→worker trigger carrying the turn's frozen output; the turn completes (`turn.completed` / `RUN_FINISHED`) only once the worker's finalizer settles, and a terminal finalizer failure completes the run as failed.

### Fixed

- A processor logged a store error at ERROR when its in-flight query was cancelled during runtime teardown, even before its cancel token was set. `spawn_blocking` join-cancellation now maps to `StoreError::Cancelled`, which the loop treats as a clean shutdown break.

### Changed

- The generic aggregate layer only served sessions. Session-specific execution and events now replace it.
- The event store round-tripped opaque JSON that every consumer re-parsed. The store now speaks typed session events end to end.
- Every event stored a full derived-state copy (message tree, prompts, decision triggers), growing storage quadratically. Events now carry a small bounded meta while the tree, anchored versions, and verbatim LLM prompts live in their own tables (new incompatible DB schema; delete existing `data.db` files).
- Decision triggers (full transcripts and LLM requests) were duplicated across queued, promoted, and retried decision events. The trigger now rides only the creating queued event; promotions and retries are id-only markers resolved from state at delivery.

## [0.2.2] - 2026-07-17

### Changed

- The engine sends an empty proposal instead of null.
- Fields the engine always emits were optional in the schema, forcing null
  checks in every generated client. They are now required on the wire.

### Added

- Added client facing `GET /api/client/sessions/{session_id}` returning status,
  open interrupts, and the full message tree and list as JSON.

### Fixed

- Regenerating recorded the new reply onto the old branch instead of forking.
  A decision whose view stops at an existing node now emits `head.moved`, rebasing
  the head so the reply records as a sibling branch.
- A client-tool round trip forked the tree when the resubmitted view raced the
  decision recording a worker tool's result, leaving a dangling duplicate branch.
  Tool echoes now also fold onto their recorded nodes at the decision-submit seam.
- The AG-UI connect endpoint required a `runId` although only `threadId` is read.
  Its body now needs `threadId` alone.

### Changed

- Better type names in protocol.rs, better generated type names.
- Interrupts are now anchored to the head that raised them and the session GET returns `interrupts[]` with head-resolved `status`.

## [0.2.1] - 2026-07-15

### Fixed

- The 0.2.0 packages pinned their platform `optionalDependencies` to a nonexistent
  0.1.22, so a clean install could not resolve the native binary. Bumped the pins
  to the release version.

### Changed

- The npm CLI wrapper booted Node on every command. A postinstall hardlinks the
  native binary over the shim so `subs` execs it directly, falling back to the Node
  shim when postinstall is skipped.

## [0.2.0] - 2026-07-15

### Changed

- **Breaking:** Pre-1.0 rework across the wire protocol, SDK, CLI, and docs. Dropped
  the TypeScript SDK for the raw wire protocol, flattened trigger/action names,
  renamed the CLI binary to `subs`, added native LLM providers with worker-run
  streaming, schema-validated tool I/O and a generated protocol spec, and rewrote
  the docs and examples.

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/substructureai/substructure/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/substructureai/substructure/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/substructureai/substructure/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/substructureai/substructure/compare/v0.1.22...v0.2.0
