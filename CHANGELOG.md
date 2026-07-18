# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/runtime`, `@substructure.ai/cli`) and the
`substructure-core` crate release together at the same version.

## [Unreleased]

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

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/substructureai/substructure/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/substructureai/substructure/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/substructureai/substructure/compare/v0.1.22...v0.2.0
