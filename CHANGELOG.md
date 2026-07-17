# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/runtime`, `@substructure.ai/cli`) and the
`substructure-core` crate release together at the same version.

## [Unreleased]

### Changed

- `DecisionRequest.proposed` was nullable, forcing every worker to null-check
  before echoing. The engine now always sends a proposal — empty when it needs
  worker knowledge — so `return proposed` is a complete worker.
- Fields the engine always emits (`Message.tool_calls`, `MessageTree.nodes`,
  `WorkerIdentity.metadata`, `TokenDelta.tool_calls`, trigger `new_from`/`client`/
  `stream`/`truncated`) were optional in the schema, forcing null checks in every
  generated client. They are now required on the wire; sessions persisted by older
  versions no longer deserialize.

### Added

- Browsers had no way to read an existing session's history, forcing frontends to
  scrape the AG-UI connect SSE stream. Added `GET /api/client/sessions/{session_id}`
  returning status, open interrupts, and the full message tree as JSON.
- Rendering history from the session's message tree forced every client to walk
  parent pointers. `GET /api/client/sessions/{session_id}` now also returns
  `messages`, the root→head lineage ready to render.

### Fixed

- A worker answering `null` — the natural "nothing to add" reply in JSON languages —
  failed to parse and the decision retried forever. A `null` response body or
  `decision.result` frame now reads as the empty decision.
- Regenerating recorded the new reply onto the old branch instead of forking: a
  truncated client view wrote nothing, so the head stayed on the abandoned leaf.
  A decision whose view stops at an existing node now emits `head.moved`, rebasing
  the head so the reply records as a sibling branch.
- A client-tool round trip forked the tree when the resubmitted view raced the
  decision recording a worker tool's result, leaving a dangling duplicate branch.
  Tool echoes now also fold onto their recorded nodes at the decision-submit seam.
- The AG-UI connect endpoint required a `runId` although only `threadId` is read.
  Its body now needs `threadId` alone.

### Changed

- Code generators named protocol types after the referencing property, producing
  mangled names like `DecisionResponseClass`. Added `#[schemars(title)]` to every
  wire type so each `$defs` entry carries its own name.
- Control nodes were a vestigial tree marker with no producer, complicating every
  tree walk. Removed `Node`/`Control`/`ControlKind`; `MessageTree.nodes` is now a
  plain `NewMessage[]`.
- Session-global interrupts blocked every branch and mismatched clients that pin
  interrupts to messages, so editing away from a parked question was impossible.
  Interrupts are now anchored to the head that raised them and the session GET returns `interrupts[]` with head-resolved `status`.

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

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/substructureai/substructure/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/substructureai/substructure/compare/v0.1.22...v0.2.0
