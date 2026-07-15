# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/runtime`, `@substructure.ai/cli`) and the
`substructure-core` crate release together at the same version.

## [Unreleased]

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

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/substructureai/substructure/compare/v0.1.22...v0.2.0
