# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/runtime`, `@substructure.ai/cli`) and the
`substructure-core` crate release together at the same version.

## [Unreleased]

### Changed

- **Breaking:** Pre-1.0 rework across the wire protocol, SDK, CLI, and docs. Dropped
  the TypeScript SDK for the raw wire protocol, flattened trigger/action names,
  renamed the CLI binary to `subs`, added native LLM providers with worker-run
  streaming, schema-validated tool I/O and a generated protocol spec, and rewrote
  the docs and examples.

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.1.22...HEAD
