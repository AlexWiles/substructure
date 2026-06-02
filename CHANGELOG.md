# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/sdk`, `@substructure.ai/runtime`,
`@substructure.ai/cli`) and the `substructure-core` crate release together at the
same version.

## [Unreleased]

### Added

- Changelog with automated release notes.

### Fixed

- Tool-result messages (`message.new` with `role: "tool"`) now carry the tool
  `name` instead of `null`, so consumers reading only the message stream (e.g.
  reconstructing a transcript from session events) no longer have to correlate
  `tool_call_id` back to the originating tool call to recover the name.

## [0.1.14] - 2026-06-02

### Changed

- SDK typing and developer-experience improvements.

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.1.14...HEAD
[0.1.14]: https://github.com/substructureai/substructure/releases/tag/v0.1.14
