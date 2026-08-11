# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The `@substructure.ai/cli` package and the `substructure-core` crate release
together at the same version.

## [Unreleased]

### Changed

- Improve MCP auth failure handling.
- Standardize usage reporting.
- Clearer split between remote/local command handling
- Split the machine caller into an API key caller and an admin caller. Only an
  API key answers a worker decision.
- A session owner now records its kind. An end user opens only a session an end
  user owns.
- Add the `admin` interrupt origin. It has more privilege than `machine`.

### Added

- CLI run works against a remote server
- Slackbot prompts for MCP reauth by default. Overrideable in config.
- Engine driven deferred tool definition support.
- Calls to Anthropic, OpenAI, and OpenRouter now cache the prompt.
- Streamed calls now report their token counts, and the cached part of the prompt.
- Support cache_ttl on llm config blocks in the manifest.

## [0.3.1] - 2026-08-07

### Added

- Add support for MCP token auth.

### Changed

## [0.3.0] - 2026-08-06

### Changed

- The changelog starts again at this release. For the changes before it, refer
  to the git history.

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/substructureai/substructure/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/substructureai/substructure/compare/v0.2.3...v0.3.0
