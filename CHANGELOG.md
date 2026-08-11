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
- The `[remote]` section now decides where every command acts. A file that names
  none describes an engine here, and the commands read it.
- `subs sessions list` and `subs sessions events` read the local database when
  the file names no `[remote]`. `--db` selects a database.
- `subs agents list`/`show` and `subs llm list` read the file when it names no
  `[remote]`, and report the variables this machine holds.
- Commands that only a deployment can answer now say the file names no
  `[remote]` instead of asking you to log in to the default one.
- `subs agents secret`/`rotate-secret` and `subs llm set-key`/`delete-key` name
  the environment variable to set when the engine runs here.
- `subs run` sends the turn to the deployment when the file names a `[remote]`,
  and streams the reply as it is written. It uses the credential you logged in
  with.

### Added

- Add `POST /api/v1/projects/{project}/run`: run one turn and stream it, for an
  operator credential. `?format=events` streams the engine's own events.
- `subs sessions events` takes `-o`, and replays a session as text with
  `-o pretty`.
- A command whose output goes into a closed pipe now stops without an error.
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
