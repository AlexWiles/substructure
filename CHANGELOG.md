# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The `@substructure.ai/cli` package and the `substructure-core` crate release
together at the same version.

## [Unreleased]

### Added

- Support approval interrupts on destructive MCP tools.
- Add agent plugins: `[plugin.<id>]` declares a plugin directory, and an agent uses it with `plugins`.
- Add the `skill` tool. It loads a skill's instructions and its files.
- Show a plugin catalog in the system prompt.

### Fixed

## [0.6.0] - 2026-08-14

### Added

- MCP revision 2026-07-28.
- Richer tool result support.
- Slack messages with audio and video reach the model.

### Fixed

- A model call that gets no answer is tried again.

## [0.5.0] - 2026-08-13

### Added

- Agents can set `effort` in the manifest.
- Slack messages with images, PDFs, and text files reach the model.
- Slack replies carry the images the model makes.
- Add a blob store for message attachments and generated images. The local engine keeps the bytes in its database.
- Add a client endpoint that serves stored blobs.

### Fixed

- `subs serve --slack-agent` stops at startup if the name is not a declared agent.
- A model that refuses a request stops the run, instead of answering it with nothing.
- Anthropic calls send the reasoning fields, the output limit, and the sampling that the model reads.
- OpenAI calls read the whole model name, and keep `temperature` for chat models and at no effort.
- OpenRouter calls keep a deferred tool out of the request, as the other providers do.
- An uauthorized mcp interrupts the session.
- Streamed calls now read `data:` lines that have no space after the colon.
- Responses keep the model's reasoning, and Anthropic and OpenRouter calls send it back.
- Calls to OpenAI-compatible providers no longer send engine-internal message fields.
- Engine context added inline keeps a message's image parts.

## [0.4.1] - 2026-08-11

### Changed

- Admin caller renamed to Operator.

### Fixed

- Session streams release their subscription when the reader goes away

## [0.4.0] - 2026-08-11

### Changed

- Improve MCP auth failure handling.
- Standardize usage reporting.
- Clearer split between remote/local command handling
- Split the machine caller into an API key caller and an admin caller.

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

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/substructureai/substructure/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/substructureai/substructure/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/substructureai/substructure/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/substructureai/substructure/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/substructureai/substructure/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/substructureai/substructure/compare/v0.2.3...v0.3.0
