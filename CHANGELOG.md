# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/sdk`, `@substructure.ai/runtime`,
`@substructure.ai/cli`) and the `substructure-core` crate release together at the
same version.

## [Unreleased]

## [0.1.19] - 2026-06-12

### Added

- `substructure new [template] [dir]` scaffolds a project by copying a starter
  template from the monorepo. With no template it shows a picker (or lists
  templates when non-interactive).

## [0.1.18] - 2026-06-10

### Removed

- The CLI no longer prints zero-balance warnings when targeting an app. App
  balances are still shown by `apps list` and `apps show`.

### Changed

- Interrupts and cancels now fully fence the session: pending work is voided,
  late results and client actions can't wake a paused session, and decision
  requests raised mid-pause queue until resume.

## [0.1.17] - 2026-06-10

### Added

- `openaiAgent` factory in the OpenAI adapter: converts an `@openai/agents`
  `Agent` (or `OpenAIAgentSettings`) without a second `new`, e.g.
  `openaiAgent(new Agent({ ... }))`.
- Interrupts record the issuing caller's origin; resuming requires equal or
  higher privilege.
- Clients can interrupt and resume their own sessions via the client API.
- New `interrupt` worker action lets an agent pause awaiting external input.
- AG-UI interrupt-aware run lifecycle: interrupt outcomes and `resume[]`.

## [0.1.16] - 2026-06-09

### Added

- Native AG-UI protocol support: endpoints that stream an agent turn as AG-UI SSE
  events, with live token and reasoning streaming.
- Reasoning controls on `LlmRequest` (`reasoning`: `effort` or `max_tokens`,
  plus `exclude` and `enabled`), passed through to providers that support them.
- `llmToolLoop` takes a `generator`: a worker-side provider generator that runs
  the LLM call on your worker, or `serverGenerate` to let the Substructure
  server's configured provider make the call.
- Anthropic adapter (`@substructure.ai/sdk/adapters/anthropic`): call the
  Anthropic Messages API from a worker via the `anthropicGenerate` generator.
- AI SDK adapter (`@substructure.ai/sdk/adapters/ai`): run an existing Vercel AI
  SDK agent on Substructure via `ToolLoopAgent`, or drive the loop directly with
  the `aiGenerate` generator.
- OpenAI adapter (`@substructure.ai/sdk/adapters/openai`): run an `@openai/agents`
  `Agent` on Substructure via `OpenAIAgent`, or drive the loop directly with the
  `openaiGenerate` generator.
- The server and embedded runtime can start without an LLM provider configured.
- `$SUBS_API_TOKEN` sets the CLI's bearer token without a login — for targeting a
  self-hosted server with auth enabled (a `serve --dev` server needs no token).
- `substructure login` now stores tokens per server URL, so you can stay logged
  in to several servers at once (e.g. cloud and a staging deploy); each command
  sends only the token for the server it targets.

### Changed

- **Breaking (CLI):** subcommands are now top-level — `substructure cloud <cmd>`
  becomes `substructure <cmd>` (e.g. `substructure sessions list`), and
  `substructure local start` becomes `substructure serve`.
- **Breaking (CLI):** the credentials file is now keyed by server URL with no
  migration of the old single top-level `token`; existing logins must re-run
  `substructure login`.
- **Breaking:** session events identify a session's end user as `owner` (was
  `identity`); consumers reading raw events should read `owner`. No alias for the
  old key.
- **Breaking:** `Caller::System` is now `System { tenant_id }` (was a unit
  variant); crate consumers constructing or matching it must supply a tenant.
- **Breaking:** the `llmLoop` middleware (and `agent.llmLoop`) is renamed
  `llmToolLoop` to make the llm-and-tool loop it drives explicit.
- **Breaking (worker protocol):** a turn's tool and sub-agent results are
  delivered as a single batched `effects.complete` trigger, replacing the
  per-effect `tool.result`, `sub_agent.turn.complete`, and `sub_agent.error`
  triggers; `spawn.sub_agent` now carries `tool_call_id`. Handled transparently by
  SDK workers.

### Fixed

- OpenRouter responses no longer drop image outputs from the stream.
- The next LLM call in a turn waits for all of the turn's effects to finish, so
  turns with multiple sub-agents or a mix of tools and sub-agents no longer call
  the model with partial results. Sub-agents run concurrently.

## [0.1.15] - 2026-06-02

### Added

- Changelog with automated release notes.

### Changed

- Agent state is serialized as JSON automatically; chains no longer need an
  explicit `.use(agent.jsonState())`. Existing chains that still include it
  keep working, and the wire format is unchanged.

### Fixed

- Tool-result messages (`message.new` with `role: "tool"`) now carry the tool
  `name` instead of `null`, so consumers reading only the message stream (e.g.
  reconstructing a transcript from session events) no longer have to correlate
  `tool_call_id` back to the originating tool call to recover the name.

## [0.1.14] - 2026-06-02

### Changed

- SDK typing and developer-experience improvements.

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.1.19...HEAD
[0.1.19]: https://github.com/substructureai/substructure/compare/v0.1.18...v0.1.19
[0.1.18]: https://github.com/substructureai/substructure/compare/v0.1.17...v0.1.18
[0.1.17]: https://github.com/substructureai/substructure/compare/v0.1.16...v0.1.17
[0.1.16]: https://github.com/substructureai/substructure/compare/v0.1.15...v0.1.16
[0.1.15]: https://github.com/substructureai/substructure/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/substructureai/substructure/releases/tag/v0.1.14
