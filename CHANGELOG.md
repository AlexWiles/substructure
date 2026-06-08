# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/sdk`, `@substructure.ai/runtime`,
`@substructure.ai/cli`) and the `substructure-core` crate release together at the
same version.

## [Unreleased]

### Added

- Native AG-UI protocol support: endpoints that stream an agent turn as AG-UI SSE
  events, with live token and reasoning streaming.
- Reasoning controls on `LlmRequest` (`reasoning`: `effort` or `max_tokens`,
  plus `exclude` and `enabled`), passed through to providers that support them.
- AI SDK adapter (`@substructure.ai/sdk/adapters/ai`): `ToolLoopAgent` drops an
  existing Vercel AI SDK agent (model, instructions, zod tools) into a handler
  chain. Substructure drives the loop and runs the tools as durable steps; each
  step is one `streamText` call and token deltas stream back.
- OpenAI adapter (`@substructure.ai/sdk/adapters/openai`): `OpenAIAgent` runs an
  `@openai/agents` `Agent` (or plain settings) on Substructure, executing each
  step via the OpenAI Responses API while Substructure drives the loop.
- The server and embedded runtime start without an LLM provider configured: the
  server-side LLM subsystem is skipped and worker-handled LLM calls still work.

### Changed

- **Breaking:** session events identify a session's end user as `owner` (was
  `identity`); consumers reading raw events should read `owner`. No alias for the
  old key.
- **Breaking:** `Caller::System` is now `System { tenant_id }` (was a unit
  variant); crate consumers constructing or matching it must supply a tenant.
- **Breaking (worker protocol):** a turn's tool and sub-agent results are
  delivered to the worker as a single batched `effects.complete` decision trigger
  once every effect in the turn reaches a terminal state, replacing the per-effect
  `tool.result`, `sub_agent.turn.complete`, and `sub_agent.error` triggers. The
  engine batches and orders the results and owns their tool-call mapping;
  `spawn.sub_agent` now carries `tool_call_id`. SDK workers handle this
  transparently and the `subAgents` middleware is now stateless.

### Fixed

- OpenRouter responses no longer drop image outputs from the stream.
- The next LLM call in a turn now waits for *all* of the turn's effects to reach a
  terminal state. Turns that issued multiple sub-agents, or mixed a tool call with
  a sub-agent, no longer call the model prematurely with only some results in.
  Sub-agents run concurrently (as independent sessions) with each other and with
  tool calls.

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

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.1.15...HEAD
[0.1.15]: https://github.com/substructureai/substructure/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/substructureai/substructure/releases/tag/v0.1.14
