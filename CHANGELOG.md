# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
All packages (`@substructure.ai/sdk`, `@substructure.ai/runtime`,
`@substructure.ai/cli`) and the `substructure-core` crate release together at the
same version.

## [Unreleased]

### Added

- Native Anthropic and OpenAI LLM providers.
- `run` CLI command sends input to the engine.
- Workers authored every decision. Requests now carry `proposed`, the engine's
  default continuation, so workers author only their own.
- Tool I/O went unvalidated. Tools declare `input`/`output` JSON Schemas the
  engine checks both ways.
- The worker wire rejected sparse frames. `handler`/`retryable`/`actions` now
  default and tool payloads accept any JSON.
- AG-UI dropped sub-agent delegations and misordered their marker. Each now
  renders as an in-order tool call resolving to the child's answer.
- A per-stream replay cursor was filtered against the global sequence column, so
  `subs run` settles and SSE resume broke on any db with prior events. Split the
  cursor into typed `GlobalPosition`/`StreamVersion`, each filtering its own column.
- The wire protocol had no machine-readable spec. `schemas/protocol.schema.json`
  (draft 2020-12, via schemars) is generated from the protocol types; a test
  regenerates it and fails on drift.
- The engine's `proposed` continuation was a separate `DecisionProposal` type, so
  echoing it back as a `DecisionResponse` was a type mismatch. Merged the two —
  `proposed` is now the `DecisionResponse` the worker echoes or replaces.
- No typed TypeScript tools example. Added `node-hono-typescript-chat-with-tools`:
  a Hono worker that declares tools and answers `tool.execute`, typed from
  schema-generated `protocol.ts`.
- No typed Python example. Added `python-fast-api-pydantic-chat-with-tools`: a
  FastAPI tools worker whose Pydantic models are generated from the schema, so
  requests are validated at the boundary.
- No assistant-ui web example. Added `node-hono-assistant-ui`: a Hono worker plus
  an assistant-ui `<Thread>` on the AG-UI runtime, streaming from the engine's
  native AG-UI endpoint.
- No CopilotKit web example. Added `node-hono-copilotkit`: the same worker with
  CopilotKit's `<CopilotChat>` wired direct-to-agent, forwarding a `useFrontendTool`
  up to the engine's native AG-UI endpoint.
- The worker contract had no OpenAPI form for server-stub generators, validating
  proxies, and rendered docs. `schemas/worker.openapi.json` (3.1) is generated
  beside the JSON Schema: one `POST /`, components pruned to the worker surface.
- AG-UI runs declared frontend tools/context/state the worker never saw. The
  engine forwards them on `client.messages` as `trigger.client` and layers the
  client tools onto the proposal by default, so browser-declared client tools work
  without being defined in the worker (which may still override to whitelist).

### Changed

- The protocol schema broke code generators: no roots, anonymous inlined defs,
  `true` schemas, a flattened-union `Effect`, two `handler` enums. Roots for
  every wire surface, named `$defs`, `{}` for any-JSON, string `cost`, a flat
  `Effect`, and one seam-validated `handler` enum — wire format unchanged.
- Webhook signing used two headers and a timestamp window. Simplified to a
  single `X-Substructure-Signature: sha256=<hex>` HMAC over the body,
  GitHub-style.
- **Breaking:** CLI binary renamed `substructure` → `subs`.
- **Breaking:** client payloads namespaced (`message` → `client.message`, etc.).
- **Breaking:** decision wire unifies on `messages` (was `transcript`).
- **Breaking:** `LlmTool` is flat `{name, description, input?, output?}`; old
  events won't deserialize, so wipe dev databases.
- Worker ids are optional (engine-assigned); the sync decision response drops
  `session_id`/`decision_id`.
- **Breaking (Rust):** public wire types were scattered across runtime modules.
  All now live logic-free in `crate::protocol`; seams and impls stay put.
- **Breaking (Rust):** decision-flow types were `Wire`-prefixed, blurring them
  with shared wire types. Renamed to a `Decision*` prefix (`DecisionRequest`,
  `DecisionResponse`, `DecisionTrigger`, `DecisionAction`, `DecisionProposal`);
  the internal `decision::DecisionTrigger` became `decision::Trigger`.
- **Breaking (Rust):** the remaining `Wire`-prefixed protocol types were unclear.
  Client-input types took a `Client*` prefix (`ClientInput`, `ClientPayload`,
  `ClientMessage`, `ClientMessages`, `ClientAction`) and `WireMessage` became
  `DraftMessage` (the id-optional, not-yet-recorded form).

### Fixed

- `--output pretty` hid the tool-call id, so a client tool couldn't be settled.
  A client-tool yield now prints each pending call's id and a ready-to-edit
  `tool.result` settle input.
- Inbound AG-UI messages dropped assistant `toolCalls`, orphaning the tool
  result on the next turn and 400-ing the provider (which interrupted the
  session). `AgUiMessage` now carries `toolCalls` and `to_messages` maps them.
- A no-arg tool call emitted no AG-UI `TOOL_CALL_ARGS`, so a client tool's
  browser executor never fired and the chat hung. Empty arguments now emit `{}`.

## [0.1.22] - 2026-07-07

### Changed

- **Breaking:** `client.message` removed. All client input delivers as
  `client.transcript`: `messages` is the full proposed conversation, and a new
  `new_from` field is the index where the unrecorded suffix starts. Bare-message
  submissions are expanded onto the active branch at delivery; the client API
  (`message`/`messages` payloads) is unchanged.
- The engine mints an `llm.call`/`tool.call` id when the worker omits one (as it
  already did for interrupts), so SDK-less workers need not generate ids. The
  `id` field on those two actions is now optional on the wire — omit it and the
  engine assigns one (previously a missing `id` failed frame parsing).
- **Breaking (Rust):** `WorkerDecisionRequest.tenant_id` removed; use the
  `tenant_id()` accessor. Fixes decisions dropped after a durable-queue round-trip.

### Fixed

- `pending_calls` now counts sibling results still awaiting recording.
- A wake no longer promotes a worker decision while one is live, which could
  fork the transcript across parallel tool results.

## [0.1.21] - 2026-07-06

### Added

- **Branch-scoped worker state.** State writes are versions anchored to the
  message tree, and a decision carries the state resolved from its branch — a
  fork sees state as of the fork point. Unchanged submissions write nothing.
- **Forking cancels the branch's outstanding work.** Calls record the tree
  node they were requested at; a fork voids in-flight calls anchored below
  the fork point (new `call.voided` event), drops their undelivered
  decisions (new `decision_request.dropped` event), and rejects late settles.
  Workers need no staleness checks.
- **Voided work is explicit and cascades.** Interrupt and cancel emit the same
  `call.voided` events instead of voiding silently, and a voided sub-agent
  delegation cancels its child session, recursively through its sub-tree.
  Cancelling an already-done session is a no-op.

### Removed

- The `stall` decision trigger; a decision with no actions parks the session.

### Changed

- **Breaking (wire): flat trigger and action names.** The `kind` sub-dispatch is
  gone — every wire name is `subject.verb` and `type` is the only discriminator.
  Triggers: `effect.execute` becomes `tool.execute`/`llm.execute`,
  `effect.settled` becomes `tool.finished`/`llm.finished`/`sub_agent.finished`
  (finished means final: the payload when `ok`, `error` when not). Actions:
  `call.llm`→`llm.call`, `call.tool`→`tool.call`, `effect.result`/`effect.error`
  → `tool.result`/`llm.result`/`tool.error`/`llm.error`,
  `spawn.sub_agent`→`sub_agent.spawn`, `send.message`→`message.send`. `kind`
  survives only where a heterogeneous list needs a tag: the request's in-flight
  list.
- **Breaking (wire): "effect" becomes "call" on the wire.** The decision
  request's `effects`/`pending_effects` are renamed `calls`/`pending_calls`, the
  settle endpoints move from `/effects/settle` to `/calls/settle`, and the
  `effect.voided` event is renamed `call.voided` — a voided sub-agent is now
  named by the `tool_call_id` it answers (like its finish), with the child in
  `session_id`.
- **Breaking (wire): a slimmer decision request.** Top-level `tenant_id` is gone
  (it lives in `identity`), and `span` left the body for a W3C `traceparent`
  header.
- **Breaking (wire):** the `user.message`/`user.transcript` triggers are renamed
  `client.message`/`client.transcript`, matching `client.action`.
- **Breaking (wire):** a sub-agent finish's `id` is now the originating
  `tool_call_id` (the child session moves to `session_id`), so every finish
  folds under `trigger.id`.
- **Breaking (SDK):** types follow the wire: the in-flight `Effect` union is now
  `InFlightCall` (`calls`/`pending_calls` on the request), and `LlmSettled` is
  `LlmOutcome`.
- **Breaking (wire):** the decision request's `worker_state` is renamed `state`;
  the `session.created` event's `owner` is renamed `identity`.
- **Breaking (wire):** `state` on the worker submit is optional — omitted or
  `null` keeps the current state, a non-null value writes it (`{}` clears).
- **Breaking (wire):** `worker.decision.completed` no longer carries `state`;
  `worker.state.updated` gains `anchor`; the admin read model serializes
  `state_versions` instead of `worker_state`.
- **Breaking (SDK):** `toolLoop` returns no `state` opinion; a wrapper persists
  its own state by returning it alongside the loop's decision.
- **Breaking (SDK):** `req.state` is delivered untouched (`S | null`); absent
  state is `null`, not `{}`.
- `attempt` is optional on the settle actions (`tool.result`/`llm.result`/
  `tool.error`/`llm.error`) and the settle endpoints; supply it only to fence
  out a stale executor.
- Worker state rides the wire as raw JSON (was a base64-encoded string).
- The engine stamps the assistant message id when an `llm_call` settles; the
  SDK `stamp` helper is removed.
- SDK message types split read from write: `Message` has a required `id`,
  `MessageInput` (what you submit) an optional one.

### Fixed

- A result settling after the client edited the conversation no longer lands on
  the edited branch.

## [0.1.20] - 2026-07-03

### Added

- **Message-tree conversation history.** History now lives in a tree the engine
  owns and sends on each worker decision, not in worker state. The worker returns
  the transcript as it should look and the engine reconciles it, forking a branch
  where it diverges. This enables edits, regenerations, plan/execute, and prompt
  compaction.
- The worker decision includes a count of effects still running, so a worker knows
  when a step is done without scanning the list.
- A language-neutral protocol reference (`docs/07-protocol.md`), so workers can be
  written in any language.
- New `assistant-ui-cloudflare-starter` example: an assistant-ui chat on a
  Cloudflare Worker, streaming from AG-UI.

### Changed

- **Breaking (SDK): the agent API is rebuilt.** An agent is now a single decision
  function (the built-in tool/sub-agent loop, or your own returning plain action
  objects), replacing the middleware/builder API. Models are configured inline for
  the server to run, or supplied as an adapter your worker runs.
- **Breaking (wire): one unified effect protocol.** Tool, model, and sub-agent
  calls share a single request/settle vocabulary instead of a message type per
  kind, and sub-agent results are reported as themselves, not as tool results. The
  settle endpoint and its SDK/runtime methods were renamed to match.
- **Breaking (wire/SDK): parallel model calls.** A worker can issue several model
  calls from one decision that settle independently; the one-at-a-time restriction
  is gone. The engine still paces server-run calls; run them on the worker for real
  overlap.
- **Breaking (SDK): tools.** A tool's execute receives the decision it runs under,
  not a separate context; async tools are flagged with an option, not a sentinel;
  tool state lives in your own store; and client tools may omit execute.
- **Breaking (SDK): streaming.** A worker-run model streams tokens through one
  delta shape; the intermediate stream-part type and its wrapper are gone.
- **Breaking (wire):** the decision request's end-user field is renamed from
  `owner` to `identity`, matching the SDK and docs.
- Workers no longer send a span with submissions; the engine owns tracing.
- Model-call actions default their retry and streaming options, and must state who
  runs the call.
- The AG-UI `/run` endpoint forwards the client's full transcript, which the engine
  reconciles into the tree.

### Removed

- **Breaking (SDK):** the middleware system and the `Substructure.agent` factory;
  the AI and OpenAI adapter classes become function equivalents.
- The `substructure new` command and the `templates/` directory (starters moved
  into `examples/`).
- The `examples/` pnpm workspace; each example is now a standalone npm project.

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

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.1.22...HEAD
[0.1.22]: https://github.com/substructureai/substructure/compare/v0.1.21...v0.1.22
[0.1.21]: https://github.com/substructureai/substructure/compare/v0.1.20...v0.1.21
[0.1.20]: https://github.com/substructureai/substructure/compare/v0.1.19...v0.1.20
[0.1.19]: https://github.com/substructureai/substructure/compare/v0.1.18...v0.1.19
[0.1.18]: https://github.com/substructureai/substructure/compare/v0.1.17...v0.1.18
[0.1.17]: https://github.com/substructureai/substructure/compare/v0.1.16...v0.1.17
[0.1.16]: https://github.com/substructureai/substructure/compare/v0.1.15...v0.1.16
[0.1.15]: https://github.com/substructureai/substructure/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/substructureai/substructure/releases/tag/v0.1.14
