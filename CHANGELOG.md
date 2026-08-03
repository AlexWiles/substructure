# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The `@substructure.ai/cli` package and the `substructure-core` crate release
together at the same version.

## [Unreleased]

### Fixed

- A turn that delegates to more than one sub-agent now waits for all of them to
  return before it continues.
- Failed decisions now retry or end the run with the error.
- Decisions and LLM calls have default timeouts and retries.
- An LLM call that times out on its last attempt now settles the turn instead of stalling it.
- A turn now completes when its finalizer decision settles without a `done` action.
- The push transport no longer cuts long streams.
- A delegation now carries its opening message, so a sub-agent turn no longer stalls when the message beats the child session.
- A failed sub-agent and an unconfigured session now report errors.
- A tool result that arrives after its deadline no longer reports an error.
- `subs login` and `subs logout` now read the environment file's
  `[deployment].url` like every other cloud command, so `subs login -c
  subs.prod.toml` authenticates against the deployment that file names instead
  of the hosted cloud.
- A CLI command that cannot reach the server now says so, instead of reporting a
  missing org or project, or a deployment that is too old.
- CLI errors now name the endpoint, and say when the response is not an API
  response.

### Changed

- An effect has a new `running` status. Work that started but did not return
  yet has this status.
- The event store keeps no store-wide event order. A processor holds a cursor
  for each session and reads the sessions independently, so a shard reads only
  its own sessions.
- An event has no `global_position` field. Use `session_id` and `seq`.
- The agent config has no `stream` field. LLM calls stream unless the `llm.call` action sets `stream: false`.
- Commands that open a browser now print the URL before they open it.
- A credential belongs to a connection id, not to a server URL. Declare
  `[mcp.sentry]` and `[mcp.sentry2]` at the same `url` to connect two accounts
  of one server. Do `subs mcp login <id>` again once for each connection: the
  credentials from before this change cannot move to the new key.
- A connection refuses a stored credential when its `url` points to a different
  server than the login did.
- `subs mcp login` connects a deployment's connection for the pinned project.
  Two projects that declare one connection id hold two credentials, and a login
  in one project does not change the other. `subs mcp list` and `subs mcp
  logout` show and remove that project's connections.
- `subs mcp login` has no `--no-grant`. A connection belongs to a project, so a
  connection that belongs to none is one that nothing can use.
- There is no `subs mcp logout`. Delete the `[mcp.<id>]` to disconnect a
  connection: an engine forgets its credential as it next starts, and a
  deployment as it next takes an apply.
- The CLI reads `substructure.toml` from the working directory only. It does not
  look in the parent directories.
- The engine database defaults to the name of the file that names it:
  `subs.staging.toml` uses `subs.staging.db`, and `substructure.toml` continues
  to use `substructure.db`. Two files in one directory are two projects with two
  databases. A relative `db` is now relative to the directory of its file, not
  to the working directory. If a file with a different name used
  `substructure.db`, write `db = "substructure.db"` in it to keep that data.

### Added

- A worker is now optional. Declare an agent in `substructure.toml`, and the engine decides its turns.
- `[agent.<id>]` sections declare each agent. The section mirrors the wire agent config, plus a `worker` URL and a `signing_secret_env`.
- An agent can give its whole config to its worker. Declare only a `worker` URL, and the worker declares the agent at `session.start`.
- `[llm.<id>]` sections declare each LLM. An agent names one, and the block's `type` sets where its calls run.
- The `session.start` decision proposes the agent config that the file declares.
- An `interrupt.resumed` decision proposes the next model call.
- `subs run` accepts the message as an argument.
- A client submit can queue a message while a turn is active.
- Add a Slack adapter and a channel abstraction.
- Add MCP support.
- substructure.toml as a config file.
- `subs init` writes a starter `substructure.toml`. It asks for a project name, a provider and model, an agent id, which MCP servers to connect, whether the agent answers in Slack, and where it runs, then prints the steps for that answer.

### Changed

- **`subs apply` replaces rather than merges.** The file is the whole
  declaration, so an agent, `[llm.*]` block, or Slack channel absent from it is
  one that was removed. Removal no longer needs an imperative command.
- **A project is born from a file.** `subs apps create` and `subs apps rename`
  are gone: apply owns a project's existence and its name. `subs link` still
  adopts an existing project, for a fresh clone.
- **Hosting is per agent, everywhere.** A deployment holds one worker per agent
  rather than one for the whole tenant, so a file whose agents point at
  different URLs is now ordinary rather than an error. The `subs webhook`
  commands and the deployment-wide worker are gone.
- **Signing secrets are per agent, and retrievable.** A deployment mints one on
  the first apply that gives an agent a `worker`, so nothing is printed once and
  lost; read it with `subs agents show <id>` and replace it with `subs agents
  rotate-secret <id>`. An agent the engine decides for has no secret, because
  nothing signs for it.
- **`[slack]` asks three questions instead of one.** `agent` meant both "who
  takes DMs" and "who takes a channel nobody named" — two decisions with very
  different blast radii behind one key, and it resolved differently locally than
  in the cloud. It is now `dm` and `mentions`, each defaulting to silence, so
  the bot answers only where it was told to. A file with the old key is a loud
  error naming both replacements.
- **Model calls run on your key.** An engine-run `[llm.*]` block needs a key
  uploaded with `subs llm set-key`; a call on a block without one fails saying
  so rather than falling back to a platform key.
- `substructure.toml` describes one system, in groups: settings moved into
  `[llm.<id>]`, `[agent.<id>]`, `[run]`, `[server]`, and `[deployment]`. A file
  carries two roles, either or both — **an engine you run** (`db`, `log`,
  `[run]`, `[server]`) and **a deployment you administer** (`[deployment]`) —
  while `name`, `[agent.<id>]`, `[llm.<id>]`, `[slack]`, and `[mcp.<id>]` are
  one declaration whichever role reads them. A self-hosted system is therefore
  one file rather than two that have to agree. There is no migration.

  ```toml
  # before
  worker_url = "http://localhost:4444"
  llm_provider = "anthropic"
  output = "pretty"
  port = 8080

  # after
  [llm.claude]
  type = "anthropic"
  [agent.assistant]
  llm = "claude"
  model = "claude-sonnet-4-5"
  worker = "http://localhost:4444"   # only if a worker decides for it
  [run]
  output = "pretty"
  [server]
  port = 8080
  [deployment]        # only if this file administers one
  org = "org_01hx…"
  ```

- `[server].dev` is `[server].auth`, and `subs serve --dev` is `--no-auth`
  (`--dev` still works as an alias): the file said "dev" for what is really
  "authenticate clients and workers", which is a decision about reachability,
  not about a stage. `auth` defaults to true.
- `subs mcp login` stores credentials in the environment's `db` instead of
  `credentials.toml`, so a login belongs to the environment that uses it. Run
  `subs mcp login <id>` once per environment; **gitignore `*.db*`**, which now
  holds credentials. `credentials.toml` keeps only `subs login` tokens, and the
  `credentials` config key and `--credentials` flag on `serve` are gone.
- `subs mcp add` is gone: declare `[mcp.<id>]` in the file. Ids and URLs are
  checked when the file is read.
- An `output` the file does not recognize is a parse error rather than a silent
  fall back to `ag-ui`.
- `subs link` writes to the file commands actually read (the discovered one, or
  `-c`) instead of always the working directory, and keeps a `url` it did not
  set.
- Rename the default database file to `substructure.db`.
- The `tool.call` action has no `handler` field; the engine finds where a call runs from its name.
- The agent config has an `llm` field, and no `handler` or `format` fields. The named `[llm.<id>]` block sets where a call runs and its wire shape.
- The `llm.call` action has an `llm` field and no `handler` field. Name a different block to move one call to a different LLM.
- Decisions route per agent. An agent with a `worker` URL gets a push; an agent without one runs on the engine; an agent that the file does not declare fails immediately.
- Remove the `[worker]` section. Set `worker` on each `[agent.<id>]` that a worker decides for.
- Remove the `subs webhook` commands and the local `PUT /apps/{app}/worker` route. `subs apply` pushes the worker URL that the file declares.
- Remove the `--worker-url`, `--signing-secret`, and `--llm-provider` options from `subs run` and `subs serve`. The file declares all three.
- `subs run` needs `--agent` or `[run].agent`, and checks the id before it makes a session.
- The engine signs a decision request only if the agent names a `signing_secret_env`.
- LLM calls and tool routing now use the agent config from the same decision.
- Effects and decisions queue in arrival order and dispatch when their prerequisites settle.
- Rename the decision events to the effect lifecycle names: `decision.queued`, `decision.dispatched`, `decision.completed`, `decision.errored`, `decision.dropped`.
- An LLM call waits for the connector fetches of its own decision and offers the fetched tools.
- One scheduler now decides all work: an effect past its deadline and a due retry settle at the next command, not only at a wake.
- A sub-agent effect is named by its child session. In `calls[]`, `id` is the child session and the new `tool_call_id` field gives the call the delegation answers.
- `calls[]` reports a connector fetch with the `connector_sync` kind.
- Each event of an effect names that effect with `id`, in place of `call_id`, `tool_call_id`, `session_id`, `connection_id` and `decision_id`.

### Removed

- Remove the N-API bindings and the `@substructure.ai/runtime` package. Use the CLI or the server.
- A client submit during an active turn now gives a conflict error and does not start a second turn.
- An AG-UI run that resumes an interrupt and sends messages now continues the resumed turn.
- An AG-UI run with a missing or partial `resume` for its open interrupts now ends with `RUN_ERROR`.
- A turn that starts while the previous turn finishes now completes the previous turn first.
- The session state has a new incompatible shape; delete existing database files.

## [0.2.3] - 2026-07-22

### Added

- Workers had no hook to run side effects (e.g. post a result) after a turn. Added a `turn.finished` engine→worker trigger carrying the turn's frozen output; the turn completes (`turn.completed` / `RUN_FINISHED`) only once the worker's finalizer settles, and a terminal finalizer failure completes the run as failed.

### Fixed

- A processor logged a store error at ERROR when its in-flight query was cancelled during runtime teardown, even before its cancel token was set. `spawn_blocking` join-cancellation now maps to `StoreError::Cancelled`, which the loop treats as a clean shutdown break.

### Changed

- The generic aggregate layer only served sessions. Session-specific execution and events now replace it.
- The event store round-tripped opaque JSON that every consumer re-parsed. The store now speaks typed session events end to end.
- Every event stored a full derived-state copy (message tree, prompts, decision triggers), growing storage quadratically. Events now carry a small bounded meta while the tree, anchored versions, and verbatim LLM prompts live in their own tables (new incompatible DB schema; delete existing `data.db` files).
- Decision triggers (full transcripts and LLM requests) were duplicated across queued, promoted, and retried decision events. The trigger now rides only the creating queued event; promotions and retries are id-only markers resolved from state at delivery.

## [0.2.2] - 2026-07-17

### Changed

- The engine sends an empty proposal instead of null.
- Fields the engine always emits were optional in the schema, forcing null
  checks in every generated client. They are now required on the wire.

### Added

- Added client facing `GET /api/client/sessions/{session_id}` returning status,
  open interrupts, and the full message tree and list as JSON.

### Fixed

- Regenerating recorded the new reply onto the old branch instead of forking.
  A decision whose view stops at an existing node now emits `head.moved`, rebasing
  the head so the reply records as a sibling branch.
- A client-tool round trip forked the tree when the resubmitted view raced the
  decision recording a worker tool's result, leaving a dangling duplicate branch.
  Tool echoes now also fold onto their recorded nodes at the decision-submit seam.
- The AG-UI connect endpoint required a `runId` although only `threadId` is read.
  Its body now needs `threadId` alone.

### Changed

- Better type names in protocol.rs, better generated type names.
- Interrupts are now anchored to the head that raised them and the session GET returns `interrupts[]` with head-resolved `status`.

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

[Unreleased]: https://github.com/substructureai/substructure/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/substructureai/substructure/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/substructureai/substructure/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/substructureai/substructure/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/substructureai/substructure/compare/v0.1.22...v0.2.0
