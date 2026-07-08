# elixir-basic

The most basic chattable agent, in Elixir — no SDK, one Plug handler. The whole
worker reads the decision request and returns the next actions. It accepts every
decision the engine has a default for (`proposed`) first, then authors the one
that is genuinely its own: `client.messages` → the LLM request (the agent's
identity). Everything else — model replies and model failures — is the engine's
default loop, accepted by echoing `proposed` back.

`Mix.install` pulls Bandit, Plug, and Jason, so the worker runs as a single
script with no project scaffolding. For an agent that also runs its own tools,
see [`elixir-tools`](../elixir-tools).

The worker contract is one JSON request in, one JSON response out. See
[`docs/07-protocol.md`](../../docs/07-protocol.md) for the full protocol.

## Run

Three terminals.

**1. Start a local Substructure server** pointed at this worker:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
substructure serve --dev --provider anthropic --worker-url http://localhost:4444
```

**2. Start the worker** (listens on `:4444`):

```sh
elixir worker.exs
```

**3. Send a message.** Prints `{"session_id":"…","turn_id":"…"}`:

```sh
curl -s localhost:8080/api/machine/sessions/submit \
  -H 'authorization: Bearer dev' -H 'content-type: application/json' \
  -d '{
    "agent_id": "assistant",
    "identity": { "id": "demo" },
    "payload": { "type": "message", "message": { "role": "user", "content": "Hello!" } }
  }'
```

Then watch the turn play out (use the `session_id` from the response):

```sh
substructure sessions events <SESSION_ID> --url http://localhost:8080
```

You'll see the user message, the model's reply, and the turn's `done` output.
