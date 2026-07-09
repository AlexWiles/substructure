# go-tools

A chattable agent with its own tool, in Go — no SDK, no dependencies, just
`net/http`. The whole worker reads the decision request and returns the next
actions. It accepts every decision the engine has a default for (`proposed`)
first, then authors the two that are genuinely its own: `client.messages` → the
LLM request (the agent's identity), and `tool.execute` → run the tool. Everything
else — tool results, model replies, model failures, even broken or hallucinated
tool calls — is the engine's default loop, accepted by echoing `proposed` back.

The single tool here, `get_current_time`, returns the current UTC time. For the
same agent without tools, see [`go-basic`](../go-basic).

The worker contract is one JSON request in, one JSON response out. See
[`docs/07-protocol.md`](../../docs/07-protocol.md) for the full protocol.

## Run

Three terminals.

**1. Start a local Substructure server** pointed at this worker:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs serve --dev --provider anthropic --worker-url http://localhost:4444
```

**2. Start the worker** (listens on `:4444`):

```sh
go run .
```

**3. Send a message.** Prints `{"session_id":"…","turn_id":"…"}`:

```sh
curl -s localhost:8080/api/machine/sessions/submit \
  -H 'authorization: Bearer dev' -H 'content-type: application/json' \
  -d '{
    "agent_id": "assistant",
    "identity": { "id": "demo" },
    "payload": { "type": "client.message", "message": { "role": "user", "content": "What time is it?" } }
  }'
```

Then watch the turn play out (use the `session_id` from the response):

```sh
subs sessions events <SESSION_ID> --url http://localhost:8080
```

You'll see the user message, the model's tool call, the tool result, the model's
reply, and the turn's `done` output.
