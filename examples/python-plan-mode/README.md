# python-plan-mode

A modal agent — plan, then execute — in Python, no SDK, one FastAPI POST handler.
The worker reads a `mode` field from state and picks the model, prompt, and tool
for that mode: a **planner** builds a checklist with the user (`add_step`), then an
**executor** walks it end-to-end (`complete_step`). One tool per mode is enough to
show the shape; the modality is the point. Everything mechanical — model replies,
tool results, broken tool calls — is the engine's default loop, accepted by
echoing `proposed` back; the worker only authors the decisions that are its own.

The client flips modes with a `client.action` (`set_mode`), not a chat message,
so the switch never appears in the transcript. Entering execution forks a fresh
branch seeded with only the rendered plan (a plan-only message with no id forks
at the root), so the executor starts clean; submitting state on that forking
decision carries the plan into the new branch.

For a single-mode agent with a tool, see [`python-tools`](../python-tools). The
[`typescript-sdk-plan-mode`](../typescript-sdk-plan-mode) example is the same
pattern on the SDK's `toolLoop`.

The worker contract is one JSON request in, one JSON response out. See
[`docs/07-protocol.md`](../../docs/07-protocol.md) for the full protocol and
[`docs/06-patterns.md`](../../docs/06-patterns.md) for the modal-agent pattern.

## Run

Three terminals.

**1. Start a local Substructure server** pointed at this worker:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs serve --dev --provider anthropic --worker-url http://localhost:4444
```

**2. Start the worker** (listens on `:4444`):

```sh
pip install fastapi uvicorn
uvicorn main:app --port 4444
```

**3. Plan, then execute.** Reuse one `session_id` across calls so the plan
persists. First, build the plan over one or more messages:

```sh
curl -s localhost:8080/api/machine/sessions/submit \
  -H 'authorization: Bearer dev' -H 'content-type: application/json' \
  -d '{
    "agent_id": "assistant",
    "identity": { "id": "demo" },
    "payload": { "type": "client.message", "message": { "role": "user", "content": "Plan a weekend trip to the coast." } }
  }'
```

Then flip to execution with a `set_mode` action (use the `session_id` from the
response):

```sh
curl -s localhost:8080/api/machine/sessions/submit \
  -H 'authorization: Bearer dev' -H 'content-type: application/json' \
  -d '{
    "agent_id": "assistant",
    "identity": { "id": "demo" },
    "session_id": "<SESSION_ID>",
    "payload": { "type": "client.action", "name": "set_mode", "args": { "mode": "executing" } }
  }'
```

Watch either turn play out:

```sh
subs sessions events <SESSION_ID> --url http://localhost:8080
```

In planning you'll see `add_step` build the checklist; after `set_mode` the
executor picks up a fresh branch holding only the plan and marks each step done.
Send `{ "mode": "planning" }` to switch back.
