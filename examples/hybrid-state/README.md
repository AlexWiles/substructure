# hybrid-state

Most agent state rides the wire as base64 JSON (cheap, no infrastructure).
Some state lives in your database instead (better for large blobs, sensitive
data, or anything you want to query directly). This example shows how to do
both in the same agent.

The `todoSlice` middleware contributes a typed `todos` slice but loads and
saves it from the database, keyed by user id. Tools that opt into the slice
see `state.todos.items` as if it were ordinary in-memory data and don't
know about the persistence. Because the key is the user id rather than the
session id, the same todo list shows up across every conversation that user
has with the agent.

The "database" is a directory of JSON files keyed by user id, so you can
run the example with no infra. Swap `loadTodos` / `saveTodos` for a real
client to use Postgres, a Durable Object, S3, or anything else.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install
pnpm start
```

Inspect what ended up where:

```sh
cat todo-db/*.json   # the actual todo items, per user
```

The agent's wire-state blob (stored inside the engine's `agent.db`) only
contains `messages` and an empty `todos: { items: [] }`. The real items
never go through Substructure.
