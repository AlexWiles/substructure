# hono

Mounts the worker fetch handler inside a Hono app on Node. The agent
lives alongside whatever other HTTP routes you already serve, in the same
process, with the same middleware stack.

Use this shape when you have an existing Node service and want to add an
agent without standing up a separate worker process.

## Run

In one terminal, start a local Substructure server pointed at this worker:

```sh
export OPENROUTER_API_KEY=sk-or-...
substructure start --dev --port 9000 --worker-url http://localhost:3000/agent
```

In another terminal, start the Hono worker:

```sh
pnpm install
pnpm start
```

In a third terminal, submit a turn via the local backend:

```sh
pnpm client
```
