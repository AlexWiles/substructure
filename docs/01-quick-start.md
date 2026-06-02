---
title: Quick start
---

This walks you through building a basic agent and deploying it to a Cloudflare Worker, driven by Substructure Cloud.

The agent is a small todo assistant with two tools. State lives in a Durable Object, keyed per session.

## 1. Install the CLI

```bash
npm i -g @substructure.ai/cli
```

Verify it:

```bash
substructure --help
```

## 2. Log in and create an app

```bash
substructure cloud login
substructure cloud apps create example-agent
```

## 3. Scaffold a worker

Create a project and add the SDK:

```bash
mkdir example-agent && cd example-agent
npm init -y
npm i @substructure.ai/sdk
npm i -D wrangler typescript @types/node
```

Link the directory to your org and app (writes `substructure.toml`):

```bash
substructure cloud link
```

## 4. Write the agent

`src/index.ts`:

```ts
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const { agent } = sub;

const todos = agent.stateSlice<{ items: { title: string }[] }>({ items: [] });

const addTodo = agent.tool({
  name: "add_todo",
  description: "Add a todo item",
  parameters: {
    type: "object",
    properties: { title: { type: "string" } },
    required: ["title"],
  },
  state: todos,
  execute: (args, state) => {
    const { title } = JSON.parse(args);
    state.items.push({ title });
    return JSON.stringify({ added: title });
  },
});

const listTodos = agent.tool({
  name: "list_todos",
  description: "List all todos",
  parameters: { type: "object", properties: {} },
  state: todos,
  execute: (_args, state) => JSON.stringify(state.items),
});

const todoAgent = agent({ id: "todo" })
  .use(agent.messageHistory("Concise todo assistant. Use tools to manage the list."))
  .use(agent.tools([addTodo, listTodos]))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));

export default {
  fetch: sub.worker({ agents: [todoAgent] }).fetchHandler({
    signingSecret: process.env.SIGNING_SECRET,
  }),
};
```

## 5. Configure Wrangler

`wrangler.jsonc`:

```jsonc
{
  "name": "example-agent",
  "main": "src/index.ts",
  "compatibility_date": "2026-05-14",
  "compatibility_flags": ["nodejs_compat"]
}
```

## 6. Deploy

```bash
wrangler deploy
```

Copy the printed `*.workers.dev` URL.

## 7. Connect the worker

Point the app at the worker, then pipe the signing secret into the worker env:

```bash
substructure cloud webhook set https://<your-worker>.workers.dev
substructure cloud webhook secret | wrangler secret put SIGNING_SECRET
```

The secret goes straight from the CLI to Wrangler; it never lands in your shell history.

## 8. Add funds

```bash
substructure cloud open
```

Add funds so your agent can execute LLM calls, then confirm the balance from the terminal:

```bash
substructure cloud apps show
```

## 9. Run a turn

Mint an API key:

```bash
export SUBSTRUCTURE_API_KEY=$(substructure cloud keys create quickstart)
```

`client.ts`:

```ts
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const client = sub.backend.client({
  url: "https://api.substructure.ai",
  apiKey: process.env.SUBSTRUCTURE_API_KEY!,
});

const scope = await client.startTurn({
  agentId: "todo",
  payload: {
    type: "message",
    message: { role: "user", content: "Add 'buy groceries' and list my todos" },
  },
  identity: { id: "demo" },
});

const { data } = await client.turnResult(scope);
console.log(data);
```

Run it:

```bash
npx tsx client.ts
```

The agent calls `add_todo`, then `list_todos`, and returns the list.

## 10. Explore the session

Each turn runs inside a session. List recent ones:

```bash
substructure cloud sessions list
```

Copy a session id and stream its events:

```bash
substructure cloud sessions events <SESSION_ID>
```

This replays the full history, then stays attached for live events (Ctrl-C to stop). You'll see the user message, each LLM response, the `add_todo` and `list_todos` tool calls with their results, and the final turn output.

Pass `--from <N>` to skip to a given event index. Or view the same session in the browser:

```bash
substructure cloud open
```

## Next

- [Concepts](./02-concepts.md) — sessions, turns, the decision loop.
- [SDK](./04-sdk.md) — tools, state, middleware, clients.
- [CLI](./03-cli.md) — full command reference and local server.
- [Patterns](./06-patterns.md) — approvals, plan mode, cross-session data.
