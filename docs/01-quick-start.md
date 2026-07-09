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
substructure login
substructure apps create example-agent
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
substructure link
```

## 4. Write the agent

`src/index.ts`:

```ts
import { agent, tool, toolLoop, worker } from "@substructure.ai/sdk";
import { DurableObject } from "cloudflare:workers";

type Todo = { title: string };
type State = { items: Todo[] };

export class AgentState extends DurableObject {
  private state: State = { items: [] };

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.blockConcurrencyWhile(async () => {
      this.state = (await ctx.storage.get<State>("state")) ?? { items: [] };
    });
  }

  async getState(): Promise<State> {
    return this.state;
  }

  async setState(state: State): Promise<void> {
    this.state = state;
    await this.ctx.storage.put("state", state);
  }
}

interface WorkerEnv extends Env {
  AGENT_STATE: DurableObjectNamespace<AgentState>;
  SIGNING_SECRET?: string;
}

function todoTools(namespace: DurableObjectNamespace<AgentState>) {
  const addTodo = tool({
    name: "add_todo",
    description: "Add a todo item",
    input: {
      type: "object",
      properties: { title: { type: "string" } },
      required: ["title"],
    },
    execute: async (args, request) => {
      const { title } = JSON.parse(args);
      const stub = namespace.getByName(request.session_id);
      const state = await stub.getState();
      await stub.setState({ items: [...state.items, { title }] });
      return JSON.stringify({ added: title });
    },
  });

  const listTodos = tool({
    name: "list_todos",
    description: "List all todos",
    execute: async (_args, request) => {
      const stub = namespace.getByName(request.session_id);
      return JSON.stringify((await stub.getState()).items);
    },
  });

  return [addTodo, listTodos];
}

export default {
  fetch(request: Request, env: WorkerEnv): Promise<Response> {
    const todoAgent = agent({
      name: "todo",
      decide: toolLoop({
        llm: { model: "anthropic/claude-sonnet-4-6" },
        instructions: "Concise todo assistant. Use tools to manage the list.",
        tools: todoTools(env.AGENT_STATE),
      }),
    });

    return worker([todoAgent]).fetch({ signingSecret: env.SIGNING_SECRET })(request);
  },
};
```

Each tool is a pure function; it reaches its store (the Durable Object, keyed by `request.session_id`) directly through the decision request. There is no SDK-held tool state.

## 5. Configure Wrangler

`wrangler.jsonc`:

```jsonc
{
  "name": "example-agent",
  "main": "src/index.ts",
  "compatibility_date": "2026-05-14",
  "compatibility_flags": ["nodejs_compat"],
  "durable_objects": {
    "bindings": [{ "name": "AGENT_STATE", "class_name": "AgentState" }]
  },
  "migrations": [{ "tag": "v1", "new_sqlite_classes": ["AgentState"] }]
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
substructure webhook set https://<your-worker>.workers.dev
substructure webhook secret | wrangler secret put SIGNING_SECRET
```

The secret goes straight from the CLI to Wrangler; it never lands in your shell history.

## 8. Add funds

```bash
substructure open
```

Add funds so your agent can execute LLM calls, then confirm the balance from the terminal:

```bash
substructure apps show
```

## 9. Run a turn

Mint an API key:

```bash
export SUBSTRUCTURE_API_KEY=$(substructure keys create quickstart)
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
    type: "client.message",
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
substructure sessions list
```

Copy a session id and stream its events:

```bash
substructure sessions events <SESSION_ID>
```

This replays the full history, then stays attached for live events (Ctrl-C to stop). You'll see the user message, each LLM response, the `add_todo` and `list_todos` tool calls with their results, and the final turn output.

Pass `--from <N>` to skip to a given event index. Or view the same session in the browser:

```bash
substructure open
```

## Next

- [Concepts](./02-concepts.md): sessions, turns, the decision loop.
- [SDK](./04-sdk.md): tools, state, custom decision functions, clients.
- [CLI](./03-cli.md): full command reference and local server.
- [Patterns](./06-patterns.md): approvals, plan mode, cross-session data.
