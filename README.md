# substructure.ai

[![sdk](https://img.shields.io/npm/v/@substructure.ai/sdk?label=sdk)](https://www.npmjs.com/package/@substructure.ai/sdk)
[![cli](https://img.shields.io/npm/v/@substructure.ai/cli?label=cli)](https://www.npmjs.com/package/@substructure.ai/cli)

> Substructure is under active development. APIs, CLI commands, and the wire protocol may change between releases for versions 0.1.x

Substructure is an open-source engine for building durable, long-running AI agents using just an HTTP endpoint hosted on your infrastructure, in your code.

Substructure drives the agentic loop, handling retries, sub-agent supervision, llm calls, real-time event streaming and more. Tool execution and agent decisions live in your codebase on your infrastructure.

## Examples

Common patterns from [`examples/`](./examples). Each snippet shows the agent definition. The linked example has the full worker.

### Simple agent with history

A system prompt, history, and the LLM loop. History persists across turns.

```typescript
const sub = new Substructure();
const { agent } = sub;

const chatAgent = agent({ id: "chat" })
  .use(agent.messageHistory("You are a helpful assistant."))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));
```

### Tools

Tools are functions with a JSON-schema signature. A tool can opt into a typed state slice. Mutations persist across turns. See [`examples/node-embedded`](./examples/node-embedded).

```typescript
type Todo = { id: string; title: string; done: boolean };
const todos = agent.stateSlice<{ items: Todo[] }>({ items: [] });

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
    const item: Todo = { id: randomUUID().slice(0, 8), title, done: false };
    state.items.push(item);
    return JSON.stringify(item);
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
  .use(agent.messageHistory("You are a concise todo assistant. Use tools to manage the list."))
  .use(agent.tools([addTodo, listTodos]))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));
```

### State hydration

State rides the wire as JSON by default. To back a slice with your own database, write a middleware that loads on the way in and saves on the way out. Tools use `state.todos` like in-memory data, but it lives in your DB. Swap `loadTodos`/`saveTodos` for Postgres, Redis, S3, or a Durable Object. See [`examples/hybrid-state`](./examples/hybrid-state).

```typescript
const todoSlice = middleware<{ todos: TodoData }>({
  state: { todos: { items: [] } },
  handler: async (ctx, next) => {
    const userId = ctx.request.identity.id;
    ctx.state.todos = (await loadTodos(userId)) ?? { items: [] };

    const res = await next(ctx);

    await saveTodos(userId, ctx.state.todos);
    ctx.state.todos = { items: [] }; // keep the wire small
    return res;
  },
});

const addTodo = agent.tool({
  name: "add_todo",
  description: "Add a todo item",
  parameters: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
  state: todoSlice,
  execute: (args, state) => {
    const { title } = JSON.parse(args);
    const item = { id: randomUUID().slice(0, 8), title, done: false };
    state.todos.items.push(item);
    return JSON.stringify(item);
  },
});

const todoAgent = agent({ id: "todo" })
  .use(todoSlice)
  .use(agent.messageHistory("Concise todo assistant. Use tools to manage the list."))
  .use(agent.tools([addTodo]))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));
```

### Mixed state: user and session

Different data has different lifetimes. The wire state holds two ids. A hydration middleware loads each from its own store. History is keyed by session, so it tracks one conversation. Todos are keyed by user, so they follow a user across sessions. See [`examples/state-hydration`](./examples/state-hydration).

```typescript
type Refs = { historyId: string; todosId: string };
type Hydrated = Refs & { messages: Message[]; todos: Todo[] };

const hydrate: MiddlewareFn<Refs, Hydrated> = async (ctx, next) => {
  // First turn the refs are empty. Mint stable ids:
  // history per session, todos per user.
  const historyId = ctx.state.historyId || ctx.request.session_id;
  const todosId = ctx.state.todosId || ctx.request.identity.id;

  const hydrated: Hydrated = {
    historyId,
    todosId,
    messages: await load<Message[]>("history", historyId, []),
    todos: await load<Todo[]>("todos", todosId, []),
  };

  const res = await next({ ...ctx, state: hydrated });

  // Persist the heavy data, hand the wire back only the references.
  const final = res.state as Hydrated;
  await save("history", historyId, final.messages);
  await save("todos", todosId, final.todos);
  return { ...res, state: { historyId, todosId } satisfies Refs };
};

const todoAgent = agent({ id: "todo" })
  .use(agent.stateSlice<Refs>({ historyId: "", todosId: "" }))
  .use(hydrate)
  .use(agent.messageHistory("Concise todo assistant. Use the tools to manage the list."))
  .use(agent.tools([addTodo, listTodos]))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));
```

## How it works

- **Server:** The engine that drives the agent loop, written in Rust. It can be run locally on your machine, embedded in process, or as a cloud hosted version available at [https://app.substructure.ai](https://app.substructure.ai). The server drives the loop, handles durability, retries, llm calls, realtime streaming, subagent supervision and more.
- **Workers:** Your agent logic. Receives a decision trigger, returns actions. Runs in your codebase with your dependencies. Can be an HTTP endpoint for use with the cloud/local server, or a callback passed to embedded substructure.
- **Clients:**  Submit work and stream events back. We have support for both backend-to-backend as well as browser based clients.
- **CLI:** Substructure comes with a CLI to help you provision, observe, and debug from the terminal. You can also start a local server.
- **SDK:** We provide a TypeScript SDK for building agents and setting up your worker with a just a few lines of code. It also includes server-to-server and browser clients.

## Why Substructure
- **Write agent logic, not agent infrastructure.** The event log, retries, timeouts, streaming, etc. are Substructure's job.
- **Add agents to the codebase you already have.** Workers are plain HTTP handlers. You can drop them into your app, deploy them to your infrastructure.
- **Ship to serverless.** Stateless workers means they can be deployed to any serverless platform. There are no long running processes.

## Install

The CLI is available at:
```sh
npm i -g @substructure.ai/cli
```

The SDK is available at:
```sh
npm i @substructure.ai/sdk
```


## A Quick Example

This walks through running an agent against [Substructure Cloud](https://app.substructure.ai). Three steps: define a worker, point the cloud at it, submit a turn.

**1. Define an agent and serve it as a worker.** Workers are plain HTTP handlers; deploy this anywhere with a public URL (Cloudflare, Vercel, Fly, your own infra). See [`examples/`](./examples) for full deployments.

```typescript
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const { agent } = sub;

const getWeather = agent.tool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  execute: (args: string) => {
    const { city } = JSON.parse(args);
    return JSON.stringify({ city, temp_f: 62, condition: "sunny" });
  },
});

const weatherAgent = agent({ id: "weather-agent" })
  .use(agent.messageHistory("You are a helpful weather assistant."))
  .use(agent.tools([getWeather]))
  .use(agent.llmLoop({
    request: { model: "anthropic/claude-sonnet-4-5" },
  }));

const worker = sub.worker({ agents: [weatherAgent] });

export default {
  fetch: worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET }),
};
```

**2. Provision Substructure Cloud and point it at your deployed worker.**

```sh
substructure cloud login

substructure cloud link                                          # link this directory to an org & app

substructure cloud webhook set https://your-worker.example.com   # tell the substructure where to call

# Prints out the signing secret for the webhook. Copy into your worker's env as SIGNING_SECRET:
substructure cloud webhook secret

# Mint an API key for your client:
export SUBSTRUCTURE_API_KEY=$(substructure cloud keys create demo)
```

**3. Submit a turn from your client.**

```typescript
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const client = sub.backend.client({
  url: "https://api.substructure.ai",
  apiKey: process.env.SUBSTRUCTURE_API_KEY!,
});

const scope = await client.startTurn({
  agentId: "weather-agent",
  payload: {
    type: "message",
    message: { role: "user", content: "What's the weather in SF?" },
  },
  identity: { id: "user-1" },
});

const { data } = await client.turnResult(scope);
console.log(data);
```

## Docs

Full documentation in [`docs/`](./docs).

## Packages

| Package | Description |
| --- | --- |
| `@substructure.ai/sdk` | TypeScript SDK -- client, worker, and agent middleware |
| `@substructure.ai/runtime` | Embedded Rust runtime via NAPI bindings |
| `@substructure.ai/cli` | CLI for running the server |

## Development

```bash
pnpm install
pnpm dev       # Run the dashboard
pnpm build     # Build all packages
pnpm test      # Run tests
```
