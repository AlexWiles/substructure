# substructure.ai

[![sdk](https://img.shields.io/npm/v/@substructure.ai/sdk?label=sdk)](https://www.npmjs.com/package/@substructure.ai/sdk)
[![cli](https://img.shields.io/npm/v/@substructure.ai/cli?label=cli)](https://www.npmjs.com/package/@substructure.ai/cli)

> Substructure is under active development. APIs, CLI commands, and the wire protocol may change between releases for versions 0.1.x

Substructure is an open-source engine for building durable, long-running AI agents using just an HTTP endpoint hosted on your infrastructure, in your code.

Substructure drives the agent loop, handling retries, sub-agent supervision, llm calls, real-time event streaming and more. Tool execution, agent decisions, llm calls (optionally) live in your codebase and execute on your infrastructure.

## How it works

- **Server:** The engine that drives the agent loop, written in Rust. It can be run locally on your machine, embedded in process, or as a cloud hosted version available at [https://app.substructure.ai](https://app.substructure.ai). The server drives the loop, handles durability, retries, llm calls (optionally), realtime streaming, subagent supervision and more.
- **Workers:** Your agent logic. Receives a decision trigger, returns actions. Runs in your codebase with your dependencies. Can be an HTTP endpoint for use with the cloud/local server, or a callback passed to embedded substructure.
- **Clients:**  Submit work and stream events back. We have support for both backend-to-backend as well as browser based clients.
- **CLI:** Substructure comes with a CLI to help you provision, observe, and debug from the terminal. You can also start a local server.
- **SDK:** We provide a TypeScript SDK for building agents and setting up your worker with a just a few lines of code. It also includes server-to-server and browser clients.

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
import { agent, server, tool, toolLoop, worker } from "@substructure.ai/sdk";

const getWeather = tool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  execute: (args) => {
    const { city } = JSON.parse(args);
    return JSON.stringify({ city, temp_f: 62, condition: "sunny" });
  },
});

const weatherAgent = agent({
  name: "weather-agent",
  decide: toolLoop({
    model: server("anthropic/claude-sonnet-4-6"),
    instructions: "You are a helpful weather assistant.",
    tools: [getWeather],
  }),
});

export default {
  fetch: worker([weatherAgent]).fetch({ signingSecret: process.env.SIGNING_SECRET }),
};
```

**2. Provision Substructure Cloud and point it at your deployed worker.**

```sh
substructure login

substructure link                                          # link this directory to an org & app

substructure webhook set https://your-worker.example.com   # tell the substructure where to call

# Prints out the signing secret for the webhook. Copy into your worker's env as SIGNING_SECRET:
substructure webhook secret

# Mint an API key for your client:
export SUBSTRUCTURE_API_KEY=$(substructure keys create demo)
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

## More Examples

Common patterns from [`examples/`](./examples). Each snippet shows the agent definition. The linked example has the full worker.

### Simple agent with history

A system prompt, history, and the LLM loop. History persists across turns.

```typescript
import { agent, server, toolLoop } from "@substructure.ai/sdk";

const chatAgent = agent({
  name: "chat",
  decide: toolLoop({
    model: server("anthropic/claude-sonnet-4-6"),
    instructions: "You are a helpful assistant.",
  }),
});
```

### Tools

Tools are pure functions with a JSON-schema signature. There is no SDK-held tool state — a tool reaches whatever store it needs. Here the list lives in a module-level object that persists for the life of the process. See [`examples/node-embedded`](./examples/node-embedded).

```typescript
import { agent, server, tool, toolLoop } from "@substructure.ai/sdk";
import { randomUUID } from "node:crypto";

type Todo = { id: string; title: string; done: boolean };
const todos: { items: Todo[] } = { items: [] };

const addTodo = tool({
  name: "add_todo",
  description: "Add a todo item",
  parameters: {
    type: "object",
    properties: { title: { type: "string" } },
    required: ["title"],
  },
  execute: (args) => {
    const { title } = JSON.parse(args);
    const item: Todo = { id: randomUUID().slice(0, 8), title, done: false };
    todos.items.push(item);
    return JSON.stringify(item);
  },
});

const listTodos = tool({
  name: "list_todos",
  description: "List all todos",
  parameters: { type: "object", properties: {} },
  execute: () => JSON.stringify(todos.items),
});

const todoAgent = agent({
  name: "todo",
  decide: toolLoop({
    model: server("anthropic/claude-sonnet-4-6"),
    instructions: "You are a concise todo assistant. Use tools to manage the list.",
    tools: [addTodo, listTodos],
  }),
});
```

### State in your own database

There is no SDK-held tool state. To persist data across sessions, a tool reaches your database directly through `ctx`, keyed by `ctx.request.identity.id`. The list lives in your store, follows the user across sessions, and never rides the wire. Swap `loadTodos`/`saveTodos` for Postgres, Redis, S3, or a Durable Object. See [`examples/hybrid-state`](./examples/hybrid-state).

```typescript
import { agent, server, tool, toolLoop } from "@substructure.ai/sdk";
import { randomUUID } from "node:crypto";

const addTodo = tool({
  name: "add_todo",
  description: "Add a todo item",
  parameters: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
  execute: async (args, ctx) => {
    const { title } = JSON.parse(args);
    const userId = ctx.request.identity.id;
    const data = await loadTodos(userId);
    const item = { id: randomUUID().slice(0, 8), title, done: false };
    data.items.push(item);
    await saveTodos(userId, data);
    return JSON.stringify(item);
  },
});

const todoAgent = agent({
  name: "todo",
  decide: toolLoop({
    model: server("anthropic/claude-sonnet-4-6"),
    instructions: "Concise todo assistant. Use tools to manage the list.",
    tools: [addTodo],
  }),
});
```

### State on the wire

Skip the database and let small state ride the decision envelope as `worker_state`, round-tripped every turn. There is no SDK-held tool state, so the agent is a custom `decide` that builds its tools per decision — each closing over the live state — hands them to `toolLoop`, and passes `state` into the loop. The loop runs the tools and echoes the state you gave it, so the mutations ride the wire with no manual plumbing. See [`examples/state-hydration`](./examples/state-hydration).

```typescript
import { agent, server, tool, toolLoop } from "@substructure.ai/sdk";
import { randomUUID } from "node:crypto";

type Todo = { id: string; title: string; done: boolean };
type State = { todos: Todo[] };

// Built fresh each decision so `execute` closes over the live list; `toolLoop`
// runs them, and the mutations land in `state.todos`.
function todoTools(state: State) {
  return [
    tool({
      name: "add_todo",
      description: "Add a todo item",
      parameters: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
      execute: (args) => {
        const todo: Todo = { id: randomUUID().slice(0, 8), title: JSON.parse(args).title, done: false };
        state.todos.push(todo);
        return JSON.stringify(todo);
      },
    }),
    tool({
      name: "list_todos",
      description: "List all todos",
      parameters: { type: "object", properties: {} },
      execute: () => JSON.stringify(state.todos),
    }),
  ];
}

const todoAgent = agent<State>({
  name: "todo",
  decide: async (req) => {
    const state: State = { todos: req.state?.todos ?? [] };
    const loop = toolLoop<State>({
      model: server("anthropic/claude-sonnet-4-6"),
      instructions: "Concise todo assistant. Use the tools to manage the list.",
      tools: todoTools(state),
    });
    return loop({ ...req, state }); // pass state in → the loop persists it back out
  },
});
```

### Bring your own agent framework

An existing agent built on another framework can run on Substructure through an adapter. The model, tools, and instructions stay as they are. Substructure handles durability, retries, and streaming around them.

- **[Vercel AI SDK](https://sdk.vercel.ai):** `aiSdkAgent` from `@substructure.ai/sdk/adapters/ai`. See [`examples/ai-sdk-example`](./examples/ai-sdk-example).
- **[OpenAI Agents](https://github.com/openai/openai-agents-js):** `openaiAgent` from `@substructure.ai/sdk/adapters/openai`. See [`examples/openai-example`](./examples/openai-example).
- **[Anthropic SDK](https://github.com/anthropics/anthropic-sdk-typescript):** `anthropicGenerate` from `@substructure.ai/sdk/adapters/anthropic` — a generator you pass as a `toolLoop`'s `model` (the core SDK has no agent type to wrap). See [`examples/anthropic-example`](./examples/anthropic-example).

The agent adapters return a `decide` you wrap with `agent({ name, decide })` and pass to `worker([...])`:

```typescript
import { agent } from "@substructure.ai/sdk";
import { aiSdkAgent } from "@substructure.ai/sdk/adapters/ai";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { tool } from "ai";
import { z } from "zod";

const openrouter = createOpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });

const chatAgent = agent({
  name: "ai-sdk-agent",
  decide: aiSdkAgent({
    model: openrouter("anthropic/claude-sonnet-4-6"),
    instructions: "You are a concise assistant.",
    tools: {
      getWeather: tool({
        description: "Get the current weather for a city.",
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => `It is 22°C and sunny in ${city}.`,
      }),
    },
  }),
});
```

The Anthropic adapter is a generator rather than an agent: you pass it as a `toolLoop`'s `model` and declare tools the usual way.

```typescript
import { agent, toolLoop } from "@substructure.ai/sdk";
import { anthropicGenerate } from "@substructure.ai/sdk/adapters/anthropic";

const chatAgent = agent({
  name: "anthropic-agent",
  decide: toolLoop({
    model: anthropicGenerate({ model: "claude-haiku-4-5", max_tokens: 1024 }),
    instructions: "You are a concise assistant.",
    tools: [getWeather],
  }),
});
```

## Docs

Full documentation in [`docs/`](./docs).
