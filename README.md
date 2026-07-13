# substructure.ai: Build production-ready AI agents in any language with no SDK

[![cli](https://img.shields.io/npm/v/@substructure.ai/cli?label=cli)](https://www.npmjs.com/package/@substructure.ai/cli)

> Substructure is under active development. APIs, CLI commands, and the wire protocol may change between releases for versions 0.1.x

Substructure is an open-source engine that drives the agent loop over HTTP. It keeps sessions durable and streams AG-UI events to your frontend. Your code stays on your infrastructure. It's just HTTP, so the engine is language agnostic. You don't even need an SDK.

## See it in action

Here is a basic chat bot built with javascript and hono.
```javascript
import { serve } from "@hono/node-server";
import { Hono } from "hono";

function decide({ trigger: t, proposed }) {
    // The engine proposes actions for a default agent tool loop
    if (proposed) return proposed;

    // The client sent an updated message list
    if (t.type === "client.messages") {
        return {
            // echo back the messages to update the session state
            messages: t.messages,
            // call the LLM with the messages
            actions: [{
                type: "llm.call", stream: true,
                request: {
                    model: "claude-haiku-4-5-20251001",
                    messages: t.messages
                }
            }],
        };
    }

    return { actions: [] };
}


const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));
serve({ fetch: app.fetch, port: 4444 });
```

Here is the same using Python and Fast API, plus using tools.

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/")
async def decide(request: Request):
    req = await request.json()

    # The engine proposes actions for a default agent tool loop
    if req["proposed"]:
        return req["proposed"]

    t = req["trigger"]

    # The client sent the conversation → record it, prompt the model.
    if t["type"] == "client.messages":
        return {
            "messages": t["messages"],
            "actions": [
                {
                    "type": "llm.call",
                    "stream": True,
                    "request": {
                        "model": "claude-haiku-4-5-20251001",
                        "messages": t["messages"],
                    },
                }
            ],
        }

    return {"actions": []}
```


Substructure drives the agent loop, handling retries, sub-agent supervision, llm calls, real-time event streaming and more. Tool execution, agent decisions, llm calls (optionally) live in your codebase and execute on your infrastructure.

Front ends get the same treatment: the engine speaks **AG-UI** natively, so a browser chat UI (TanStack AI, CopilotKit, assistant-ui) drops in and can speak directly to the engine.

## How it works

- **Server:** The engine that drives the agent loop, written in Rust. It can be run locally on your machine, embedded in process, or as a cloud hosted version available at [https://app.substructure.ai](https://app.substructure.ai). The server drives the loop, handles durability, retries, llm calls (optionally), realtime streaming, subagent supervision and more.
- **Workers:** Your agent logic. Receives a decision trigger, returns actions. Runs in your codebase with your dependencies. Can be an HTTP endpoint for use with the cloud/local server, or a callback passed to embedded substructure.
- **Clients:** Submit work and stream events back, backend-to-backend or straight from the browser. The engine also serves a native **AG-UI** endpoint, so any AG-UI chat frontend connects and streams directly. No Substructure SDK in the browser. See [`examples/typescript-sdk-ag-ui`](./examples/typescript-sdk-ag-ui).
- **CLI:** Substructure comes with a CLI to help you provision, observe, and debug from the terminal. You can also start a local server.
- **SDK (optional):** We provide a TypeScript SDK for building agents and setting up your worker with a just a few lines of code. It also includes server-to-server and browser clients. It's a convenience layer over the protocol, never a requirement.

## Install

The CLI is available at:
```sh
npm i -g @substructure.ai/cli
```

The SDK is available at:
```sh
npm i @substructure.ai/sdk
```


## A Quick Example. No SDK required

A worker is one stateless HTTP handler. The engine `POST`s a JSON **decision request** (`trigger`) and the conversation so far (`messages`) and your handler replies with a JSON **decision**. The decision is the updated conversation (`messages`) and what to do next (`actions`). Durability, retries, scheduling, and streaming are all handled by the engine.

### 1. The smallest agent

Two triggers make a chat agent: the client proposes the conversation → prompt the model; the model answers → end the turn.

```javascript
// worker.js: a complete Substructure worker. No SDK, no dependencies.
export default {
  async fetch(request) {
    return decide(await request.json());
  },
};

function decide(req) {
  switch (req.trigger.type) {
    // The client proposed the conversation (a new message, an edit, a full
    // view — all one shape) → write it in, prompt the model.
    case "client.messages": {
      const messages = req.trigger.messages;
      return reply(req, { messages, actions: [promptModel(messages)] });
    }

    // The model answered → write it in, end the turn.
    case "llm.finished": {
      if (!req.trigger.ok) {
        return reply(req, { actions: [{ type: "done", data: { error: req.trigger.error } }] });
      }
      const messages = [...req.messages, req.trigger.message];
      return reply(req, { messages, actions: [{ type: "done", data: req.trigger.message.content }] });
    }

    default:
      return reply(req, {});
  }
}

function promptModel(messages) {
  return {
    type: "llm.call",
    id: crypto.randomUUID(),
    handler: "server", // the engine calls the provider…
    stream: true,      // …and streams tokens to any listening client
    request: {
      model: "anthropic/claude-sonnet-4-6",
      messages: [{ role: "system", content: "You are a helpful assistant." }, ...messages],
    },
  };
}

function reply(req, decision) {
  return Response.json({
    messages: req.messages,
    actions: [],
    ...decision,
  });
}
```

Under fifty lines, and it is already durable: every message and model call is journaled by the engine, a failed LLM call is retried before you ever hear about it, and if your process dies mid-turn the decision is redelivered. The handler holds nothing in memory, and `stream: true` is all it takes for tokens to reach a browser, because streaming happens engine-side.

It's also just JSON over HTTP: the same handler is an afternoon's work in Python, Go, or Rust. The full wire spec is [`docs/07-protocol.md`](./docs/07-protocol.md).

### 2. Add tools

Tools are plain functions with a JSON-schema signature. Declare them:

```javascript
const tools = {
  get_weather: {
    description: "Get the current weather for a city.",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
    run: (args) => JSON.stringify({ city: JSON.parse(args).city, temp_f: 62, condition: "sunny" }),
  },
};
```

Advertise them to the model with one new field on the `llm.call` request:

```javascript
    request: {
      model: "anthropic/claude-sonnet-4-6",
      messages: [{ role: "system", content: "You are a helpful assistant." }, ...messages],
      tools: Object.entries(tools).map(([name, t]) => ({
        function: { name, description: t.description, parameters: t.parameters },
      })),
    },
```

And handle their lifecycle: `llm.finished` now schedules tool calls, and two new cases run them and fold the results back in.

```javascript
    // The model answered → run its tool calls, or end the turn if there are none.
    case "llm.finished": {
      if (!req.trigger.ok) {
        return reply(req, { actions: [{ type: "done", data: { error: req.trigger.error } }] });
      }
      const message = req.trigger.message;
      const messages = [...req.messages, message];
      const calls = message.tool_calls ?? [];
      if (calls.length === 0) {
        return reply(req, { messages, actions: [{ type: "done", data: message.content }] });
      }
      return reply(req, {
        messages,
        actions: calls.map((c) => ({
          type: "tool.call",
          id: c.id,
          name: c.function.name,
          arguments: c.function.arguments,
          handler: "worker",
        })),
      });
    }

    // The engine hands a scheduled call back → run the function, answer.
    case "tool.execute": {
      const { id, name, arguments: args } = req.trigger;
      return reply(req, { actions: [{ type: "tool.result", id, result: tools[name].run(args) }] });
    }

    // A tool finished → write the result in; prompt again once the step is done.
    case "tool.finished": {
      const { id, name, ok, result, error } = req.trigger;
      const messages = [
        ...req.messages,
        { role: "tool", content: ok ? result : error, tool_call_id: id, name },
      ];
      if (req.pending_calls > 0) return reply(req, { messages }); // siblings still running
      return reply(req, { messages, actions: [promptModel(messages)] });
    }
```

Notice the shape: you never run a tool inline. You *schedule* it (`tool.call`), the engine journals it and hands it back (`tool.execute`), and its result arrives as news (`tool.finished`). That round trip is what buys durability: failed tools retry with backoff, parallel calls fan out while `pending_calls` gates the next prompt, and a crash between any two steps loses nothing.

<details>
<summary>The complete worker: under a hundred lines, zero dependencies</summary>

```javascript
// worker.js: a complete Substructure worker. No SDK, no dependencies.

const tools = {
  get_weather: {
    description: "Get the current weather for a city.",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
    run: (args) => JSON.stringify({ city: JSON.parse(args).city, temp_f: 62, condition: "sunny" }),
  },
};

export default {
  async fetch(request) {
    return decide(await request.json());
  },
};

function decide(req) {
  switch (req.trigger.type) {
    // The client proposed the conversation → write it in, prompt the model.
    case "client.messages": {
      const messages = req.trigger.messages;
      return reply(req, { messages, actions: [promptModel(messages)] });
    }

    // The model answered → run its tool calls, or end the turn if there are none.
    case "llm.finished": {
      if (!req.trigger.ok) {
        return reply(req, { actions: [{ type: "done", data: { error: req.trigger.error } }] });
      }
      const message = req.trigger.message;
      const messages = [...req.messages, message];
      const calls = message.tool_calls ?? [];
      if (calls.length === 0) {
        return reply(req, { messages, actions: [{ type: "done", data: message.content }] });
      }
      return reply(req, {
        messages,
        actions: calls.map((c) => ({
          type: "tool.call",
          id: c.id,
          name: c.function.name,
          arguments: c.function.arguments,
          handler: "worker",
        })),
      });
    }

    // The engine hands a scheduled call back → run the function, answer.
    case "tool.execute": {
      const { id, name, arguments: args } = req.trigger;
      return reply(req, { actions: [{ type: "tool.result", id, result: tools[name].run(args) }] });
    }

    // A tool finished → write the result in; prompt again once the step is done.
    case "tool.finished": {
      const { id, name, ok, result, error } = req.trigger;
      const messages = [
        ...req.messages,
        { role: "tool", content: ok ? result : error, tool_call_id: id, name },
      ];
      if (req.pending_calls > 0) return reply(req, { messages }); // siblings still running
      return reply(req, { messages, actions: [promptModel(messages)] });
    }

    default:
      return reply(req, {});
  }
}

function promptModel(messages) {
  return {
    type: "llm.call",
    id: crypto.randomUUID(),
    handler: "server", // the engine calls the provider…
    stream: true,      // …and streams tokens to any listening client
    request: {
      model: "anthropic/claude-sonnet-4-6",
      messages: [{ role: "system", content: "You are a helpful assistant." }, ...messages],
      tools: Object.entries(tools).map(([name, t]) => ({
        function: { name, description: t.description, parameters: t.parameters },
      })),
    },
  };
}

function reply(req, decision) {
  return Response.json({
    messages: req.messages,
    actions: [],
    ...decision,
  });
}
```

</details>

### 3. Wire it up

The worker is a standard `fetch` handler, so deploy it anywhere that serves HTTP (Cloudflare Workers, Deno, Bun, or Node behind any adapter). Then point [Substructure Cloud](https://app.substructure.ai) at it:

```sh
subs login
subs link                                          # link this directory to an org & app
subs webhook set https://your-worker.example.com   # where the engine sends decision requests
export SUBSTRUCTURE_API_KEY=$(subs keys create demo)
```

Engine requests are HMAC-signed: `X-Substructure-Signature` carries HMAC-SHA256 of `"{timestamp}.{body}"` under the secret from `subs webhook secret`. Verify it in production (the SDK's `verifyWebhookSignature` does this if you'd rather not).

Submitting a turn is one POST, no client library:

```sh
curl -s https://api.substructure.ai/api/machine/sessions/submit \
  -H "Authorization: Bearer $SUBSTRUCTURE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "assistant",
    "identity": { "id": "user-1" },
    "payload": { "type": "message", "message": { "role": "user", "content": "What's the weather in SF?" } }
  }'
# → { "session_id": "…", "turn_id": "…" }
```

And the UI stream is one GET: server-sent events carrying the whole turn as it happens (`turn.started`, `message.new`, token-by-token `llm.token.delta`s, `tool.call.completed`, `turn.completed`).

```sh
curl -N "https://api.substructure.ai/api/machine/sessions/$SESSION_ID/events/stream?turn_id=$TURN_ID" \
  -H "Authorization: Bearer $SUBSTRUCTURE_API_KEY"
```

Prefer to run the engine yourself? `subs serve` starts a local one and the same flow applies (see [`docs/03-cli.md`](./docs/03-cli.md)).

### 4. Where the SDK fits

Everything above is the protocol; the SDK is a convenience layer over it. `toolLoop` is the loop from steps 1–2 (plus state, sub-agents, stop conditions, parallel calls) in a few lines, and there are typed backend and browser clients plus adapters for existing agent frameworks:

```typescript
import { agent, toolLoop, worker } from "@substructure.ai/sdk";

const assistant = agent({
  name: "assistant",
  decide: toolLoop({
    llm: { model: "anthropic/claude-sonnet-4-6", stream: true },
    instructions: "You are a helpful assistant.",
    tools: [getWeather], // tool({ name, description, parameters, execute })
  }),
});

export default {
  fetch: worker([assistant]).fetch({ signingSecret: process.env.SIGNING_SECRET }),
};
```

The examples below use it for browser streaming without curl, state patterns, and adapters for the Vercel AI SDK, OpenAI Agents, and the Anthropic SDK.

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

Tools are pure functions with a JSON-schema signature. There is no SDK-held tool state; a tool reaches whatever store it needs. Here the list lives in a module-level object that persists for the life of the process. See [`examples/typescript-sdk-node-embedded`](./examples/typescript-sdk-node-embedded).

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

There is no SDK-held tool state. To persist data across sessions, a tool reaches your database directly through `ctx`, keyed by `ctx.request.identity.id`. The list lives in your store, follows the user across sessions, and never rides the wire. Swap `loadTodos`/`saveTodos` for Postgres, Redis, S3, or a Durable Object. See [`examples/typescript-sdk-hybrid-state`](./examples/typescript-sdk-hybrid-state).

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

Skip the database and let small state ride the decision envelope as `worker_state`, round-tripped every turn. There is no SDK-held tool state, so the agent is a custom `decide` that builds its tools per decision (each closing over the live state), hands them to `toolLoop`, and passes `state` into the loop. The loop runs the tools and echoes the state you gave it, so the mutations ride the wire with no manual plumbing. See [`examples/typescript-sdk-state-hydration`](./examples/typescript-sdk-state-hydration).

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

- **[Vercel AI SDK](https://sdk.vercel.ai):** `aiSdkAgent` from `@substructure.ai/sdk/adapters/ai`. See [`examples/typescript-sdk-ai-sdk`](./examples/typescript-sdk-ai-sdk).
- **[OpenAI Agents](https://github.com/openai/openai-agents-js):** `openaiAgent` from `@substructure.ai/sdk/adapters/openai`. See [`examples/typescript-sdk-openai`](./examples/typescript-sdk-openai).
- **[Anthropic SDK](https://github.com/anthropics/anthropic-sdk-typescript):** `anthropicGenerate` from `@substructure.ai/sdk/adapters/anthropic`, a generator you pass as a `toolLoop`'s `model` (the core SDK has no agent type to wrap). See [`examples/typescript-sdk-anthropic`](./examples/typescript-sdk-anthropic).

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
