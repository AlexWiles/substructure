---
title: SDK
---

The `@substructure.ai/sdk` package is how you build agents and connect to Substructure from TypeScript. It gives you three things in one package:

- An **agent** API for defining what your agent does (tools, state, LLM calls).
- A **worker** wrapper that exposes those agents as an HTTP endpoint Substructure can call into.
- **Clients** for submitting turns from your backend or browser.

## Install

```sh
npm i @substructure.ai/sdk
```

## The `Substructure` class

Everything starts from a single instance:

```ts
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
```

From `sub` you get:

- `sub.agent` — factory for defining agents, tools, and middleware.
- `sub.worker(...)` — wrap agents into an HTTP handler.
- `sub.backend.client(...)` — server-to-server client (uses an API key).
- `sub.frontend.client(...)` — browser client (uses a short-lived token).
- `sub.embedded(...)` — run the engine in-process with a SQLite event log.

## Defining tools

Tools are how the agent acts on the world. Define one with `agent.tool`:

```ts
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
    return { city, temp_f: 62, condition: "sunny" };
  },
});
```

A few notes:

- `parameters` is a plain JSON Schema object. The LLM uses it to format calls.
- `execute` receives the raw stringified JSON args. Parse it yourself.
- Return any JSON-serializable value. It's stringified and fed back to the LLM as the tool result.
- Tools can be async. Substructure will wait for the promise.

### Typed state per tool

If a tool needs to read or mutate state, declare a state slice and pass it in:

```ts
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
    const item = { id: crypto.randomUUID(), title, done: false };
    state.items.push(item);
    return item;
  },
});
```

The `state` you mutate inside `execute` is durably persisted by Substructure across turns.

## Building an agent

An agent is a chain of middleware. You start with `agent({ id })` and stack behavior with `.use(...)`:

```ts
const weatherAgent = agent({ id: "weather-agent" })
  .use(agent.jsonState())
  .use(agent.systemMessage("You are a helpful weather assistant."))
  .use(agent.messageHistory())
  .use(agent.tools([getWeather]))
  .use(agent.llmLoop({
    request: { model: "anthropic/claude-sonnet-4-5" },
  }));
```

The built-in middleware:

| Middleware | What it does |
| --- | --- |
| `agent.jsonState()` | Decodes incoming worker state and encodes the result. Almost always the first middleware. |
| `agent.systemMessage(str \| fn)` | Prepends a system message to every LLM call. Pass a function `(state, req) => string` to compute it dynamically. |
| `agent.messageHistory()` | Tracks the full message history across turns and injects it into LLM calls. |
| `agent.messageHistoryCurrentTurn()` | Same, but scoped to a single turn. |
| `agent.tools([...])` | Registers tools, dispatches tool calls from the LLM, and feeds results back. |
| `agent.llmLoop({ request })` | Drives the core loop: on a user message or tool result, call the LLM; on an LLM response with no tool calls, finish the turn. |
| `agent.subAgents({ agents })` | Lets the agent delegate to child agents as if they were tools. |
| `agent.logging()` | Logs each decision lifecycle to stdout. Handy in development. |

The order matters. State middleware first, then context (system message, history), then tools, then `llmLoop` at the end to drive the loop.

### Writing your own middleware

If the built-ins don't cover what you need, write your own. A middleware is just a function:

```ts
import type { MiddlewareFn } from "@substructure.ai/sdk";

const timing: MiddlewareFn = async (req, next) => {
  const start = Date.now();
  const res = await next(req);
  console.log(`decision took ${Date.now() - start}ms`);
  return res;
};

const myAgent = agent({ id: "..." })
  .use(timing)
  .use(/* ... */);
```

The middleware receives:

- `req` — the incoming decision request. The interesting fields are `req.state` (the agent state so far) and `req.wire` (the raw envelope, including `wire.session_id`, `wire.turn_id`, and the decision trigger).
- `next(req)` — runs the rest of the chain. Returns a response containing `actions` (what the engine will do next), `state` (the new state to persist), and optionally `workerState` (the raw, serialized form Substructure sends back next time).

You can mutate `req` before calling `next`, inspect or rewrite `res.actions` after, or short-circuit entirely.

### Example: keep conversation state in your own database

The default `agent.jsonState()` round-trips the agent's state through Substructure as a base64-encoded blob on every request. That's fine for small state, but for large message histories or sensitive data you may want state to live entirely in your own database. Write a middleware that loads state on the way in and saves it on the way out, then return a tiny reference instead of the blob:

Declare the shape of the state your middleware loads and the rest of the chain (and your tools) will see it as that type:

```ts
import { middleware } from "@substructure.ai/sdk";
import type { Message } from "@substructure.ai/sdk";

type SupportState = {
  messages: Message[];
  ticketId: string | null;
};

const dbState = (db: MyDatabase) =>
  middleware<SupportState>({
    state: { messages: [], ticketId: null },
    handler: async (req, next) => {
      const userId = req.wire.identity.id;
      const sessionId = req.wire.session_id;

      const loaded = await db.loadAgentState(userId, sessionId);
      if (loaded) req.state = loaded;

      const res = await next(req);

      await db.saveAgentState(userId, sessionId, res.state);

      return res;
    },
  });

const myAgent = agent({ id: "support" })
  .use(dbState(db))
  .use(agent.systemMessage("You are a support agent."))
  .use(agent.messageHistory())
  .use(agent.tools([/* ... */]))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-5" } }));
```

The `state` field on `middleware` does two things: it gives you the initial value used on the first turn, and it locks in the type so `req.state` is typed inside `handler` and any tool that takes `state: this slice` gets the same type. Downstream middleware like `messageHistory` will populate `req.state.messages` for you; `ticketId` is a slot you can read and write from your own tools.

`req.wire.identity.id` is the user id the client passed when calling `startTurn`, and `req.wire.session_id` is the conversation. Keying on both means a single user can have multiple parallel conversations and you can scope, list, or delete state per user without ever touching Substructure.

Because this middleware loads and saves state directly to your database, you don't need `agent.jsonState()` in the chain at all: there's no `workerState` to round-trip. Substructure will pass an empty wire state on the next turn and your middleware will load the real state from the DB.

The same pattern works for anything that needs to bridge the agent to your infrastructure: pulling user profile data into state, writing audit logs alongside the response, gating tool calls by feature flag, or short-circuiting a turn when a per-user quota is exceeded.

### Example: hybrid wire and database state

Sometimes you want most state on the wire (cheap, no infrastructure) and just one slice in your database (because it's large, sensitive, or you want to share it across sessions). The pattern is to keep `agent.jsonState()` and add a middleware that contributes a typed slice but loads and saves it from the database, keyed by something stable like the user id:

```ts
import { middleware } from "@substructure.ai/sdk";

type Todo = { id: string; title: string; done: boolean };
type TodoData = { items: Todo[] };

const todoSlice = middleware<{ todos: TodoData }>({
  state: { todos: { items: [] } },
  handler: async (req, next) => {
    const userId = req.wire.identity.id;
    req.state.todos = (await db.loadTodos(userId)) ?? { items: [] };

    const res = await next(req);

    await db.saveTodos(userId, req.state.todos);
    req.state.todos = { items: [] };   // DB has the items; don't ship them again
    return res;
  },
});
```

Tools opt into the slice and see `state.todos: TodoData`, fully typed without casts. The DB is the source of truth; the wire only ever carries `{ todos: { items: [] } }` so it stays small. Because the key is the user id rather than the session id, the same todo list shows up across every conversation that user has with the agent.

Tools opt in as normal:

```ts
const addTodo = agent.tool({
  name: "add_todo",
  description: "Add a todo",
  parameters: {
    type: "object",
    properties: { title: { type: "string" } },
    required: ["title"],
  },
  state: todoSlice,
  execute: (args, state) => {
    const { title } = JSON.parse(args);
    const item = { id: crypto.randomUUID(), title, done: false };
    state.todos.items.push(item);
    return item;
  },
});
```

And the chain stays small: one middleware covers both the slice and the persistence.

```ts
const todoAgent = agent({ id: "todo" })
  .use(agent.jsonState())          // wire <-> req.state
  .use(todoSlice)                  // contributes + hydrates `todos`
  .use(agent.messageHistory())     // wire-backed via jsonState
  .use(agent.tools([addTodo]))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-5" } }));
```

What ends up where:

- **On the wire:** `{ messages: [...], todos: { ref: "session-123" } }`. The conversation history rides along, the todos are just a pointer.
- **In your database:** the actual `{ items: [...] }`, keyed by session id.

The same trick scales to multiple DB-backed slices: chain a `hydrateX(db)` for each one. Keep all consumers (tools, other middleware) below the hydrate middleware so they see the loaded form, not the ref.

Full runnable version: [`examples/hybrid-state`](https://github.com/substructureai/substructure/tree/main/examples/hybrid-state).

### Contributing typed state

If your middleware needs its own state slice (like the built-ins do), declare it with `state` and the slice will be initialized and typed for you:

```ts
import { middleware } from "@substructure.ai/sdk";

type RateState = { callsThisTurn: number };

const rateLimit = middleware<RateState>({
  state: { callsThisTurn: 0 } as RateState,
  handler: async (req, next) => {
    req.state.callsThisTurn += 1;
    if (req.state.callsThisTurn > 10) {
      return { actions: [{ type: "done", data: "rate limit exceeded" }], state: req.state };
    }
    return next(req);
  },
});
```

The contributed slice is merged with whatever other slices the chain declares, so multiple middlewares and tools can share state without colliding.

## Serving as a worker

Wrap one or more agents into a worker, then expose its `fetchHandler` from your HTTP server:

```ts
const worker = sub.worker({ agents: [weatherAgent] });

export default {
  fetch: worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET }),
};
```

`fetchHandler` returns a plain `(Request) => Promise<Response>` function, so it works in any fetch-compatible runtime:

- **Cloudflare Workers / Vercel / Deno / Bun**: export it directly as the default `fetch`.
- **Hono**: `app.post("/agent", (c) => handler(c.req.raw))`.
- **Node + Express / Fastify**: adapt the request/response using a fetch shim.

The worker is stateless. Each request is one decision; the engine holds the durable state. Scale to zero, deploy to any serverless platform.

### Signing secrets

`signingSecret` is the secret you got when you ran `substructure cloud apps create`. The handler verifies an HMAC-SHA256 `X-Substructure-Signature` header on every request. Skip the option to disable verification (only for local development).

## Submitting turns from a client

### Backend client

Use the backend client from any server. It authenticates with an API key minted via `substructure cloud keys create`.

```ts
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

`startTurn` returns a `SessionScope` containing `sessionId` and `turnId`. From there you have two choices:

- `await client.turnResult(scope)` waits for the turn to finish and returns `{ data, cost, tokenUsage }`.
- `for await (const event of client.stream(scope))` streams individual events as they arrive: LLM responses, tool calls, sub-agent updates, and so on. Use `sequenceAfter` to resume from a known event.

The client also exposes admin APIs: `listSessions`, `getSession`, and `sessionEvents` for tooling and dashboards.

### Browser client

`sub.frontend.client({ token })` is the browser equivalent. The shape is identical (`startTurn`, `stream`, `turnResult`), but it authenticates with a short-lived user token instead of an API key. Never ship an API key to the browser.

The flow is two steps: your backend mints a token for a specific user, the browser uses that token to drive the turn.

**Backend: mint a token for the logged-in user.**

```ts
// app/api/agent-token/route.ts (Next.js, but any backend works)
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const client = sub.backend.client({ apiKey: process.env.SUBSTRUCTURE_API_KEY! });

export async function POST(req: Request) {
  const user = await authenticateUser(req); // your auth

  const { token, expiresAt } = await client.mintClientToken({
    identity: { id: user.id },
    ttlSeconds: 60 * 15,
  });

  return Response.json({ token, expiresAt });
}
```

The token is scoped to that identity. The browser can only submit turns as that user; it can't impersonate anyone else even though it holds the token directly.

**Browser: use the token to start a turn and stream events.**

```ts
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();

const { token } = await fetch("/api/agent-token", { method: "POST" }).then(r => r.json());

const client = sub.frontend.client({ token });

const scope = await client.startTurn({
  agentId: "weather-agent",
  payload: {
    type: "message",
    message: { role: "user", content: "What's the weather in SF?" },
  },
});

for await (const event of client.stream(scope)) {
  if (event.payload.type === "message.new") {
    appendToUi(event.payload);
  }
}
```

Note that the browser does not pass `identity`: it's already baked into the token. Mint a fresh token when the current one nears `expiresAt`, or on every page load if your TTL is short.

## Embedded runtime

For local development, testing, or single-machine deployments, you can run the Substructure engine in-process. No separate server, no cloud account, just a SQLite file.

The engine itself is a native Rust binary, so it ships as a separate package. Install it alongside the SDK:

```sh
npm i @substructure.ai/runtime
```

It's listed as an optional peer of `@substructure.ai/sdk`, so the main SDK install doesn't pull it down by default. `sub.embedded(...)` will throw at call time if it can't find the runtime package.

Then use it like this:

```ts
const embedded = await sub.embedded({
  agents: [todoAgent],
  db: "agent.db",
  openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
  agentId: "todo",
  payload: {
    type: "message",
    message: { role: "user", content: "Add 'buy groceries' and list my todos" },
  },
  identity: { id: "demo" },
});

const { data } = await embedded.turnResult(scope);
console.log(data);

await embedded.shutdown();
```

The embedded instance exposes the same `startTurn` / `stream` / `turnResult` surface as the backend client, plus a `fetchHandler` if you want to put an HTTP face on it. Use `db: ":memory:"` for a transient instance in tests.

## Models

Models are specified inside `llmLoop`:

```ts
agent.llmLoop({
  request: { model: "anthropic/claude-sonnet-4-5" },
});
```

Substructure uses OpenRouter under the hood, so any OpenRouter model identifier works. When running embedded, pass `openrouterApiKey` to `sub.embedded(...)`. With cloud or local server, the provider credentials live on the server (set via `OPENROUTER_API_KEY` when you ran `substructure local start`, or configured for your org in the cloud dashboard).

## Examples

See [`examples/`](https://github.com/substructure-ai/substructure/tree/main/examples) for full deployments:

- `node-embedded` — in-process agent with persistent SQLite state.
- `cloudflare-worker` — worker deployed to Cloudflare, with a backend client driving turns.
- `hono` — `fetchHandler` mounted on a Hono route in Node.
- `vercel` — serverless worker on Vercel.
- `sub-agent` — a parent agent delegating to a child via `subAgents`.
- `hybrid-state` — most state on the wire via `jsonState`, one slice swapped in and out of a database.
