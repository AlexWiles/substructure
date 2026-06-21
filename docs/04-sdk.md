---
title: TypeScript SDK
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
- `SubstructureEmbedded.create(...)` (from `@substructure.ai/sdk/embedded`) — run the engine in-process with a SQLite event log.

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
    return JSON.stringify({ city, temp_f: 62, condition: "sunny" });
  },
});
```

A few notes:

- `parameters` is a plain JSON Schema object. The LLM uses it to format calls.
- `execute` receives the raw stringified JSON args. Parse it yourself.
- Return a string — the tool result fed back to the LLM. For structured data, `JSON.stringify` it yourself. (Or return `ctx.defer()` to complete the call out-of-band.)
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
    return JSON.stringify(item);
  },
});
```

The `state` you mutate inside `execute` is durably persisted by Substructure across turns.

### Deferred (async) tool calls

By default, the value `execute` returns *is* the tool result: it ships back to the LLM as soon as the worker finishes the decision. That works when the answer is already in hand by the time `execute` returns (a database lookup, an HTTP call you `await`, a computation).

It does not work for tools that hand work off to something the worker can't await — a webhook callback, a queued job, a human approval, an external system that pings you when ready. For those, `execute` calls `ctx.defer()`, kicks off the work, and the result arrives later via `submitToolCallResult`.

```ts
const wait = agent.tool({
  name: "wait",
  description: "Wait for the given number of seconds, then return.",
  parameters: {
    type: "object",
    properties: { seconds: { type: "number" } },
    required: ["seconds"],
  },
  execute: (args, ctx) => {
    const { seconds } = JSON.parse(args);

    setTimeout(() => {
      client.submitToolCallResult({
        sessionId: ctx.sessionId,
        toolCallId: ctx.toolCallId,
        attempt: ctx.attempt,
        result: JSON.stringify({ waited_seconds: seconds }),
      });
    }, seconds * 1000);

    return ctx.defer();
  },
});
```

`client` here is a backend client minted with your API key:

```ts
const client = sub.backend.client({
  url: "https://api.substructure.ai",
  apiKey: process.env.SUBSTRUCTURE_API_KEY!,
});
```

The third argument to `execute` is a `ToolExecutionContext`:

- `ctx.sessionId` — the session this call belongs to.
- `ctx.toolCallId` — the LLM-assigned id you must pass back.
- `ctx.attempt` — the current retry attempt; pass it back unchanged.
- `ctx.defer()` — returns the sentinel value to `return`.

Capture the ids *before* you return, since the worker decision ends as soon as `execute` returns.

What happens on the wire:

1. The LLM emits a tool call. The engine records it as pending and dispatches a `tool.execute` trigger to your worker.
2. Your `execute` returns `ctx.defer()`. The `tools` middleware emits no `return.tool.result`, so the worker submits zero actions for that decision. The engine leaves the tool call pending.
3. Later — minutes, hours, however long — the external work completes. You call `submitToolCallResult({ tool_call_id, result, attempt })`. The engine treats this exactly like a synchronous tool return: emits `tool.call.completed`, fires a `tool.result` trigger, the chain runs, and `llmToolLoop` issues the next `call.llm` once every pending tool result is in.

`submitToolCallResult` is available on every flavor of client — `sub.backend.client(...)` on your servers, `sub.frontend.client(...)` in the browser, and `SubstructureEmbedded.create(...)` for in-process runs — so the call back can come from wherever finishes the work (a webhook handler, a queue worker, a UI button). To report a failure instead of a result, pass `error` (and optional `retryable`) in place of `result`.

If you never call `submitToolCallResult`, the tool call stays pending forever — the agent will not resume. For tools where that's a real risk, set a `retry` policy on the tool (which carries a `timeout_secs`) so the engine eventually fails the call and lets the chain see a `tool.result` with `is_error: true`.

Full runnable version: [`examples/deferred-tool`](https://github.com/substructureai/substructure/tree/main/examples/deferred-tool).

## Defining client actions

Tools react to the LLM; actions react to the client. A **client action** is a named handler that fires when a client calls `startTurn({ payload: { type: "action", name, args } })`. Use them for anything that isn't a chat message: approvals, cancellations, replays, typed events from a UI.

Define one with `agent.action`:

```ts
const approveCommand = agent.action({
  name: "approve_command",
  handler: (args: { approved: boolean; reason?: string }) => {
    console.log("approval received:", args);
  },
});
```

Register them with `agent.actions(...)` in the chain:

```ts
const myAgent = agent({ id: "..." })
  .use(agent.actions([approveCommand]))
  .use(/* ... */);
```

A few notes:

- `handler` receives `args` cast to the type you declare. There's no runtime validation; treat it like a typed `JSON.parse`.
- Return `void` to let the chain continue, so `llmToolLoop` runs as it would for a normal trigger. Return a `WorkerAction[]` to short-circuit with exactly those actions.
- Pass `state: someSlice` to get typed access to that slice inside the handler, same as `agent.tool`.

Clients submit an action the same way they submit a message, just with a different payload shape:

```ts
await client.startTurn({
  agentId: "...",
  sessionId,
  payload: { type: "action", name: "approve_command", args: { approved: true } },
  identity: { id: "user-1" },
});
```

## Building an agent

An agent is a chain of middleware. You start with `agent({ id })` and stack behavior with `.use(...)`:

```ts
const weatherAgent = agent({ id: "weather-agent" })
  .use(agent.messageHistory("You are a helpful weather assistant."))
  .use(agent.tools([getWeather]))
  .use(agent.llmToolLoop({
    generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-5" }),
  }));
```

Agent state is serialized as JSON and carried across turns for you. To keep large or sensitive state off the wire, see [Keep conversation state in your own database](#example-keep-conversation-state-in-your-own-database).

The built-in middleware:

| Middleware | What it does |
| --- | --- |
| `agent.messageHistory(system?, opts?)` | Tracks the full message history across turns and injects it into LLM calls. Pass `system`, a string or a `(state, ctx) => string` selector, to also prepend a system message. Pass `{ stateKey }` as a second arg to change where the transcript lives (default `"messages"`). |
| `agent.messageHistoryCurrentTurn(system?, opts?)` | Same arguments, but scoped to a single turn. |
| `agent.tools([...])` | Registers tools, dispatches tool calls from the LLM, and feeds results back. |
| `agent.actions([...])` | Dispatches `client.action` triggers to their handlers. See [Defining client actions](#defining-client-actions). |
| `agent.llmToolLoop({ generator })` | Drives the core loop: on a user message or tool result, call the LLM; on an LLM response with no tool calls, finish the turn. |
| `agent.subAgents({ agents })` | Lets the agent delegate to child agents as if they were tools. See [Sub-agents](./05-sub-agents.md). |
| `agent.logging()` | Logs each decision lifecycle to stdout. Handy in development. |

The order matters. State middleware first, then context (history, which also carries the system message), then tools, then `llmToolLoop` at the end to drive the loop.

### Writing your own middleware

If the built-ins don't cover what you need, write your own. A middleware is just a function:

```ts
import type { MiddlewareFn } from "@substructure.ai/sdk";

const timing: MiddlewareFn = async (ctx, next) => {
  const start = Date.now();
  const res = await next(ctx);
  console.log(`decision took ${Date.now() - start}ms`);
  return res;
};

const myAgent = agent({ id: "..." })
  .use(timing)
  .use(/* ... */);
```

The middleware receives:

- `ctx` — the decision context. The interesting fields are `ctx.state` (the agent state so far) and `ctx.request` (the raw decision envelope, including `request.session_id`, `request.turn_id`, `request.identity`, and `request.trigger`).
- `next(ctx)` — runs the rest of the chain. Returns a response containing `actions` (what the engine will do next), `state` (the new state to persist), and optionally `workerState` (the raw, serialized form Substructure sends back next time).

You can mutate `ctx` before calling `next`, inspect or rewrite `res.actions` after, or short-circuit entirely.

### Example: keep conversation state in your own database

By default your agent's state is serialized as JSON and round-tripped through Substructure on every request. That's fine for small state, but for large message histories or sensitive data you may want state to live entirely in your own database. Write a middleware that loads state on the way in, saves it on the way out, and returns an empty placeholder so nothing heavy rides the wire:

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
    handler: async (ctx, next) => {
      const userId = ctx.request.identity.id;
      const sessionId = ctx.request.session_id;

      const loaded = await db.loadAgentState(userId, sessionId);
      if (loaded) ctx.state = loaded;

      const res = await next(ctx);

      // The DB is the source of truth. Persist the full state, then hand the
      // wire back an empty placeholder so the (possibly large) state never
      // rides along. Next turn this middleware reloads it from the DB.
      await db.saveAgentState(userId, sessionId, res.state);

      return { ...res, state: { messages: [], ticketId: null } satisfies SupportState };
    },
  });

const myAgent = agent({ id: "support" })
  .use(dbState(db))
  .use(agent.messageHistory("You are a support agent."))
  .use(agent.tools([/* ... */]))
  .use(agent.llmToolLoop({ generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-5" }) }));
```

The `state` field on `middleware` does two things: it gives you the initial value used on the first turn, and it locks in the type so `ctx.state` is typed inside `handler` and any tool that takes `state: this slice` gets the same type. Downstream middleware like `messageHistory` will populate `ctx.state.messages` for you; `ticketId` is a slot you can read and write from your own tools.

`ctx.request.identity.id` is the user id the client passed when calling `startTurn`, and `ctx.request.session_id` is the conversation. Keying on both means a single user can have multiple parallel conversations and you can scope, list, or delete state per user without ever touching Substructure.

Because the state you return is what gets serialized onto the wire, the way-out half hands back an empty placeholder rather than the loaded data. Substructure ships that tiny blob, passes it back as the wire state on the next turn, and your middleware loads the real state from the DB, so the heavy or sensitive data never leaves your infrastructure.

The same pattern works for anything that needs to bridge the agent to your infrastructure: pulling user profile data into state, writing audit logs alongside the response, gating tool calls by feature flag, or short-circuiting a turn when a per-user quota is exceeded.

### Example: hybrid wire and database state

Sometimes you want most state on the wire (cheap, no infrastructure) and just one slice in your database (because it's large, sensitive, or you want to share it across sessions). Since state rides the wire by default, you just add a middleware that contributes a typed slice but loads and saves it from the database, keyed by something stable like the user id, and empties the field on the way out so only that one slice stays off the wire:

```ts
import { middleware } from "@substructure.ai/sdk";

type Todo = { id: string; title: string; done: boolean };
type TodoData = { items: Todo[] };

const todoSlice = middleware<{ todos: TodoData }>({
  state: { todos: { items: [] } },
  handler: async (ctx, next) => {
    const userId = ctx.request.identity.id;
    ctx.state.todos = (await db.loadTodos(userId)) ?? { items: [] };

    const res = await next(ctx);

    await db.saveTodos(userId, ctx.state.todos);
    ctx.state.todos = { items: [] };   // DB has the items; don't ship them again
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
    return JSON.stringify(item);
  },
});
```

And the chain stays small: one middleware covers both the slice and the persistence.

```ts
const todoAgent = agent({ id: "todo" })
  .use(todoSlice)                  // contributes + hydrates `todos`
  .use(agent.messageHistory())     // rides the wire by default
  .use(agent.tools([addTodo]))
  .use(agent.llmToolLoop({ generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-5" }) }));
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
  handler: async (ctx, next) => {
    ctx.state.callsThisTurn += 1;
    if (ctx.state.callsThisTurn > 10) {
      return { actions: [{ type: "done", data: "rate limit exceeded" }], state: ctx.state };
    }
    return next(ctx);
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

`signingSecret` is the secret you got when you ran `substructure apps create`. The handler verifies an HMAC-SHA256 `X-Substructure-Signature` header on every request. Skip the option to disable verification (only for local development).

## Submitting turns from a client

There are two clients for talking to a deployed worker, picked by where the code runs:

- `sub.backend.client({ apiKey })` — for code that runs on **your servers**. Authenticates with a long-lived API key. Can act as any identity and exposes admin APIs (`listSessions`, `getSession`, `sessionEvents`).
- `sub.frontend.client({ token })` — for code that runs in **the browser** (or any untrusted environment). Authenticates with a short-lived per-user token your backend mints. Scoped to a single identity; no admin APIs.

The two have the same core surface (`startTurn`, `stream`, `turnResult`), so most agent code is identical regardless of which client drives it. Pick by trust boundary, not by feature.

### Backend client

Use the backend client from any server. It authenticates with an API key minted via `substructure keys create`.

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
- `for await (const event of client.stream(scope))` streams individual events as they arrive: LLM responses, tool calls, sub-agent updates, and so on. Use `sequenceAfter` to resume from a known event. By default the stream yields only persisted events, so you can `switch` on `event.payload.type` directly. Pass `{ tokens: true }` to also receive transient `llm.token.delta` items for progressive rendering (only emitted when the agent's `llmToolLoop` has `stream: true`) — they arrive as bare payloads (no envelope, no `sequence`), are not replayed on reconnect, and are discriminated with the exported `isTokenDelta(event)` guard.

The client also exposes admin APIs: `listSessions`, `getSession`, and `sessionEvents` for tooling and dashboards.

### Frontend client

`sub.frontend.client({ token })` is the browser-side counterpart. The shape mirrors the backend client (`startTurn`, `stream`, `turnResult`, `submitToolCallResult`), but it authenticates with a short-lived user token instead of an API key, so it's safe to use in code shipped to a browser, a mobile app, or any other untrusted environment. **Never ship an API key to a client.**

Reach for the frontend client when:

- You want a chat UI, dashboard, or other interface to talk to your agent directly from the browser without round-tripping each message through your backend.
- You want to stream events (token-by-token responses, tool calls, sub-agent progress) straight to the UI over SSE without standing up your own proxy.
- You're building a mobile or desktop client that has user-level auth but no shared secret with Substructure.

Stay on the backend client when the caller is a trusted server, when you need admin APIs like `listSessions`, or when you want to act as multiple identities from one process (cron jobs, webhooks, batch jobs).

Key differences from the backend client:

- **Auth.** Authorized with a JWT minted by your backend via `client.mintClientToken({ identity, ttlSeconds })`. The token is bound to a single identity and expires.
- **No `identity` field on `startTurn`.** The identity is already baked into the token; the browser can't impersonate other users even though it holds the token directly.
- **No admin APIs.** `listSessions`, `getSession`, and `sessionEvents` are server-only.
- **Endpoint surface.** The frontend client talks to `/api/client/*` routes that are scoped to the token's identity; the backend client talks to `/api/worker/*` and `/api/admin/*`.

The typical flow is two steps: your backend mints a token for the signed-in user, the browser uses that token to drive the turn.

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

for await (const event of client.stream(scope, { tokens: true })) {
  if (isTokenDelta(event)) {
    // Transient live chunk. Order within a call by `seq` and append to the
    // in-progress assistant bubble. Drop the partial when the matching
    // llm.call.completed arrives — the persisted message.new that follows
    // carries the canonical content.
    appendDelta(event);
  } else if (event.payload.type === "message.new") {
    appendToUi(event.payload);
  }
}
```

Token deltas only flow when the agent's `llmToolLoop` was configured with `stream: true`. They're transient: the engine does not persist them, and a client reconnecting mid-call will not see deltas already emitted — only the final `message.new` once the call completes.

Note that the browser does not pass `identity`: it's already baked into the token. Mint a fresh token when the current one nears `expiresAt`, or on every page load if your TTL is short.

## Embedded runtime

For scripts, tests, and local development, you can run the Substructure engine in-process.

The engine itself is a native Rust binary, so it ships as a separate package. Install it alongside the SDK:

```sh
npm i @substructure.ai/runtime
```

It's listed as an optional peer of `@substructure.ai/sdk`, so the main SDK install doesn't pull it down by default. The embedded entry lives at its own subpath, `@substructure.ai/sdk/embedded`, so the main `@substructure.ai/sdk` entry stays free of the native dependency and bundles cleanly for workers/edge. `SubstructureEmbedded.create(...)` will throw at call time if it can't find the runtime package.

Then use it like this:

```ts
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

const embedded = await SubstructureEmbedded.create({
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

For the Substructure server to make the LLM call, pass a `serverGenerate` generator to `llmToolLoop`:

```ts
agent.llmToolLoop({
  generator: agent.serverGenerate({ model: "anthropic/claude-sonnet-4-5" }),
});
```

Server models are identified by `provider/model` strings. When running embedded or locally, provider credentials are read from the environment; with cloud, they're configured for your org in the dashboard. To make the call from your own worker instead, pass a provider generator (`anthropicGenerate`, `openaiGenerate`, `aiGenerate`) from `@substructure.ai/sdk/adapters/*`.

## Examples

See [`examples/`](https://github.com/substructureai/substructure/tree/main/examples) for full deployments:

- `node-embedded` — in-process agent with persistent SQLite state.
- `cloudflare-worker` — worker deployed to Cloudflare, with a backend client driving turns.
- `hono` — `fetchHandler` mounted on a Hono route in Node.
- `vercel` — serverless worker on Vercel.
- `sub-agent` — a parent agent delegating to a child via `subAgents`.
- `hybrid-state` — most state on the wire as JSON, one slice swapped in and out of a database.
- `deferred-tool` — async tool call: `execute` returns `ctx.defer()`, the result is posted later via `submitToolCallResult`.
- `frontend-tool` — chat UI where tools run in the browser (geolocation, theme). The worker defers; the page executes locally and posts the result back via `submitToolCallResult` using the frontend client. Also demonstrates `stream: true` on `llmToolLoop` — the assistant message renders token-by-token from `llm.token.delta` events.
