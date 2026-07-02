---
title: TypeScript SDK
---

The `@substructure.ai/sdk` package is how you build agents and connect to Substructure from TypeScript. It gives you three things:

- An **agent** — `agent({ name, decide })`, where `decide` is `toolLoop(...)` for the common tool/sub-agent loop, or your own function for full control over each decision.
- A **worker** — `worker([...]).fetch(...)`, which exposes those agents as an HTTP endpoint the engine calls into.
- **Clients** — for submitting turns from your backend or browser.

## Install

```sh
npm i @substructure.ai/sdk
```

## The model: an agent is a decision function

The engine is the runtime. It owns the durable session, the conversation tree, scheduling, and retries. An agent is `agent({ name, decide })`: `name` is the id the engine routes to, and `decide` is the function it calls once per **decision**. The engine sends a **`DecisionRequest`** — a trigger (a user message, an LLM response, a tool result) plus the active transcript, state, and pending effects — and `decide` returns a **`Decision`**: the **actions** to take (call the LLM, call a tool, finish the turn), plus the transcript and state as they should now read.

You write `decide` one of two ways:

- `toolLoop({...})` — the batteries-included loop. Give it a model, instructions, and tools, and it handles the whole user → LLM → tools → LLM cycle for you.
- `async (req) => {...}` — your own decision function. You get the `DecisionRequest` and return the `Decision` yourself. Nothing is hidden: branching, approval gates, mode switches, and compaction are all just code that returns actions.

`toolLoop(config)` returns a decision function, so a custom `decide` can start from the default loop and override a single case — call `toolLoop(config)(req)` to delegate to it.

## `agent` and `toolLoop`

The common case. An agent is `agent({ name, decide })`; for the default loop, `decide` is `toolLoop(...)` — instructions + model + tools is a working agent:

```ts
import { agent, tool, toolLoop } from "@substructure.ai/sdk";

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
    llm: { model: "anthropic/claude-sonnet-4-6" },
    instructions: "You are a helpful weather assistant.",
    tools: [getWeather],
  }),
});
```

The `agent` config is `{ name, decide }`: `name` is the id clients target and the worker registers; `decide` is the decision function. `toolLoop` takes:

| Field | What it does |
| --- | --- |
| `llm` | The LLM to call — `{ model: "provider/model" }` (the Substructure server makes the call) or an adapter generator that runs it on your worker. Add `stream: true` here to stream tokens; add `temperature`/`reasoning`/etc. alongside `model`. |
| `instructions` | The system prompt. A string, or a function resolved per call. |
| `tools` | Worker- and client-handled tools (see below). |
| `subAgents` | Child agents the model can delegate to as tools, referenced by value. See [Sub-agents](./05-sub-agents.md). |

`agent(...)` returns an `Agent` — pass it straight to `worker([...])` or `SubstructureEmbedded.create({ agents })`.

## Defining tools

Tools are how the agent acts on the world. A tool is a pure function of its arguments:

```ts
const getWeather = tool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  execute: (args, ctx) => {
    const { city } = JSON.parse(args);
    return JSON.stringify({ city, temp_f: 62, condition: "sunny" });
  },
});
```

A few notes:

- `parameters` is a plain JSON Schema object. The LLM uses it to format calls.
- `execute` receives the raw stringified JSON args. Parse it yourself.
- Return a string — the tool result fed back to the LLM. For structured data, `JSON.stringify` it yourself. (Or return `ctx.defer()` to complete the call out-of-band.)
- Tools can be async. Substructure waits for the promise.
- `execute(args, ctx)` takes only its arguments and a `ToolExecutionContext`. **There is no SDK-held tool state** — see [State](#state) for where state lives.

### Deferred (async) tool calls

By default, the value `execute` returns *is* the tool result: it ships back to the LLM as soon as the worker finishes the decision. That works when the answer is in hand by the time `execute` returns (a database lookup, an HTTP call you `await`, a computation).

It does not work for tools that hand work off to something the worker can't await — a webhook callback, a queued job, a human approval. For those, `execute` calls `ctx.defer()`, kicks off the work, and the result arrives later via `submitToolCallResult`.

```ts
const wait = tool({
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

`client` here is a backend client minted with your API key (see [Submitting turns from a client](#submitting-turns-from-a-client)). The `ToolExecutionContext` gives you:

- `ctx.sessionId` — the session this call belongs to.
- `ctx.toolCallId` — the LLM-assigned id you must pass back.
- `ctx.attempt` — the current retry attempt; pass it back unchanged.
- `ctx.request` — the full decision envelope (`request.identity`, `request.turn_id`, ...).
- `ctx.defer()` — returns the sentinel value to `return`.

Capture the ids *before* you return, since the worker decision ends as soon as `execute` returns.

What happens on the wire:

1. The LLM emits a tool call. The engine records it as pending and dispatches an `effect.execute` trigger to your worker.
2. Your `execute` returns `ctx.defer()`. The worker emits no `effect.result`, so the engine leaves the tool call pending.
3. Later — minutes, hours, however long — the external work completes. You call `submitToolCallResult({...})`. The engine treats it exactly like a synchronous tool return: fires an `effect.settled` trigger, and the loop issues the next `call.llm` once every pending result is in.

`submitToolCallResult` is available on every flavor of client, so the callback can come from wherever finishes the work. To report a failure instead, pass `error` (and optional `retryable`) in place of `result`. If you never call it, the tool stays pending forever — set a `retry` policy (with a `timeout_secs`) on the tool so the engine eventually fails the call and the loop sees an `effect.settled` with `ok: false`.

Full runnable version: [`examples/deferred-tool`](https://github.com/substructureai/substructure/tree/main/examples/deferred-tool).

### Client (frontend) tools

A tool with `handler: "client"` runs in the browser, not the worker. The engine routes the call to the frontend, which completes it via `submitToolCallResult`. The worker never runs `execute`, so it's optional:

```ts
const setTheme = tool({
  name: "set_theme",
  description: "Set the page's colors. Runs in the user's browser.",
  parameters: {
    type: "object",
    properties: { background: { type: "string" }, accent: { type: "string" } },
    required: ["background", "accent"],
  },
  handler: "client",
});
```

See [`examples/frontend-tool`](https://github.com/substructureai/substructure/tree/main/examples/frontend-tool).

## State

There is no SDK-held tool state. State lives in one of two places, and you choose per agent:

### In your own store

Tools reach their store directly through `ctx`. Key it by `ctx.sessionId` (per conversation) or `ctx.request.identity.id` (per user) — whichever scope you want. This keeps large or sensitive state entirely in your infrastructure; only the conversation tree (owned by the engine) and whatever you put on the wire ever leave it.

```ts
const addTodo = tool({
  name: "add_todo",
  description: "Add a todo item",
  parameters: {
    type: "object",
    properties: { title: { type: "string" } },
    required: ["title"],
  },
  execute: async (args, ctx) => {
    const { title } = JSON.parse(args);
    const userId = ctx.request.identity.id;
    const data = await db.loadTodos(userId);
    const item = { id: crypto.randomUUID(), title, done: false };
    data.items.push(item);
    await db.saveTodos(userId, data);
    return JSON.stringify(item);
  },
});
```

Swap `db` for Postgres, Redis, S3, or a Durable Object — the agent doesn't change. The [`cloudflare-worker`](https://github.com/substructureai/substructure/tree/main/examples/cloudflare-worker) example keys a Durable Object by `ctx.sessionId`; [`hybrid-state`](https://github.com/substructureai/substructure/tree/main/examples/hybrid-state) keys a file store by user id so a todo list follows the user across sessions.

### On the wire (custom `decide`)

If you want small state round-tripped for you instead of standing up a store, keep it in `worker_state` with a custom `decide`. The engine ships the decoded state in as `req.state`. Build the tools **inside `decide`** so each `execute` closes over the live `state`, hand them to `toolLoop`, and pass `state` into the loop — the loop runs the tools, and echoes the `state` you gave it so the agent persists it:

```ts
import { agent, tool, toolLoop } from "@substructure.ai/sdk";

type State = { todos: Todo[] };

// Built per decision, closing over the live list; `toolLoop` runs them.
function todoTools(state: State) {
  return [
    tool({
      name: "add_todo",
      description: "Add a todo item",
      parameters: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
      execute: (args) => {
        state.todos.push({ id: crypto.randomUUID(), title: JSON.parse(args).title, done: false });
        return "added";
      },
    }),
    tool({ name: "list_todos", description: "List todos", parameters: { type: "object", properties: {} },
           execute: () => JSON.stringify(state.todos) }),
  ];
}

const todoAgent = agent<State>({
  name: "todo",
  decide: async (req) => {
    const state: State = { todos: req.state?.todos ?? [] };
    const loop = toolLoop<State>({
      llm: { model: "anthropic/claude-sonnet-4-6" },
      instructions: "Concise todo assistant.",
      tools: todoTools(state),
    });
    return loop({ ...req, state });   // pass state in → the loop persists it back out
  },
});
```

`toolLoop` echoes the `state` it's given, so a custom `decide` that needs to intercept a trigger (a mode switch, an approval gate) can do that work, then `return loop({ ...req, state })` for everything else. See [`examples/state-hydration`](https://github.com/substructureai/substructure/tree/main/examples/state-hydration) and [`examples/plan-mode`](https://github.com/substructureai/substructure/tree/main/examples/plan-mode).

## Custom decision functions

When the default loop isn't the shape you want — an approval gate, a modal agent, custom routing, compaction — write `decide` directly. The engine hands you a `DecisionRequest` and you return a `Decision`:

```ts
import { agent } from "@substructure.ai/sdk";

const echo = agent({
  name: "echo",
  decide: (req) => {
    switch (req.trigger.type) {
      case "user.message":
        return {
          actions: [
            {
              type: "call.llm",
              request: { model: "anthropic/claude-sonnet-4-6", messages: [...(req.transcript ?? []), req.trigger.message] },
              handler: "server",
            },
          ],
        };
      case "effect.settled":
        if (req.trigger.kind !== "llm_call" || !req.trigger.message) return {};
        return { actions: [{ type: "done", data: req.trigger.message.content ?? null }] };
      default:
        return {};
    }
  },
});
```

The `DecisionRequest` (conventionally `req`) is the engine's wire envelope with `worker_state` decoded into `state`. Read off it:

- `req.trigger` — what happened: `user.message`, `user.transcript`, `client.action`, `effect.execute`, `effect.settled`, ...
- `req.transcript` — the active transcript (the head-to-root path); may be empty.
- `req.state` — the decoded `worker_state` (return a new value to persist it).
- `req.effects` — the in-flight effects as a flat, tagged list (each with `id`, `kind`, `status`, `attempt`, plus kind-specific fields like a tool's `name`/`arguments`); branch on `kind` + `status` to know when a tool step is complete (no `tool_call`/`sub_agent` effect left in flight). `kind`/`status` are open, so new effect kinds are additive.
- `req.session_id`, `req.identity`, `req.turn_id`, ... — the rest of the envelope, read directly.

Return a `Decision` — `{ actions?, transcript?, state? }`. `actions` defaults to none; `transcript` echoes `req.transcript`; `state` echoes `req.state`.

Actions are plain objects — you return them directly:

| Action | Shape |
| --- | --- |
| Ask the model | `{ type: "call.llm", request: { model, messages, tools? }, handler }` |
| Run a tool | `{ type: "call.tool", id, name, arguments, handler }` |
| Return an effect result / error | `{ type: "effect.result", kind, id, result \| response, attempt }` / `{ type: "effect.error", kind, id, error, retryable, attempt }` |
| Delegate to a sub-agent | `{ type: "spawn.sub_agent", session_id, agent_id, tool_call_id }` + `{ type: "send.message", session_id, message }` |
| Finish the turn | `{ type: "done", data }` |

A `client.action` trigger is how a custom `decide` reacts to the client — approvals, mode switches, replays. The [`tool-approval`](https://github.com/substructureai/substructure/tree/main/examples/tool-approval) example parks a tool call when the model replies (`effect.settled`, `kind: "llm_call"`) and re-emits it on a `client.action approve_command`; [`plan-mode`](https://github.com/substructureai/substructure/tree/main/examples/plan-mode) reads its mode from state and forks a fresh branch when it switches to executing.

## Serving as a worker

Wrap one or more agents into a worker and expose its fetch handler:

```ts
import { worker } from "@substructure.ai/sdk";

export default {
  fetch: worker([weatherAgent]).fetch({ signingSecret: process.env.SIGNING_SECRET }),
};
```

`worker([...]).fetch(...)` returns a plain `(Request) => Promise<Response>`, so it works in any fetch-compatible runtime (`serve([weatherAgent], opts)` is a one-liner shorthand for the same thing):

- **Cloudflare Workers / Vercel / Deno / Bun**: export it directly as the default `fetch`.
- **Hono**: assign the returned fetch function to a const and call it in the route — `app.post("/agent", (c) => agentFetch(c.req.raw))`.
- **Node + Express / Fastify**: adapt the request/response using a fetch shim.

The worker is stateless. Each request is one decision; the engine holds the durable state. Scale to zero, deploy to any serverless platform.

### Signing secrets

`signingSecret` is the secret you got when you ran `substructure apps create`. The handler verifies an HMAC-SHA256 `X-Substructure-Signature` header on every request. Skip the option to disable verification (only for local development).

## Submitting turns from a client

There are two clients for talking to a deployed worker, picked by where the code runs:

- `sub.backend.client({ apiKey })` — for code that runs on **your servers**. Authenticates with a long-lived API key. Can act as any identity and exposes admin APIs (`listSessions`, `getSession`, `sessionEvents`).
- `sub.frontend.client({ token })` — for code that runs in **the browser** (or any untrusted environment). Authenticates with a short-lived per-user token your backend mints. Scoped to a single identity; no admin APIs.

Both come from a `Substructure` instance:

```ts
import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
```

The two have the same core surface (`startTurn`, `stream`, `turnResult`), so most code is identical regardless of which client drives it. Pick by trust boundary, not by feature.

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
- `for await (const event of client.stream(scope))` streams individual events as they arrive: LLM responses, tool calls, sub-agent updates, and so on. Use `sequenceAfter` to resume from a known event. By default the stream yields only persisted events, so you can `switch` on `event.payload.type` directly. Pass `{ tokens: true }` to also receive transient `llm.token.delta` items for progressive rendering (only emitted when the agent has `stream: true`) — they arrive as bare payloads (no envelope, no `sequence`), are not replayed on reconnect, and are discriminated with the exported `isTokenDelta(event)` guard.

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
import Substructure, { isTokenDelta } from "@substructure.ai/sdk";

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

Token deltas only flow when the agent was defined with `stream: true`. They're transient: the engine does not persist them, and a client reconnecting mid-call will not see deltas already emitted — only the final `message.new` once the call completes.

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

The embedded instance exposes the same `startTurn` / `stream` / `turnResult` surface as the backend client, plus a `fetch` handler if you want to put an HTTP face on it. Use `db: ":memory:"` for a transient instance in tests.

## The LLM

`llm` is the LLM the loop calls. In the common case it's just `{ model: "provider/model" }` — the Substructure server makes the call against its configured provider:

```ts
agent({ name, decide: toolLoop({ llm: { model: "anthropic/claude-sonnet-4-6" }, /* ... */ }) });
```

Add `temperature`, `reasoning`, or `stream: true` alongside `model`. Server models are identified by `provider/model` strings; when running embedded or locally, provider credentials are read from the environment; with cloud, they're configured for your org in the dashboard. To make the call from your own worker instead (so your provider key never leaves your infrastructure), pass a provider generator (`anthropicGenerate`, `openaiGenerate`, `aiGenerate`) from `@substructure.ai/sdk/adapters/*` as `llm`. The AI and OpenAI adapters also ship `aiSdkAgent(settings)` and `openaiAgent(input)`, which return a `decide` — the default loop over an AI SDK toolset or an `@openai/agents` `Agent` — that you wrap with `agent({ name, decide: aiSdkAgent(...) })`.

## Examples

See [`examples/`](https://github.com/substructureai/substructure/tree/main/examples) for full deployments:

- `node-embedded` — in-process agent with persistent SQLite state.
- `cloudflare-worker` — worker deployed to Cloudflare; tools keep state in a Durable Object keyed by session.
- `hono` — fetch handler mounted on a Hono route in Node.
- `vercel` — serverless worker on Vercel.
- `sub-agent` — a parent agent delegating to a child via `subAgents`.
- `hybrid-state` — tool state in a per-user database, reached through `ctx`.
- `state-hydration` — state on the wire: a custom `decide` that runs a `toolLoop` against `worker_state`.
- `tool-approval` — a custom `decide` that gates real shell commands behind a `client.action` approval.
- `plan-mode` — a modal agent: a custom `decide` that switches model/prompt/tools by mode and forks a branch to execute.
- `deferred-tool` — async tool call: `execute` returns `ctx.defer()`, the result is posted later via `submitToolCallResult`.
- `frontend-tool` — chat UI where tools run in the browser (geolocation, theme). Also demonstrates `stream: true` — the assistant message renders token-by-token from `llm.token.delta` events.
