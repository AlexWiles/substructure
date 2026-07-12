# node-hono-assistant-ui

A chat agent with a web frontend built on [assistant-ui](https://www.assistant-ui.com),
served with [Hono](https://hono.dev). The browser talks to substructure's native
**AG-UI** endpoint through assistant-ui's
[AG-UI runtime](https://www.assistant-ui.com/docs/runtimes/ag-ui/overview) — no
translation layer in between.

`server.ts` does three things: it's the agent worker (the engine calls it), it mints a
short-lived browser token, and it serves the built UI. The browser (`web/`) renders a
pre-built `<Thread>` wired to the engine with `useAgUiRuntime`.

## Run

Three things: the engine, the app, the browser.

**1. Start the engine** — it holds the LLM key and drives the loop:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs serve --dev --port 9000 --provider anthropic --worker-url http://localhost:4444/agent
```

**2. Build and start the app**:

```sh
npm install
npm run build
npm start          # http://localhost:4444
```

**3. Open** <http://localhost:4444> and ask *"what time is it in my time zone?"* — the reply
streams in and both tool calls render inline as cards.

## Tools

Two tools, one of each kind:

- **`get_current_time`** runs on the **worker** (server-side): the engine sends a
  `tool.execute`, `decide()` runs it, returns the result.
- **`get_timezone`** runs in the **browser** (client-side): the worker only declares it
  (`handler: "client"`, no `exec`); the engine suspends the turn while assistant-ui runs the
  matching toolkit tool in `web/chat.tsx`, then resumes with its result. Only the
  browser knows the user's time zone, so this has to run there.

`ToolFallback` in `web/chat.tsx` renders a generic card for every tool call, whichever side
runs it.

## How it works

- **`server.ts`** — `POST /agent` is the worker: it declares the agent and both tools, and
  runs the worker tool when the model calls it. `GET /token` mints a client token. `/*`
  serves `web/dist`.
- **`web/chat.tsx`** — `new HttpAgent({ url, headers })` points at
  `…/api/client/ag-ui/agents/chat-agent/run`; `useAgUiRuntime({ agent })` turns the AG-UI
  SSE stream into an assistant-ui runtime; `<Thread>` renders it. The frontend tool is a
  `defineToolkit` entry registered with `useAui({ tools: Tools({ toolkit }) })`.

The browser streams from the engine **directly**, so the engine must be reachable from the
browser (dev CORS is already open). The API key never leaves the server — only the
short-lived token does. Set `SUBSTRUCTURE_PUBLIC_URL` when the browser-facing engine URL
differs from the server-side one.

## Regenerate types

`protocol.ts` is generated from `schemas/protocol.schema.json` and committed. To regenerate
after a protocol change:

```sh
npx quicktype --src-lang schema --lang typescript \
    --src ../../schemas/protocol.schema.json \
    --top-level Protocol --just-types --prefer-unions -o protocol.ts
npx @biomejs/biome format --write protocol.ts
```
