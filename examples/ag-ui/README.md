# ag-ui

A [TanStack Start](https://tanstack.com/start) app deployed to **Cloudflare
Workers**, with a [TanStack AI](https://tanstack.com/ai) chat that talks to
substructure's **native AG-UI endpoint** directly — no runtime broker, no
translation layer.

TanStack AI's client speaks AG-UI natively: `fetchServerSentEvents` POSTs a
`RunAgentInput` straight to the engine's `/api/client/ag-ui/agents/{agentId}/run`
and parses the AG-UI SSE stream. The only thing the edge app does is host the
worker and mint a token, so the Worker stays tiny.

## Architecture

```
browser <TanStack AI useChat + fetchServerSentEvents>
   │  1. route loader → server fn mints a short-lived token (server-side)
   │  2. AG-UI run, streamed   (cross-origin, DIRECT to the engine)
   ▼
<engine>/api/client/ag-ui/agents/weather-agent/run
   │  drives the agent loop
   ▼
POST /api/agent   (the substructure worker — also this Worker)
```

- **`routes/api/agent.ts`** — the substructure worker. The engine calls it
  server-to-server.
- **`routes/index.tsx`** — a `createServerFn` mints the short-lived,
  identity-locked client token in the route **loader** (the API key stays on the
  server) and passes it to the chat.
- **`components/chat.tsx`** — `useChat({ connection: fetchServerSentEvents(...) })`
  pointed at the engine endpoint with the token as a Bearer header.

Because the browser streams from the engine **directly**, the engine must be
reachable from the browser with permissive CORS (the local/dev server already
is). Set `SUBSTRUCTURE_PUBLIC_URL` to the browser-facing engine URL when it
differs from the server-side one.

## Run locally

In one terminal, a local engine pointed at this app's worker route:

```sh
export OPENROUTER_API_KEY=sk-or-...
substructure start --dev --port 9000 --worker-url http://localhost:3030/api/agent
```

In another, the TanStack Start app (Node dev server):

```sh
pnpm install
pnpm dev          # http://localhost:3030
```

Open <http://localhost:3030> and ask *“What's the weather in SF and Tokyo?”* —
the reply streams in and the `get_weather` tool calls render inline.

## Frontend (client-handled) tools

Ask *“What timezone am I in?”* to exercise a **frontend tool**. `get_user_timezone`
runs in the **browser**, not on the server:

- The worker only *declares* it — `handler: "client"` + `ctx.defer()`
  (`substructure.ts`). The engine suspends the turn and waits.
- The matching executor is a TanStack AI client tool
  (`toolDefinition(...).client(execute)` in `src/components/tools.ts`, passed to
  `useChat({ tools })`). TanStack AI runs it and continues the run with the result.
- The AG-UI endpoint streams the call as `TOOL_CALL_START/ARGS/END` → `RUN_FINISHED`,
  then on the continuation run maps the tool-result message to
  `submit_tool_call_result`, resuming the turn → `TOOL_CALL_RESULT` → reply.

The tool **name** must match on both sides (`get_user_timezone`). The browser
never sees the substructure API key; only the short-lived client token.

## Deploy to Cloudflare Workers

```sh
# set the engine URLs in wrangler.jsonc `vars`, then the secrets:
wrangler secret put SUBSTRUCTURE_API_KEY
wrangler secret put SIGNING_SECRET        # only if the engine signs webhooks

pnpm deploy                                # build:cf + wrangler deploy
```

Point your engine's webhook at `https://<your-worker>/api/agent`. The build
uses `@cloudflare/vite-plugin`; `nodejs_compat` is enabled and also populates
`process.env` from the vars/secrets above.

## Why this shape

[TanStack AI](https://tanstack.com/ai) is a type-safe, provider-agnostic SDK
that's [fully AG-UI compliant](https://tanstack.com/blog/ag-ui-compliance) — its
client can hit any AG-UI server, so it speaks straight to substructure's endpoint
with nothing in between. It's headless (you render `messages[].parts` yourself),
client tools are first-class (`.client(execute)`, plus `needsApproval` for HITL),
and it lives in the same TanStack ecosystem as the rest of the app.
