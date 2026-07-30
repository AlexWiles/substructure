# node-hono-copilotkit

A chat agent with a web frontend built on [CopilotKit](https://www.copilotkit.ai),
served with [Hono](https://hono.dev). The browser talks to substructure's native
**AG-UI** endpoint directly — CopilotKit's
[direct-to-agent](https://docs.copilotkit.ai/backend/ag-ui) mode — with no CopilotKit
runtime in between.

`server.ts` does three things: it's the agent worker (the engine calls it), it mints a
short-lived browser token, and it serves the built UI. The browser (`web/`) renders
CopilotKit's `<CopilotChat>` pointed at the engine via an `HttpAgent`.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Three things: the engine, the app, the browser.

**1. Start the engine** — it holds the LLM key and drives the loop:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs serve -c substructure.toml
```

**2. Build and start the app**:

```sh
npm install
npm run build
npm start          # http://localhost:4444
```

**3. Open** <http://localhost:4444> and ask *"what time is it in my time zone?"* — the reply
streams in and the browser tool renders inline as a card.

## Tools

Two tools, one of each kind:

- **`get_current_time`** runs on the **worker** (server-side) and is declared in the worker.
- **`get_timezone`** runs in the **browser** (client-side) and is declared in the client via
  `useFrontendTool`.

CopilotKit forwards every `useFrontendTool` up in the AG-UI run's `tools`, and the engine layers
them onto the proposed agent by default — so `get_timezone` never has to be defined in the worker.
To restrict which client tools are honored, `decide()` can handle `client.messages` and return its
own `agent` config with a filtered `trigger.client.tools` to override the default proposal.

## How it works

- **`server.ts`** — `POST /agent` is the worker: it declares the agent and the worker tool, and
  runs it when the model calls it. `GET /token` mints a client token. `/*` serves `web/dist`.
- **`web/chat.tsx`** — `new HttpAgent({ url, headers })` points at
  `…/api/channels/ag-ui/agents/chat-agent/run`; `<CopilotKit agents__unsafe_dev_only={{ … }}>`
  connects the browser straight to it (no runtime), and `<CopilotChat agentId="chat-agent">`
  renders the thread. The frontend tool is a `useFrontendTool` entry with a `render`.

## Regenerate types

`protocol.ts` is generated from `schemas/protocol.schema.json` and committed. To regenerate
after a protocol change:

```sh
npx quicktype --src-lang schema --lang typescript \
    --src ../../schemas/protocol.schema.json \
    --top-level Protocol --just-types --prefer-unions -o protocol.ts
npx @biomejs/biome format --write protocol.ts
```
