# assistant-ui-cloudflare-starter

A [TanStack Start](https://tanstack.com/start) app (deployable to **Cloudflare
Workers**) with an [assistant-ui](https://www.assistant-ui.com/) chat that streams
from a Substructure agent over its **AG-UI endpoint**. Includes multi-thread
support, starter suggestions, and a server-side (`get_current_time`) plus
client-side (`browser_alert`) tool.

## Run locally

In one terminal, a local engine pointed at this app's worker route:

```sh
export OPENROUTER_API_KEY=sk-or-...
subs start --dev --port 9000 --worker-url http://localhost:3030/api/agent
```

In another, the app:

```sh
npm install
npm run dev          # http://localhost:3030
```

Open <http://localhost:3030> and chat. Ask *"what time is it?"* to run the
server-side tool, or *"show me a browser alert"* to run the client-side one.

## Deploy to Cloudflare Workers

```sh
npm run deploy       # build:cf + wrangler deploy
```

Set the engine URLs in `wrangler.jsonc` and the `SUBSTRUCTURE_API_KEY` /
`SIGNING_SECRET` secrets first, and point your engine's webhook at
`https://<your-worker>/api/agent`.
