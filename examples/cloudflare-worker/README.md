# cloudflare-worker

A Cloudflare Worker that serves the agent over HTTP, with per-session
agent state in a Durable Object. The DO is keyed by session id, so each
session gets its own isolated, persistent state at the edge.

Use this shape when you want the agent at the Cloudflare edge with
state pinned to the Worker (no separate state service required by the
Substructure backend).

## Deploy

1. Deploy the cloudflare worker and copy its URL.
   ```sh
   pnpm deploy
   ```
2. In the substructure.ai dashboard, create an app.
   Set the worker URL to the copied worker URL.
   Copy the signing secret.
3. Put the signing secret in the cloudflare worker env.
   ```
   wrangler secret put SIGNING_SECRET
   ```
3. Enable the worker in the substructure dashboard.

## Trigger a turn

Create an API key in the substructure dashboard.

`client.ts` submits a turn against the hosted backend:

```sh
export SUBSTRUCTURE_API_KEY=...
pnpm client
```
