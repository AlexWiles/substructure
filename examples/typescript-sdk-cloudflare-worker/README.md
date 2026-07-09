# cloudflare-worker

A Cloudflare Worker that serves the agent over HTTP, with per-session
agent state in a Durable Object. The DO is keyed by session id, so each
session gets its own isolated, persistent state at the edge.

Use this shape when you want the agent at the Cloudflare edge with
state pinned to the Worker (no separate state service required by the
Substructure backend).

## Deploy

1. Log in to Substructure.
   ```sh
   subs login
   ```

2. Link this directory to your org/app (writes `substructure.toml`).
   ```sh
   subs link
   ```

3. Deploy the worker and copy its URL.
   ```sh
   wrangler deploy
   ```

4. Point the app at the worker URL (this also enables delivery).
   ```sh
   subs webhook set https://<your-worker>.workers.dev
   ```

5. Pipe the signing secret into the worker env — the secret never
   touches your terminal or shell history.
   ```sh
   subs webhook secret | wrangler secret put SIGNING_SECRET
   ```

## Trigger a turn

Mint an API key and pipe it straight into the client process:

```sh
export SUBSTRUCTURE_API_KEY=$(subs keys create local-dev)
tsx client.ts
```

`client.ts` submits a turn against the hosted backend using that key.

## Useful commands

```sh
subs webhook show          # endpoint + state
subs webhook secret        # print signing secret to stdout
subs webhook rotate-secret # rotate and print new signing secret
subs webhook disable       # pause delivery (keeps URL)
subs sessions list         # recent sessions
subs keys list             # active API keys
```

All secret-emitting commands (`keys create`, `webhook secret`,
`webhook rotate-secret`) print the raw value to stdout and human
messages to stderr, so they pipe cleanly into `wrangler secret put`,
`op`, `doppler secrets set`, etc.
