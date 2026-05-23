# cloudflare-worker

A Cloudflare Worker that serves the agent over HTTP, with per-session
agent state in a Durable Object. The DO is keyed by session id, so each
session gets its own isolated, persistent state at the edge.

Use this shape when you want the agent at the Cloudflare edge with
state pinned to the Worker (no separate state service required by the
Substructure backend).

## Deploy

1. Deploy the worker and copy its URL.
   ```sh
   pnpm deploy
   ```

2. Log in to Substructure.
   ```sh
   subs cloud login
   ```

3. Find your org id.
   ```sh
   subs cloud orgs list
   ```

4. Create an app. The printed `signing_secret` can be re-read later with
   `subs cloud apps webhook show`, or rotated with `webhook rotate-secret`.
   ```sh
   subs cloud apps create my-worker --org <ORG_ID>
   ```

5. Pin this directory to the org + app so subsequent commands don't need
   `--org` / `--app`.
   ```sh
   subs cloud init --org <ORG_ID> --app <APP_ID>
   ```

6. Put the signing secret in the worker env.
   ```sh
   wrangler secret put SIGNING_SECRET
   ```

7. Point the app at the worker URL (this also enables delivery).
   ```sh
   subs cloud apps webhook set https://<your-worker>.workers.dev
   ```

## Trigger a turn

Mint an API key for the app:

```sh
subs cloud apps keys create local-dev
```

`client.ts` submits a turn against the hosted backend:

```sh
export SUBSTRUCTURE_API_KEY=<api_key from previous step>
pnpm client
```

## Useful commands

```sh
subs cloud apps webhook show          # current endpoint + signing secret + state
subs cloud apps webhook rotate-secret # rotate the signing secret
subs cloud apps webhook disable       # pause delivery (keeps URL)
subs cloud apps sessions              # recent debug sessions
subs cloud apps keys list             # active API keys
```
