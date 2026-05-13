# Email expense agent

A Cloudflare Email Worker that hands inbound mail to a
[substructure.ai](../../../README.md) agent. The agent extracts expenses
into a SQLite Durable Object, answers questions, and replies via Email
Routing.

Whitelisted senders only (see `EMAIL_WHITELIST` in `src/index.ts`).

## Tools

- `record_expense` — typed insert into `expenses`.
- `query_sql` — arbitrary SQL against `emails` + `expenses`.
- `send_reply` — sends a threaded reply via `env.EMAIL.send()`.

Same email thread → same session → message history accumulates across
replies.

## Deploy

1. In the substructure.ai dashboard, create an app.
2. Deploy the Worker:
   ```sh
   pnpm deploy
   ```
   Copy the deployed Worker URL.
3. Back in the Substructure dashboard, paste the Worker URL into the
   app's settings.
4. Copy the app's **API key** and **signing secret** from the dashboard,
   then set them on the Worker:
   ```sh
   wrangler secret put SUBSTRUCTURE_API_KEY
   wrangler secret put SIGNING_SECRET
   ```
5. In the Cloudflare dashboard: *Email Routing → Routes → Create address
   → Send to a Worker → `email-agent-example`*.
