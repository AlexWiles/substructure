# Tutorial: An email-driven expense agent on Cloudflare

This tutorial walks through building an **AI email expense tracker** that
sits behind a Cloudflare Email Worker. Whitelisted senders email the
Worker, an agent built with [substructure.ai](../../../README.md) records
the expense in a SQLite Durable Object and replies.

```
Inbound mail ─┐
              ▼
  ┌──────────────────────────────┐
  │ email() handler              │  1. whitelist
  │  - parse with PostalMime     │  2. log email to database DO (RETURNING id)
  │  - insert email row          │  3. ctx.waitUntil(runAgent(...))
  │  - submit & forget           │  4. return
  └──────────────────────────────┘
                                       (agent loop continues in background)
   Substructure backend ◄──── decision loop ────► fetch() handler
                                                  - tool.execute "send_reply"
                                                  - dispatches via env.EMAIL.send()
```

## The agent

```ts
const emailAgent = agent({ id: "email-agent" })
  .use(agent.logging())
  .use(durableObjectState(() => workerEnv.AGENT_STATE))
  .use(agent.systemMessage(SYSTEM_PROMPT))
  .use(agent.messageHistory())
  .use(agent.tools([recordExpense, querySql]))
  .use(agent.llmLoop({
    request: { model: "anthropic/claude-sonnet-4-5" },
    llm_client: "openrouter",
  }));
```

Six `.use()` lines, each is a built-in middleware:

- **`logging`** — every decision (LLM call, tool call) prints with timing.
- **`durableObjectState`** — loads/saves agent state to a DO keyed by
  session id, so message history survives across the decision loop.
- **`systemMessage`** — prepends a fixed system prompt to every LLM call.
- **`messageHistory`** — records user/assistant/tool messages and injects
  them into every LLM call.
- **`tools`** — registers tool defs and handles tool execution.
- **`llmLoop`** — drives the call-LLM / call-tool / repeat loop until
  the model returns a final assistant message.

## The tools

Three tools. `record_expense` is the typed write path; `query_sql` is the
read/SQL escape hatch; `send_reply` is how the agent chooses to write back
to the sender.

```ts
const recordExpense = agent.tool({
  name: "record_expense",
  description: "Record one expense extracted from the current email.",
  parameters: {
    type: "object",
    properties: {
      email_id: { type: "number" },
      amount: { type: "number" },
      description: { type: "string" },
    },
    required: ["email_id", "amount", "description"],
  },
  execute: async (args) => {
    const { email_id, amount, description } = JSON.parse(args);
    await db().recordExpense({
      emailId: email_id,
      amountCents: Math.round(amount * 100),
      description,
    });
    return { ok: true, email_id, amount, description };
  },
});
```

The `email_id` is passed as a tool parameter rather than hidden in state.
The system prompt tells the LLM to copy it from the `email_id: N` line at
the top of the user message. The handler also writes two time headers:
`now_ms: <integer>` (Unix epoch milliseconds, matching the DB's
`received_at` / `created_at`) and `now: <ISO>` (the same instant in
human-readable form). LLMs have no concept of "now" on their own — the
millisecond value goes straight into SQL comparisons, the ISO string is
there for resolving phrases like "this week" or "yesterday".

One email can produce many expenses — call `record_expense` once per
line item.

`query_sql` runs arbitrary SQL through `db().query(sql, params)`. Its
description carries both table schemas so the model knows what to write:

```ts
const querySql = agent.tool({
  name: "query_sql",
  description:
    "Run any SQL statement against the database and return the rows.\n" +
    "Schema:\n" +
    "  emails(id, message_id UNIQUE, thread_id, in_reply_to, from_addr,\n" +
    "         to_addr, subject, body, received_at)\n" +
    "  expenses(id, email_id REFERENCES emails(id),\n" +
    "           amount_cents, description, created_at)\n" +
    "Amounts are integer cents. Timestamps are Unix epoch in milliseconds. " +
    "Use ? placeholders for parameters. Prefer record_expense for " +
    "inserting new expense rows.",
  parameters: {
    type: "object",
    properties: {
      sql:    { type: "string" },
      params: { type: "array", items: {} },
    },
    required: ["sql"],
  },
  execute: async (args) => {
    const { sql, params = [] } = JSON.parse(args);
    const rows = await db().query(sql, params);
    return { rows, count: rows.length };
  },
});
```

Aggregation queries (`SUM(amount_cents) GROUP BY ...`), filtered lookups
across both tables (`JOIN emails ON ...`), and one-off schema inspection
(`pragma_table_info('expenses')`) are all reachable without adding new
typed methods.

The third tool, `send_reply`, actually sends the reply itself — no
handler-side plumbing.

```ts
const sendReplyTool = agent.tool({
  name: "send_reply",
  description: "Send a plain-text reply to the sender of the given email.",
  parameters: {
    type: "object",
    properties: {
      email_id: { type: "number" },
      body: { type: "string" },
    },
    required: ["email_id", "body"],
  },
  execute: async (args) => {
    const { email_id, body } = JSON.parse(args);
    await dispatchReply(email_id, body);
    return { ok: true };
  },
});
```

`dispatchReply` looks up the email row, builds a MIME message with the
right `In-Reply-To` header for threading, and hands it off to the
[`send_email` binding](https://developers.cloudflare.com/email-routing/email-workers/send-email-workers/):

```ts
async function dispatchReply(emailId, body) {
  const email = await db().getEmail(emailId);
  const mime = createMimeMessage();
  if (email.message_id) mime.setHeader("In-Reply-To", `<${email.message_id}>`);
  mime.setSender({ name: "Inbox agent", addr: email.to_addr });
  mime.setRecipient(email.from_addr);
  mime.setSubject(`Re: ${email.subject}`);
  mime.addMessage({ contentType: "text/plain", data: body });
  await workerEnv.EMAIL.send(
    new EmailMessage(email.to_addr, email.from_addr, mime.asRaw()),
  );
}
```

That works because `env.EMAIL.send()` is callable from any Worker
invocation, including a `fetch()` invocation handling a tool execution.
The catch: the destination must be a **verified destination address** in
Email Routing. For this worker that's fine — the same addresses on the
whitelist need to be verified as destinations too.

If the agent never calls `send_reply` — say it just logged an expense
with nothing to say back — no reply goes out. The agent decides.

### A note on giving an LLM raw SQL

Exposing arbitrary SQL is convenient but powerful — the agent can in
principle `DROP TABLE`. The whitelist limits exposure to senders you
trust. If you want tighter sandboxing, narrow `db().query()` to
`SELECT`-only and add narrow typed methods (`listExpenses`, etc.) for the
read patterns you actually want.

## The handler

```ts
async email(message, env, ctx) {
  workerEnv = env;
  if (!WHITELIST.has(message.from.toLowerCase())) {
    message.setReject("Sender not allowed");
    return;
  }

  const parsed = await PostalMime.parse(message.raw);
  const subject = parsed.subject ?? "(no subject)";
  const body = (parsed.text ?? "").slice(0, 4000);

  const emailId = await db().recordEmail({
    messageId: message.headers.get("Message-ID"),
    threadId: threadRoot(message.headers),
    inReplyTo: inReplyToOf(message.headers),
    from: message.from,
    to: message.to,
    subject,
    body,
  });
  const now = Date.now();
  const userContent =
    `email_id: ${emailId}\n` +
    `now_ms: ${now}\n` +
    `now: ${new Date(now).toISOString()}\n` +
    `From: ${message.from}\nSubject: ${subject}\n\n${body}`;

  // Submit and forget. The agent's send_reply tool will dispatch any
  // reply via env.EMAIL.send() from a subsequent fetch() invocation.
  ctx.waitUntil(
    runAgent(env, sessionIdForThread(message.headers), message.from, userContent),
  );
}
```

`ctx.waitUntil()` keeps the Worker alive long enough for `runAgent` to
finish without blocking the email-handler return. The body of `runAgent`
shrinks to just draining the stream:

```ts
async function runAgent(env, sessionId, from, userContent) {
  const stream = client.submit({ ... });
  for await (const _ of stream) { /* drain */ }
  await stream.result;
}
```

The DO method does the upsert with `RETURNING id`:

```ts
async recordEmail(input: RecordEmailInput): Promise<number> {
  const row = this.ctx.storage.sql.exec<{ id: number }>(
    `INSERT INTO emails
       (message_id, thread_id, in_reply_to, from_addr, to_addr, subject, body)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(message_id) DO UPDATE SET message_id = excluded.message_id
     RETURNING id`,
    input.messageId, input.threadId, input.inReplyTo,
    input.from, input.to, input.subject, input.body,
  ).one();
  return row.id;
}
```

The `DO UPDATE` is a no-op write — its only purpose is to make `RETURNING`
fire on conflict. Without it, `ON CONFLICT DO NOTHING` returns zero rows
when the message already exists.

## Sessions are per-thread

```ts
sessionId: sessionIdForThread(message.headers)
```

Same email thread → same session → same `AGENT_STATE` DO → message
history accumulates across every reply. The agent sees a thread as one
ongoing conversation, remembers what was already discussed, and can
reference earlier expenses naturally. A sender starting a fresh subject
gets a fresh session.

Resolution order in `sessionIdForThread(headers)`:

1. First entry of `References` — thread root, present on every reply ≥2 deep.
2. `In-Reply-To` — present on direct replies.
3. `Message-ID` — new thread, this message is the root.
4. `crypto.randomUUID()` — pathological headerless mail.

## The database Durable Object

`Database` owns the SQLite storage and exposes three RPC methods —
`recordEmail`, `recordExpense` (both typed and idempotent), and `query`
(passes any SQL through to the storage API). Callers deal in typed inputs
for writes and rows for reads.

Two tables, linked by `email_id`, with natural unique keys on both:

```ts
ctx.storage.sql.exec(`
  CREATE TABLE IF NOT EXISTS emails (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  TEXT    UNIQUE,
    thread_id   TEXT,
    in_reply_to TEXT,
    from_addr   TEXT    NOT NULL,
    to_addr     TEXT    NOT NULL,
    subject     TEXT    NOT NULL,
    body        TEXT    NOT NULL,
    received_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
  )
`);
ctx.storage.sql.exec(`
  CREATE TABLE IF NOT EXISTS expenses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id     INTEGER NOT NULL REFERENCES emails(id),
    amount_cents INTEGER NOT NULL,
    description  TEXT    NOT NULL,
    created_at   INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
  )
`);
```

`emails.message_id UNIQUE` means a retried delivery (same Message-ID
arriving twice) collapses onto the existing row. `expenses` has no
uniqueness constraint — one email can carry multiple line items, even
with the same description (two coffees, two cab rides, etc.).

`thread_id` is the same value used for the session id — the first
`References` entry, falling back to `In-Reply-To`, falling back to
`Message-ID`. `in_reply_to` is the immediate parent. Together they let
you reconstruct any conversation: `SELECT * FROM emails WHERE thread_id =
? ORDER BY received_at` gives you the whole thread in order; following
`in_reply_to` chains gives you the reply tree.

## `wrangler.jsonc`

```jsonc
{
  "name": "email-agent-example",
  "main": "src/index.ts",
  "compatibility_date": "2026-03-24",
  "compatibility_flags": ["nodejs_compat"],
  "send_email": [{ "name": "EMAIL" }],
  "durable_objects": {
    "bindings": [
      { "name": "AGENT_STATE", "class_name": "AgentState" },
      { "name": "DATABASE",      "class_name": "Database" }
    ]
  },
  "migrations": [
    { "tag": "v1", "new_classes":        ["AgentState"] },
    { "tag": "v2", "new_sqlite_classes": ["Database"] }
  ]
}
```

## Running it

In one terminal:

```sh
export OPENROUTER_API_KEY=sk-or-...
substructure start --dev \
  --provider openrouter \
  --port 9000 \
  --worker-url http://localhost:8787
```

In another:

```sh
pnpm dev
echo 'SUBSTRUCTURE_URL=http://localhost:9000' >> .dev.vars
echo 'SUBSTRUCTURE_API_KEY=dev-worker-key'    >> .dev.vars
```

### Simulate an inbound message

```eml
From: alice@example.com
To: hello@yourdomain.com
Subject: Coffee + lunch
Message-ID: <test-1@example.com>

Spent $4.75 on coffee and $14.20 on lunch today. What's my total spend so far?
```

```sh
curl -X POST http://localhost:8787/cdn-cgi/handler/email \
  -H 'Content-Type: message/rfc822' \
  --data-binary @test.eml \
  -G --data-urlencode 'from=alice@example.com' \
  --data-urlencode 'to=hello@yourdomain.com'
```

Wrangler prints the path of the `.eml` file containing the reply. Send a
follow-up from `alice@example.com` with `References: <test-1@example.com>`
to continue the thread — the agent will recall its prior turns.

## Deploy

```sh
pnpm deploy
wrangler secret put SUBSTRUCTURE_URL
wrangler secret put SUBSTRUCTURE_API_KEY
wrangler secret put SIGNING_SECRET   # optional
```

Then in the Cloudflare dashboard: *Email Routing* → *Routes* → *Create
address* → action **Send to a Worker** → pick `email-agent-example`.

## Where to take it

- Move the whitelist into a Durable Object so it can change without a redeploy.
- Shard the database per sender (`DATABASE.idFromName(message.from)`).
- Add an OCR sub-agent for attached receipts via `agent.subAgents([...])`.
