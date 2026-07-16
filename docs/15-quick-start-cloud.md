---
title: Quick start (cloud)
group: Getting started
---

## 1. Sign in and create an app

```sh
npm install -g @substructure.ai/cli
subs login
```

`login` authenticates in your browser and stores a token under
`~/.config/substructure`. Create an app:

```sh
subs apps create my-bot
```

In your project root.

This prints the app id and its signing secret, shown once — save the secret now.
Pin the app so later commands need no `--app`:

```sh
subs link
```

## 2. Make sure to verify the signature in your worker

The hosted engine signs every decision it POSTs with the app's signing secret. Your
worker should verify that signature before acting on a decision. Add the check:

```javascript title="server.mjs"
import { createHmac, timingSafeEqual } from "node:crypto";

const SECRET = process.env.SUBS_SIGNING_SECRET;

function verify(body, header) {
    const expected = "sha256=" + createHmac("sha256", SECRET).update(body).digest("hex");
    const a = Buffer.from(expected);
    const b = Buffer.from(header ?? "");
    return a.length === b.length && timingSafeEqual(a, b);
}
```

Then reject unsigned requests in the server's handler, before calling `decide`:

```javascript title="server.mjs"
if (!verify(body, req.headers["x-substructure-signature"])) {
    res.writeHead(401).end();
    return;
}
```

The provider key stays out of this. By default the hosted engine runs the LLM
against its own provider and bills the app, so your worker sets no `ANTHROPIC_API_KEY`.

## 3. Deploy the worker

The engine reaches your worker over the public internet, so deploy it wherever you
run Node and set the signing secret in its environment:

```sh
subs webhook secret   # prints the secret; set it as SUBS_SIGNING_SECRET where the worker runs
```

The deploy gives the worker a public HTTPS URL to point the app at.

## 4. Point the app at it

```sh
subs webhook set https://my-worker.example.com/
```

The engine now POSTs each decision to that URL, signed with the app's secret.

## 5. Send a message

Mint an app API key and submit on a user's behalf through the machine API.

```sh
export SUBS_API_KEY=$(subs keys create --label quickstart)
export BASE=https://api.substructure.ai

curl $BASE/api/machine/sessions/submit \
    -H "Authorization: Bearer $SUBS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "agent_id": "my-agent",
      "identity": { "id": "user_42" },
      "payload": { "type": "client.message", "message": { "role": "user", "content": "hi" } }
    }'
```

The response returns the `session_id` and `turn_id`. The engine delivered the
decision to your worker, ran the loop, and called the model itself.

## 6. Watch it run

```sh
subs sessions list
subs sessions events <session-id> --stream
subs open
```

The reply, tool calls, and results stream as events. The whole session lives in the
cloud; resume it by passing the same `session_id` to another `submit`.

## Next

- [Cloud](./170-cloud.md): apps, keys, webhooks, and where provider keys live.
- [Authentication](./180-auth.md): client tokens for browsers, and worker signing.
- [Client API](./190-api.md): the machine and client surfaces this used.
- [Quick start](./10-quick-start.md): build the worker, add tools and a sub-agent.
