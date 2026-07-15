---
title: Quick start (cloud)
group: Getting started
---

The [local quick start](./10-quick-start.md) runs the engine on your machine.

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

## 2. Verify the signature in your worker

The hosted engine signs every decision it POSTs with the app's signing secret. Add
a check to the `server.mjs` worker from the local quick start, then set the secret
in its environment:

```javascript title="server.mjs"
import { createHmac, timingSafeEqual } from "node:crypto";

const SECRET = process.env.SUBS_SIGNING_SECRET;

function verify(body, header) {
    const expected = "sha256=" + createHmac("sha256", SECRET).update(body).digest("hex");
    const a = Buffer.from(expected);
    const b = Buffer.from(header ?? "");
    return a.length === b.length && timingSafeEqual(a, b);
}

const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        if (!verify(body, req.headers["x-substructure-signature"])) {
            res.writeHead(401).end();
            return;
        }
        const decision = decide(JSON.parse(body));
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(decision ?? null));
    });
});

server.listen(4444, () => console.log("worker listening on http://localhost:4444"));
```

The provider key stays out of this. By default the hosted engine runs the LLM
against its own provider and bills the app, so your worker sets no `ANTHROPIC_API_KEY`.

## 3. Expose the worker

The engine reaches your worker over the public internet, so it needs a URL. Deploy
it anywhere, or for local development open a tunnel:

```sh
SUBS_SIGNING_SECRET=$(subs webhook secret) node server.mjs
# in another terminal, tunnel port 4444 and copy the https URL
```

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
