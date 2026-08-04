---
title: Quick start (cloud)
group: Getting started
---

## 1. Sign in

```sh
npm install -g @substructure.ai/cli
subs login
```

`login` authenticates in your browser and stores a token under
`~/.config/substructure`.

## 2. Describe the project

One `substructure.toml` is one project. Write it in your project root — this is
the whole declaration, and applying it is how the project comes into existence:

```toml title="substructure.toml"
name = "my-bot"

[llm.claude]
type = "anthropic"

[agent.my-agent]
llm = "claude"
model = "claude-sonnet-4-5"
```

```sh
subs apply
```

That creates the project and writes `[remote].project` back into the file,
so a second apply is a no-op rather than a second project. See
[Environments](./160-cli.md#environments).

A second environment is a second file: `subs apply -c substructure.staging.toml`
deploys a separate project with its own wallet, quota, and keys.

## 3. Give it a key

Calls run on your key, so upload one for the block the agent names:

```sh
subs llm set-key claude    # reads the key from stdin
```

The key never appears in argv, and no read ever returns it. Until one is set, a
call on that block fails saying so.

At this point `my-agent` already works: with no `worker`, the engine decides its
turns by accepting its own proposal. Skip to step 6 to send it a message.

## 4. Add a worker, if the agent needs your code

`worker` on an agent is the whole routing switch — set it, and the engine POSTs
that agent's decisions to your code instead of deciding them itself:

```toml title="substructure.toml"
[agent.triage]
worker = "https://my-worker.example.com/agent"
```

```sh
subs apply
```

The first apply that gives an agent a worker mints a signing secret for it. The
secret is the deployment's, not the file's, so read it back:

```sh
subs agents show triage
```

Set it as `SUBS_SIGNING_SECRET` where the worker runs.

## 5. Verify the signature in your worker

The engine signs every decision it POSTs. Your worker should verify that
signature before acting on one:

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

## 6. Send a message

Mint a client API key and submit on a user's behalf through the machine API.

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

The response returns the `session_id` and `turn_id`.

## 7. Watch it run

```sh
subs sessions list
subs sessions events <session-id> --stream
subs open
```

The reply, tool calls, and results stream as events. The whole session lives in the
cloud; resume it by passing the same `session_id` to another `submit`.

## Next

- [Cloud](./170-cloud.md): projects, agents, keys, and where provider keys live.
- [Authentication](./180-auth.md): client tokens for browsers, and worker signing.
- [Client API](./190-api.md): the machine and client surfaces this used.
- [Quick start](./10-quick-start.md): build the worker, add tools and a sub-agent.
