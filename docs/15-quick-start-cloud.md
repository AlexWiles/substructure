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

One `substructure.toml` is one project. Write it in your project root. This file
is the whole declaration, and you create the project by applying it:

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

That creates the project and writes `[remote].project` into the file. So a second
apply changes nothing. It does not create a second project. See
[Environments](./160-cli.md#environments).

A second environment is a second file. `subs apply -c substructure.staging.toml`
deploys a separate project with its own wallet, quota, and keys.

## 3. Give it a key

Calls run on your key, so upload one for the block the agent names:

```sh
subs llm set-key claude    # reads the key from stdin
```

The key never appears in the command line, and no read returns it. Until you set
one, every call on that block fails with an error that says so.

Every command that finishes a step prints what is left, and `subs doctor` shows
the same list at any time. So you set up the project by following one command to
the next.

`my-agent` already works. It has no `worker`, so the engine decides its turns by
accepting its own proposal. Go to step 6 to send it a message.

## 4. Add a worker, if the agent needs your code

`worker` on an agent selects who decides. Set it, and the engine POSTs that
agent's decisions to your code instead of deciding them itself:

```toml title="substructure.toml"
[agent.triage]
worker = "https://my-worker.example.com/agent"
```

```sh
subs apply
```

The first apply that gives an agent a worker creates a signing secret for it. The
secret belongs to the deployment, not to the file, so read it back:

```sh
subs agents show triage
```

Set it as `SUBS_SIGNING_SECRET` where the worker runs.

## 5. Verify the signature in your worker

The engine signs every decision it POSTs. Verify the signature before your worker
acts on a decision:

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

Then refuse unsigned requests in the server's handler, before it calls `decide`:

```javascript title="server.mjs"
if (!verify(body, req.headers["x-substructure-signature"])) {
    res.writeHead(401).end();
    return;
}
```

## 6. Send a message

Create a client API key and submit for a user through the machine API.

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

The response holds the `session_id` and the `turn_id`.

## 7. Watch it run

```sh
subs sessions list
subs sessions events <session-id> --stream
subs open
```

The reply, the tool calls, and the results stream as events. The whole session is
in the cloud. To continue it, pass the same `session_id` to another `submit`.

## Next

- [Cloud](./170-cloud.md): projects, agents, keys, and where provider keys live.
- [Authentication](./180-auth.md): client tokens for browsers, and worker signing.
- [Client API](./190-api.md): the machine and client APIs this used.
- [Quick start](./10-quick-start.md): build the worker, add tools and a sub-agent.
