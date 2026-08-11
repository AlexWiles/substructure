---
title: Cloud
group: Running it
---

Substructure cloud hosts the engine. A project receives client traffic, decides
each turn, and makes the model calls you gave it a key for.

You manage a project with `subs`. There is no dashboard step you cannot do from
the terminal.

## One file is one project

`substructure.toml` is the whole declaration: that the project exists, its name,
and everything in it. You create a project by applying the file.

```toml title="substructure.toml"
name = "my-bot"

[llm.claude]
type = "anthropic"

# The engine decides.
[agent.support]
llm = "claude"
model = "claude-sonnet-4-5"

# Your code decides.
[agent.triage]
worker = "https://my-worker.example.com/agent"
```

```sh
subs apply
```

Apply creates the project and writes the pin into `[remote].project`. A second
apply changes nothing.

A second environment is a second file.

```sh
subs apply -c substructure.staging.toml
```

Each project has its own wallet, quota, keys, and sessions.

To use a project that already exists, from a fresh clone or a teammate's
machine, run `subs link`.

## Apply replaces

Apply does not merge. An agent, an LLM block, a connection, or a Slack channel
that is not in the file is one you removed.

Apply is idempotent. An unchanged file writes nothing and exits 0. It is safe to
run on every merge.

## Provider keys

Calls run on your key. A block the engine calls needs one.

```sh
subs llm set-key claude              # reads the key from stdin
subs llm set-key claude --env MY_KEY # or from an environment variable
subs llm list
```

The key never appears in the command line. No read returns it. Until you set
one, every call on that block fails with an error that says so.

A `type = "worker"` block needs no key here. The call runs in your worker. See
[LLMs](./70-llms.md).

`api_key_env` names a variable on your machine, so it applies to `subs serve`
and `subs run`. `subs apply` removes it. A deployment refuses a document that
carries one.

## Secrets

| Secret | Purpose | Where it comes from |
| --- | --- | --- |
| Signing secret | Your worker verifies the engine's decision requests with it. | The deployment creates one per agent, on the first apply that gives it a `worker`. Read it with `subs agents secret <id>`. |
| Client API key | The bearer token your clients send. | `subs keys create --label <label>`. Printed once. |
| Provider key | Authenticates to Anthropic, OpenAI, or OpenRouter. | `subs llm set-key <block>`. |
| Slack bot token | The bot reads and posts as your app. | `subs slack connect`. The token stays in the deployment. |

The signing secret belongs to the deployment. It is never written in the file.
Any member of the organization can read or rotate it: whoever can deploy the
project can run its worker.

Printing one is its own command, so no other output carries it.

```sh
subs agents list
subs agents secret triage        # the secret, on stdout
subs agents rotate-secret triage # the old secret stops working at once
```

## Send a message from your backend

Create an API key and submit for a user through the machine API.

```sh
export SUBS_API_KEY=$(subs keys create --label backend)
export BASE=https://api.substructure.ai

curl $BASE/api/machine/sessions/submit \
    -H "Authorization: Bearer $SUBS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "agent_id": "support",
      "identity": { "id": "user_42" },
      "payload": { "type": "client.message", "message": { "role": "user", "content": "hi" } }
    }'
```

The response holds the `session_id` and the `turn_id`. Pass the same
`session_id` to another submit to continue the conversation.

For a browser, mint a short-lived client token instead. See
[Authentication](./190-auth.md).

## Run a turn against it

```sh
subs run "hi"
```

With a `[remote]`, `subs run` sends the turn to the deployment and streams it
back. It uses the credential `subs login` stored.

## Watch it run

```sh
subs sessions list
subs sessions events <session-id> --stream
subs config log
subs open
```

## Next

- [Config](./220-config.md): every key `subs apply` sends.
- [Authentication](./190-auth.md): client tokens and worker signing.
- [REST API](./250-api.md): the endpoints your clients call.
- [Self-hosting](./180-self-hosting.md): run the same engine yourself.
