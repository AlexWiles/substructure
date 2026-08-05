---
title: Cloud
group: Operations
---

Substructure cloud hosts the engine. A project receives client traffic. It sends
each decision to the worker its agent names, or decides itself. It makes the LLM
calls you gave it a key for. You manage the project with `subs`.

One `substructure.toml` is one project. A second environment is a second file,
deployed as a second project:

```sh
subs apply -c substructure.staging.toml
```

Each project has its own wallet, quota, keys, and sessions.

## Sign in

```sh
subs login
```

This authenticates in your browser and stores a token under
`~/.config/substructure`.

## Deploy a project

There is no create command. The file is the source of truth for the project: that
it exists, its name, and everything in it. You create a project by writing the
file and applying it:

```toml title="substructure.toml"
name = "my-bot"

[llm.claude]
type = "anthropic"

# The engine decides. It proposes, then accepts its own proposal.
[agent.support]
llm = "claude"
model = "claude-sonnet-4-5"

# Your code decides. The engine POSTs here and waits for the reply.
[agent.triage]
worker = "https://my-worker.example.com/agent"
```

```sh
subs apply
```

Apply creates the project, declares the agents, and writes the pin into
`[remote].project`. So a second apply changes nothing. It does not create a
second project. To use a project that already exists, from a fresh clone or a
teammate's machine, use `subs link`.

Apply **replaces**. It does not merge. The file is the whole declaration, so an
agent, an `[llm.*]` block, or a Slack channel that is not in the file is one you
removed.

## Each agent chooses its own host

`worker` on an agent selects who decides:

- **Set** — the engine POSTs decisions there. It signs each one with a secret it
  created for that agent. The signature is an HMAC of the body, sent as
  `X-Substructure-Signature`. Your worker checks it with the same secret.
- **Unset** — the engine decides for that agent by accepting its own proposal.
  You deploy nothing but the file.
- **Not declared** — a decision for an agent that no file declares fails
  immediately. The error lists the ids that are declared.

The secret belongs to the deployment, not to the file, so it is never written in
`substructure.toml`:

```sh
subs agents list
subs agents show triage          # includes the signing secret
subs agents rotate-secret triage # owner only
```

## Secrets

| Secret | Purpose | Where it comes from |
| --- | --- | --- |
| Signing secret | Your worker uses it to verify the engine's decision requests. | The deployment creates one for each agent, on the first apply that gives it a `worker`. Read it with `subs agents show <id>`. |
| Client API key | The bearer token your clients send to the project. | `subs keys create <label>`. It is printed once. |
| Provider key | Authenticates to Anthropic, OpenAI, or OpenRouter. | `subs llm set-key <block>`. See below. |
| Slack bot token | The bot reads and posts as your app. | `subs slack connect` installs the app into a workspace. The token stays in the deployment. |

`subs keys create <label>` creates a client key and writes only the value to
stdout, so you can pipe it into your client's secret store.

### Provider keys

Calls run on your key. A block the engine calls — `anthropic`, `openai`, or
`openrouter` — needs one:

```sh
subs llm set-key claude              # reads the key from stdin
subs llm set-key claude --env MY_KEY # or from an environment variable
subs llm list
```

The key never appears in the command line, and no read returns it. Until you set
one, every call on that block fails with an error that says so. On a `type =
"worker"` block, the call runs in your worker, which calls the provider with a
key from its own environment. There is nothing to upload. See
[LLMs](./50-llms.md).

`api_key_env` in the file names a variable on *your* machine, so it applies only
to `subs serve` and `subs run`. `subs apply` removes it, and a deployment that
receives one refuses the document. It does not ignore the field.

## Observe

```sh
subs projects list
subs sessions list
subs sessions events <session-id> --stream
subs config log
subs open
```

## Next

- [Quick start](./15-quick-start-cloud.md): build a project and run a turn.
- [CLI](./160-cli.md): the full command reference.
- [LLMs](./50-llms.md): where provider keys live.
- [Protocol](./150-protocol.md): the signed request the engine sends.
