---
title: Cloud
group: Operations
---

Substructure cloud hosts the engine. A project takes client traffic, delivers
each decision to the worker that agent names — or decides it itself — and runs
the LLM calls it is given a key for. `subs` is the control plane for that
project.

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

Authenticates in your browser and stores a token under `~/.config/substructure`.

## Deploy a project

There is no create command. The file is the source of truth for a project's
existence, its name, and everything in it — writing one and applying it is how
a project is born:

```toml title="substructure.toml"
name = "my-bot"

[llm.claude]
type = "anthropic"

# Decided by the engine: it proposes, and accepts its own proposal.
[agent.support]
llm = "claude"
model = "claude-sonnet-4-5"

# Decided by your code: the engine POSTs here and waits for the reply.
[agent.triage]
worker = "https://my-worker.example.com/agent"
```

```sh
subs apply
```

Apply creates the project, declares the agents, and writes the pin back into
`[deployment].project`, so a second apply is a no-op rather than a second
project. To adopt a project that already exists — a fresh clone, a teammate's
machine — use `subs link` instead.

Apply **replaces** rather than merges: the file is the whole declaration, so an
agent, `[llm.*]` block, or Slack channel absent from it is one that was removed.

## Hosting is per agent

`worker` on an agent is the whole routing switch:

- **Set** — decisions are POSTed there, signed with a secret the deployment
  minted for that agent (an HMAC of the body, sent as
  `X-Substructure-Signature`). Your worker verifies it with the same secret.
- **Unset** — the engine decides for that agent by accepting its own proposal.
  Nothing to deploy beyond the file.
- **Undeclared** — a decision for an agent no file declares fails immediately,
  naming the ids that were declared.

The secret is the deployment's, not the file's, so it is never written in
`substructure.toml`:

```sh
subs agents list
subs agents show triage          # includes the signing secret
subs agents rotate-secret triage # owner only
```

## Secrets

| Secret | Purpose | Where it comes from |
| --- | --- | --- |
| Signing secret | Your worker verifies the engine's decision requests. | Minted per agent on the first apply that gives it a `worker`. Read it with `subs agents show <id>`. |
| Client API key | The bearer token your clients present to the project. | `subs keys create <label>`, printed once. |
| Provider key | Auth to Anthropic, OpenAI, or OpenRouter. | `subs llm set-key <block>` — see below. |

`subs keys create <label>` mints a client key and writes only the value to
stdout, so you can pipe it straight into your client's secret store.

### Provider keys

Calls run on your key. A block the engine runs (`anthropic`, `openai`,
`openrouter`) needs one uploaded:

```sh
subs llm set-key claude              # reads the key from stdin
subs llm set-key claude --env MY_KEY # or from an environment variable
subs llm list
```

The key never appears in argv, and no read ever returns it. Until one is set,
a call on that block fails saying so. On a `type = "worker"` block the call runs
in your worker instead, which calls the provider with a key from its own
environment — there is nothing to upload. See [LLMs](./50-llms.md).

`api_key_env` in the file names a variable on *your* machine, so it applies to
`subs serve` and `subs run` only; `subs apply` strips it, and a deployment
that receives one rejects the document rather than ignoring the field.

## Observe

```sh
subs projects list
subs sessions list
subs sessions events <session-id> --stream
subs config log
subs open
```

## Next

- [Quick start](./15-quick-start-cloud.md): a project from nothing to a turn.
- [CLI](./160-cli.md): the full command reference.
- [LLMs](./50-llms.md): where provider keys live.
- [Protocol](./150-protocol.md): the signed request the engine delivers.
