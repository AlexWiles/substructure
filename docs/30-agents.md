---
title: Agents
group: Building agents
---

An agent is a model, a prompt, and a set of tools. You declare one with an
`[agent.<id>]` section.

```toml title="subs.toml"
[llm.openrouter]
type = "openrouter"

[agent.oncall]
llm = "openrouter"
model = "anthropic/claude-sonnet-4-5"
system = "You are the on-call assistant."
```

The ID is how everything else names the agent. Slack routes to it. A client
submits to it. A parent agent calls it.

## Agent keys

| Key | Meaning |
| --- | --- |
| `llm` | Which `[llm.<id>]` block the model call runs on. |
| `model` | The model to call. |
| `system` | The system prompt. |
| `effort` | How hard the model thinks. |
| `description` | What this agent does, for a parent that calls it. |
| `mcp` | Connections whose tools the model can call. See [Connectors](./40-connectors.md). |
| `plugins` | Plugins whose skills and servers this agent gets. See [Plugins](./45-plugins.md). |
| `subagents` | Other agents this one can call. See [Subagents](./80-subagents.md). |
| `tools` | Tools the browser runs. See [Client-side tools](./150-client-tools.md). |
| `worker` | Where decisions go. See [Workers](./50-workers.md). |
| `retry` | Timeouts and attempts. See [Retries](./210-retries.md). |
| `signing_secret_env` | The variable holding the signing secret, for an engine you run. |

[Config](./220-config.md) has the full reference, including the defaults and the
table forms of `mcp` and `plugins`.

## Declare more than one agent

A second agent is a second section. Each one names its own model.

```toml title="subs.toml"
[llm.claude]
type = "anthropic"

[llm.cheap]
type = "openai"

[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
system = "Delegate research to the researcher, then answer."
subagents = ["agent.researcher"]

[agent.researcher]
description = "Finds and reads sources."
llm = "cheap"
model = "gpt-5-mini"
system = "Answer with citations. Be brief."
```

The model sees `researcher` as a tool. The child runs in its own session with
its own transcript and cost.

## Choose who decides

`worker` sets who answers this agent's decisions.

| `worker` | Result |
| --- | --- |
| Set | The engine POSTs every decision to that URL. |
| Not set | The engine decides by accepting its own proposal. |

Routing is per agent. An engine-hosted parent can call a worker-hosted child.

```toml title="subs.toml"
[agent.triage]
llm = "claude"
model = "claude-haiku-4-5"
worker = "https://triage.internal/agent"
```

## Let a worker write the config

A section does two things: it declares that the agent exists, and it can set the
agent's config. An agent whose worker builds its own config needs only the
first.

```toml title="subs.toml"
[agent.support]
worker = "http://localhost:4000/substructure/agent"
```

That section sets no config, so `session.start` carries no proposal and the
worker declares the whole agent.

Two rules follow:

- An agent that sets no config needs a `worker`.
- An agent that sets any config needs `llm` and `model`.

A partial config is a parse error.

## Where tools come from

| Source | Runs on | Declared in |
| --- | --- | --- |
| A connector | The engine | `mcp` on the agent |
| A plugin | The engine | `plugins` on the agent |
| Your code | Your worker | The config the worker returns |
| The browser | The client | `tools` on the agent, with `handler = "client"` |

Only browser tools go in the file. The tools that your worker runs are worker
code.

## Next steps

- [Connectors](./40-connectors.md): tools from Sentry, GitHub, and any MCP server.
- [Plugins](./45-plugins.md): skills and servers from a plugin directory.
- [Workers](./50-workers.md): decide with your own code.
- [Subagents](./80-subagents.md): agents that call agents.
- [Config](./220-config.md): every key in the file.
