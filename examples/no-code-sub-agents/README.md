# no-code-sub-agents

A team of agents in one file. There is no worker and no code.

An agent names the agents it may delegate to. Each one is another section, with
its own model and its own prompt. A delegation runs in its own session, so the
parent gets the answer and not the work.

```toml
[agent.assistant]
sub_agents = ["writer", "critic"]

[agent.writer]
description = "Drafts short copy from a brief"
```

`description` is what the parent's model reads when it chooses. It sits on the
agent, not on the edge, so two parents cannot describe one specialist
differently.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Deploy the file and give it a key:

```sh
subs login
subs apply
subs auth llm.openrouter
```

```sh
subs run "two sentences for the homepage of a bicycle repair shop that comes to you"
```

Pretty output shows each delegation as it runs.

## Run it here instead

Delete `[remote]` and the turn runs on this machine, on your own key:

```sh
export OPENROUTER_API_KEY=sk-or-...
subs run -c substructure.toml --agent assistant \
    "two sentences for the homepage of a bicycle repair shop that comes to you"
```
