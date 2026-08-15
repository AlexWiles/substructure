---
title: Plugins
group: Building agents
---

A plugin is an [agent-plugins](https://agent-plugins.org) directory: a
`plugin.json` manifest, skills under `skills/`, and MCP servers in `mcp.json`.
The CLI resolves the directory to data — at startup for a local engine, at
`subs apply` for a deployment — so a session never reads plugin files.

Declare a plugin at the project level and attach it to agents, the same way
connections work:

```toml title="substructure.toml"
[plugin.pdf]
path = "./plugins/pdf-tools"

[agent.support]
llm = "claude"
model = "claude-sonnet-5"
plugins = [
  "pdf",
  { id = "crm", enabled = true, tools = { read_only = true }, approve = "destructive" },
]
```

The table form's knobs apply to the plugin's MCP servers, with the same
meanings as an `mcp` entry. `enabled = true` turns the plugin on from the
session's first turn.

## What the model sees

Each plugin puts one entry in the system prompt from turn one: its
description and its skill names, with skill descriptions included until a
budget is spent. The entry is written once per branch, through the same
placement ladder MCP announcements use, so a plugin added mid-session
introduces itself without rewriting the cached prefix.

One constant engine tool, `skill`, loads a skill:

```json
{ "name": "pdf:form-filling" }
```

The answer is the skill's instructions and its file listing. Passing `file`
reads one of the listed files. A wrong name answers with the directory — the
plugin's skills, or the agent's plugins — so one round trip corrects it.

## Enabling

Using a skill enables its plugin. The engine records the enablement on the
branch where it happened, then fetches the plugin's MCP servers. The next
model call waits for the fetch, the servers are announced once, and their
tools behave like any connection's from then on — filters, deferral,
approval, and auth policy all come from the plugin entry.

The record is anchored: a fork from after the enablement inherits it, a fork
from before does not, and a rewind takes it back.

An unused plugin costs its catalog entry and nothing else. Its servers are
never dialled.

## Servers and credentials

A plugin's `mcp.json` servers join the connection registry as
`<plugin>-<server>` — `pdf-renderer` above. They authorize like any
connection: `subs mcp login pdf-renderer`, `subs mcp set-token pdf-renderer`.
`stdio` servers are skipped with a notice: plugin code does not run in a
deployment. `scripts/` in a skill ride along as readable files only.
