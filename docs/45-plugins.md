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

```toml title="subs.toml"
[plugin.pdf]
path = "./plugins/pdf-tools"

[agent.support]
llm = "claude"
model = "claude-sonnet-5"
plugins = [
  "pdf",
  { id = "mcp.crm", tools = { read_only = true }, approve = "destructive" },
]
```

The table form's settings apply to the plugin's MCP servers, with the same
meanings as an `mcp` entry.

## What the model sees

Each plugin puts one entry in the system prompt from turn one: its
description and its skills, with the name and the description of each skill.
The engine writes the entry once per branch, in the same placement order that
MCP announcements use, so a plugin added mid-session introduces itself without
rewriting the cached prefix.

One constant engine tool, `skill`, loads a skill:

```json
{ "name": "pdf:form-filling" }
```

The answer is the skill's instructions and its file listing. Passing `file`
reads one of the listed files. A wrong name answers with the directory — the
plugin's skills, or the agent's plugins — so one round trip corrects the name.

An answer is a tool result like any other, so what a skill holds decides the
answer's shape. A text file goes into the answer as text. The engine lists
anything else — an image, a PDF — with its MIME type and its size, and returns
it as an attachment, which the model sees the way it sees an attachment from a
connection. The bytes go to blob storage; the skill keeps a reference, so
nothing large travels with the config.

A plugin is content, not a declaration, so it goes to the deployment on its
own, and the hash of the directory it came from names it. `subs apply` sends
the plugin first and the config after — the config names a plugin by hash, so
the plugin has to be there for the document to mean anything. `subs apply`
uploads only what the deployment does not already hold, so an unchanged
directory sends nothing, and apply reports what it sent.

A local engine needs none of this. It reads the directory at startup and
stores the binaries while it reads.

## Enable a plugin

Naming a plugin is what turns it on. The engine fetches its MCP servers, the
next model call waits for the fetch, the engine announces the servers once,
and their tools then behave like the tools of any connection: filters,
deferral, approval, and auth policy all come from the plugin entry.

The config in force is the whole answer, so a worker enables a plugin the way
it changes anything else about an agent: by writing the plugin into the config
that it returns. A plugin added mid-session lists itself and starts its servers
on the branch where the worker wrote it, without rewriting the cached
prefix.

The model does not enable plugins. `skill` reads a skill and nothing else.

## Servers and credentials

A plugin's `mcp.json` servers join the connection registry, and a file names one
by where it is declared: the `pdf` plugin's `renderer` server is
`plugin.pdf.mcp.renderer`. They authorize like any connection:
`subs auth plugin.pdf.mcp.renderer`. The engine skips `stdio` servers with a
notice: plugin code does not run in a deployment. Files under `scripts/` in a
skill travel with the plugin as readable files only.

`mcp.json` carries what the plugin author knows. What this deployment knows goes
in a section of its own, one per server:

```toml
[plugin.pdf]
path = "./plugins/pdf-tools"

[plugin.pdf.mcp.renderer]
auth = "none"
```

`auth = "none"` says the server wants no credential, so nothing keeps asking you
to authorize it. `url` belongs here too: point a plugin's server at a staging
host without forking the plugin.

```toml
[plugin.pdf.mcp.renderer]
url = "https://pdf.staging.example.com/mcp"
```

## Next

- [Connectors](./40-connectors.md): the rules a plugin's servers follow.
- [Agents](./30-agents.md): the section that names a plugin.
- [Config](./220-config.md): every key of a `[plugin.<id>]` section.
- [Tool calls](./60-tools.md): what the model does with the tools a plugin
  brings.
