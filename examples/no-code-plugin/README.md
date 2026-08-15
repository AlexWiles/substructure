# no-code-plugin

An agent that uses an [agent plugin](https://agent-plugins.org): a directory
holding a skill and an MCP server, declared in one line. There is no worker
and no code — plugin code never runs; the CLI resolves the directory to data.

The plugin bundles an incident-response *method* (`skills/respond/`) with the
file tools to carry it out (`mcp.json`). `mcp-server.mjs` serves the
`runbooks/` directory over `list_dir` and `read_file`, standing in for
wherever your runbooks really live, so the example runs with nothing to sign
up for.

The knowledge sits in two places on purpose. The escalation policy rides
*inside* the plugin (`references/escalation.md`), fixed at apply. The
runbooks sit *behind the server*, read live — edit one and the next incident
reads the new version.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Start the file server and give the engine a key. The server is on this
machine, so the turn runs here too — there is no `[remote]` in this file.

```sh
node mcp-server.mjs &
export OPENROUTER_API_KEY=sk-or-...
```

```sh
subs run "the database is down, what do I do?"
```

```text
→ skill {"name":"runbooks:respond"}
← skill  Skill runbooks:respond — Respond to an on-call incident…
→ files__list_dir {}
→ files__read_file {"path":"database-down.md"}
→ skill {"name":"runbooks:respond","file":"references/escalation.md"}
  1. Check the connection pool first …
  2. Restart the replica, never the primary: `db-ctl restart replica-1`
  …
  Escalate only if the primary itself is down.
```

## What happens

The model's system prompt carries a catalog entry: the plugin, its skills,
and their descriptions. Nothing else — the file server has not been dialled.

The incident matches the skill, so the model calls
`skill({name: "runbooks:respond"})`. That one call loads the skill's
instructions, **enables the plugin**, and connects its server. The skill says
to answer from the runbooks, so the model lists them, reads the matching one,
and reads the bundled escalation policy with the same `skill` tool and a
`file` argument.

Enabling is recorded where it happened: a fork of the session from before the
call does not inherit it, and an unused plugin never dials anything.

## The plugin's server is a connection

`mcp.json`'s `files` joins the registry as `runbooks-files` —
`<plugin>-<server>` — and everything about connections applies to it:
filters, approval, `subs mcp login runbooks-files` when a server needs one.
`mcp.json` has no field for how a server authenticates, so the declaration
says it where it knows better:

```toml
[plugin.runbooks]
path = "./plugins/runbooks"
auth = { files = "none" }
```

## Narrow it, like any connection

The knobs on the agent's entry apply to the plugin's servers:

```toml
[agent.oncall]
plugins = [{ id = "runbooks", tools = { include = ["read_file"] } }]
```

`read_only` works too, and reads the server's annotations: both tools here
declare `readOnlyHint`, so both survive it — a server that annotates nothing
would give you none.
