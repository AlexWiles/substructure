---
title: Subagents
group: Building agents
---

A subagent is another agent that the model sees as a tool.

When the model calls it, the engine starts a child session. The child's result
goes back to the parent as the tool's result.

## Example

Each agent is its own section. The parent names the child. The child can use a
cheaper model.

```toml title="subs.toml"
[llm.claude]
type = "anthropic"

[llm.cheap]
type = "anthropic"

[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
subagents = ["agent.poet"]

[agent.poet]
description = "Writes a haiku on any topic."
llm = "cheap"
model = "claude-haiku-4-5"
system = "You are a poet. Respond with a single haiku."
```

Neither agent needs a worker. Routing is per agent, so an engine-hosted parent
can call a worker-hosted child.

One worker can serve both. Route on `agent_id`.

```javascript title="server.mjs"
function assistant({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return { agent: { ...proposed.agent, system: "Delegate poetry to the poet." } };
    }
    return proposed;
}

const poet = ({ proposed }) => proposed;

const decide = (req) => (req.agent_id === "poet" ? poet(req) : assistant(req));
```

## Declare a subagent

`subagents` lists the agents this agent can call, each written `agent.<id>`.
The model sees each one as a tool with the bare ID as its name. The tool takes
one `message` argument.

Agent IDs and tool names share one namespace. An ID must not match a tool name.

The tool's description comes from the `description` on the section it names, so
two parents that call the same child describe it the same way.

An agent that exists only to be called can carry only a `description` and a
`worker`.

A worker can also write the description. The expanded `subagents` arrive in the
`session.start` proposal, so `description` is required only for an agent with no
worker.

## How the tool is offered

An entry can also be a table with `id`, `defer`, `prefix`, and `mode`. The
string form is the table with all of them unset.

```toml title="subs.toml"
[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
subagents = [
    "agent.poet",
    { id = "agent.researcher", defer = true, prefix = true },
]
```

`defer = true` keeps the tool out of the request, the same way it works for
[deferred tools](./65-deferred-tools.md). The model finds it with `tool_search`
and delegates with `call_tool`, passing the `message` argument. An agent-wide
`defer_tools` covers subagent tools too; an entry's own `defer` overrides it
either way.

`prefix = true` offers the tool as `agent__<id>` instead of `<id>`. Use it when
the bare ID would collide with another tool's name. A prefixed ID frees the
bare name, and the collision rule applies to the offered name.

## One tool for every subagent

`subagent_tools` says what shape the subagents take. The default is
`per_agent`: one tool per subagent, as above.

```toml title="subs.toml"
[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
subagents = ["agent.poet", "agent.researcher"]
subagent_tools = { strategy = "per_agent" }
```

`single` offers one tool named `subagent` for all of them. The call names the
agent in `agent`; the description lists each agent and what it does.

```toml title="subs.toml"
[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
subagents = ["agent.poet", "agent.researcher"]
subagent_tools = { strategy = "single" }
```

Use `single` when an agent has many subagents: the model reads one tool instead
of one per agent. `agent` accepts only the IDs the agent declares, so a call
that names another one comes back as a tool error.

Under `single` the tool takes the agent's own `defer_tools`, and `subagent`
joins `tool_search` and `call_tool` as a name no tool of yours may take. An
entry's `prefix` and `defer` shape a per-agent tool, so the file rejects them
under this strategy.

At the depth limit the engine leaves `subagent` off the list, the same way it
leaves the per-agent tools off.

## How deep subagents nest

A child can have subagents of its own. `max_subagent_depth` bounds the chain: a
session that many parents deep may not delegate. The default is 5.

Set it at the top of the file for the whole project, or on an agent section;
the agent's own value wins. `max_subagent_depth = 0` means the agent never
delegates.

At the limit the engine leaves the subagent tools off the model's tool list,
and a spawn that arrives anyway comes back as a tool error naming the limit.

## Call a subagent

When the model calls a subagent, `proposed` starts a child session. It carries
the child's first message, taken from the `message` argument.

The child runs as a normal session with its own `agent_id`, transcript, and
cost. Its decision requests carry an `ancestry` list of the sessions above it.

## When the child finishes

When the child's turn ends, the parent receives a `subagent.finished` trigger.

`proposed` records the result as the tool's result and prompts the parent again.
The engine adds the child's cost and token use to the parent's turn.

The tool's result is an object with the child's answer and the session it ran
in.

```json
{ "session": "9f2c…", "result": "A haiku about rain." }
```

A call that failed answers with the error text instead, the way any tool error
does.

## Continue a conversation

Every subagent tool takes an optional `session`. Pass the `session` from an
earlier answer and the call runs another turn in that same child session, which
still holds the whole conversation. Leave `session` out to start a fresh child.

A parent may continue only a session it started itself, running the agent the
call names. A `session` that names anything else — another parent's child, a
session of another agent, or no session at all — comes back as a tool error.

A child answers one call at a time. A second call to a session that is still
working comes back as a tool error; wait for the first result.

## Detached calls

A blocking call holds the parent's turn until the child answers. A detached
call does not. It answers at once with the child's session id, and the parent
continues its turn. When the child finishes, the result arrives in the
parent's session as a message.

```
<subagent_result agent="poet" session="9f2c…">
A haiku about rain.
</subagent_result>
```

If the parent is mid-turn when the result lands, the message waits for the
turn to end and then opens a turn of its own. Results that land while one is
still waiting join it: one turn delivers them all. A failed child turn arrives
the same way, with `error="true"` and the error text as the body.

Set `mode` on a `subagents` entry, or `subagent_mode` on the agent section, to
control who decides. The entry's own value wins.

```toml title="subs.toml"
[agent.assistant]
llm = "claude"
model = "claude-sonnet-4-5"
subagents = [
    "agent.poet",                                   # the model picks per call
    { id = "agent.researcher", mode = "detached" }, # always detached
    { id = "agent.critic", mode = "blocking" },     # always blocking
]
```

- Unset means `any`: the tool offers `mode` with `blocking` and `detached`,
  and the model picks per call. A call that names no `mode` blocks.
- `detached` pins the mode. The tool loses `mode` and says in its description
  that it runs detached.
- `blocking` pins the old behavior: the tool keeps the schema it always had.

Under `single`, one tool serves every agent, so the offered `mode` values are
the union across them and each pin is named in the agent list.

A worker that authors its own `subagent.spawn` passes `mode` on the action; an
entry's pinned mode still wins.

To collect a detached result instead of waiting for the message, the model
calls `subagent_wait` with the child's `session`. If the result is already in,
the call answers at once with the newest one; if the child is still working,
the call holds the turn until the child's turn ends, like a blocking call. A
wait that answers withdraws that child's undelivered message, so one result
never arrives twice; other children's messages stay queued.

The engine offers `subagent_wait` whenever any subagent can run detached, and
its name joins the names no tool of yours may take.
`subagent_tools = { wait = false }` removes it; results then arrive only as
messages. A worker can still author a `subagent.spawn` with `mode: "wait"`
either way.

While a detached child is working, another message to its session comes back
as a tool error naming `wait`. Cancelling the parent no longer reaches a
detached child whose call has settled: the child finishes on its own.

## Reference

```typescript
type SubagentMode = "blocking" | "detached" | "any"  // configuration
type SpawnMode = "blocking" | "detached" | "wait"    // one call
type Subagent = { id: string; description?: string; defer?: boolean; prefix?: boolean; mode?: SubagentMode }
type SubagentTools = { strategy?: "per_agent" | "single"; wait?: boolean }

// what the model passes a per-agent tool, and what `subagent` takes
{ message: string, session?: string, mode?: SpawnMode }
{ agent: string, message: string, session?: string, mode?: SpawnMode }

// what `subagent_wait` takes
{ session: string }

// the actions that start a child. the engine proposes them for you
{ type: "subagent.spawn", session_id?: string, agent_id: string, tool_call_id: string, mode?: SpawnMode }
{ type: "message.send", session_id: string, message: DraftMessage }

// the trigger
{ type: "subagent.finished", id: string, ok: boolean, session_id: string, agent_id: string }
```

On `subagent.finished`, `id` is the tool call and `session_id` is the child.

On `subagent.spawn`, leave `session_id` out to start a child. The engine mints
the child's id from the decision and the tool call, so a replay of the same
spawn opens the same session. A `session_id` continues that session instead,
and the engine takes only a session this parent started, running the agent the
call names. Anything else comes back as a tool error.

## Next steps

- [Tool calls](./60-tools.md): the child's result comes back as a tool result.
- [Agents](./30-agents.md): the section a child agent declares.
- [Durability](./200-durability.md): the engine saves child sessions.
