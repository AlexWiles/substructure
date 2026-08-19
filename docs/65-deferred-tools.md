---
title: Deferred tools
group: Building agents
---

A deferred tool is one that the request does not carry. The engine still holds
it, still finds it in a search, and still runs a call to it.

Tool definitions sit at the front of the request, before the conversation. A
provider caches an exact prefix, so a definition that moves costs the cache of
everything behind it. Deferral keeps a large tool set out of that prefix.

Use it when the list is long enough that the model starts choosing badly, or
when the definitions cost more than they earn. The model reads every definition
on every turn.

## Example

Set `defer` on the tools the agent seldom needs.

```javascript title="server.mjs"
const tools = [
    { name: "get_weather", description: "The weather for a city.", input: citySchema },
    { name: "run_payroll", description: "Run payroll for a month.", input: monthSchema, defer: true },
    { name: "restate_ledger", description: "Restate the ledger.", input: ledgerSchema, defer: true }
];

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return { agent: { model: "claude-haiku-4-5", tools } };
    }

    // A deferred tool arrives here under its own name.
    if (trigger.type === "tool.execute") {
        const text = run(trigger.name, trigger.input.value);
        return { actions: [{ type: "tool.result", result: { content: [{ type: "text", text: text }] } }] };
    }

    return proposed;
}
```

The request carries `get_weather` and two tools that the engine adds. It
carries neither deferred tool.

## The two tools

An agent with one or more deferred tools gets two more. It gets one set of two,
whatever the number of deferred tools.

```jsonc
// tool_search — the tools that match, with their schemas
{ "query": "payroll" }
// call_tool
{ "name": "run_payroll", "arguments": { "month": "2026-07" } }
```

Each tool does one thing. A search answers with the name, the description, and
the input schema of each match, so one search gives the model everything it
needs to make a call. An empty query matches every tool, so a model that does
not know what is available starts there.

A search gives each tool one name. `call_tool` takes that name back, exactly as
the search gave it. It is the same name that the model calls directly when the
tool is not deferred.

A search covers every tool of the agent: from your worker or from a connection,
deferred or not. So an empty answer means that the agent has no tools.

## Any source

Deferral is a property of a tool. Each source sets the flag its own way.

| Source | How |
| --- | --- |
| A tool your worker declares | `defer: true` on the definition |
| One connection | `tools = { defer = true }` on the entry |
| Every tool of an agent | `defer_tools = true` |

`defer_tools` is the agent's default, and a tool or a connection that states
its own `defer` overrides it. See
[Connectors](./40-connectors.md#defer-a-connection).

An agent can mix the two. A tool that does not defer stays in the request,
beside `tool_search` and `call_tool`.

`defer_tools` takes `true` for the defaults, or a table for the settings. The
presence of the key is the switch, so an agent cannot carry a setting that does
nothing.

```toml title="substructure.toml"
[agent.support]
defer_tools = true

# The same agent, with the settings written out.
[agent.other.defer_tools]
strategy = "search"
```

| Key | Default | What it sets |
| --- | --- | --- |
| `strategy` | `search` | Which tools the engine gives the model. `search` is the only value today. |
| `max_matches` | `5` | The most matches one search answers with. A match carries a whole schema, so an answer with many matches is the tool list that the search replaced. An answer says how many matches it left out. |

`max_matches` is never zero: a search that can never return a match is a search
that the model cannot use.

## What does not change

The wrapper stops at the engine. A `call_tool` becomes the call that it names,
with that tool's own name, that tool's own arguments, and that tool's own
route.

| | |
| --- | --- |
| Where it runs | Its own handler decides: your worker, the client, or the engine. |
| `tool.execute` | Arrives with the tool's own name. Your worker cannot tell the difference. |
| `tool.finished` | Reports the tool's own name. |
| Schemas | The engine checks the arguments against the tool's own `input`, because the provider never received it. |
| Retries | The call's own policy. |

The engine refuses a `call_tool` that names a tool the agent cannot reach. The
error names the tools that the agent can reach.

A deferred name has no length limit. It never reaches a provider, so there is no
provider limit to fit.

## The cache

The two definitions are constant. Their names and their text say nothing about
which tools exist, so the request does not change when the set of tools behind
them changes.

| What changes mid-session | What the request does |
| --- | --- |
| A connection is added | Nothing, if its tools defer. |
| A connection is removed | Nothing, if its tools deferred. |
| A connection's fetch settles | Nothing. |
| A tool that does not defer is added | It enters the request, and the cache behind it is lost. |

The engine decides from the config alone, and not from what a fetch has
answered. An agent that sets `defer_tools` therefore carries the two tools from
its first turn, even before it names a connection. A connection added in turn 50
moves no definition.

The answers carry everything that varies: which tools exist, and what each one
takes. An answer is a tool result, at the end of the request, behind the
cache.

## Next

- [Tool calls](./60-tools.md): the rules a deferred tool still follows.
- [Connectors](./40-connectors.md): a connection that defers its tools.
- [Sub-agents](./80-sub-agents.md): a third answer to a large tool set.
- [Async tools](./110-async-tools.md): answer a call later. A separate idea.
