---
title: Deferred tools
group: Building agents
---

A deferred tool is one the request does not carry. The engine still holds it,
still finds it in a search, and still runs a call to it.

Tool definitions sit at the front of the request, before the conversation. A
provider caches an exact prefix, so a definition that moves costs the cache of
everything behind it. Deferral keeps a large tool set out of that prefix.

Use it above about forty tools. A model chooses badly above that number, and it
reads every definition on every turn.

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
        return { actions: [{ type: "tool.result", result: run(trigger.name, trigger.input.value) }] };
    }

    return proposed;
}
```

The request carries `get_weather` and three tools the engine adds. It carries
neither deferred tool.

## The three tools

An agent with one or more deferred tools gets three more. It gets one set,
whatever the number of deferred tools.

```jsonc
// list_tools — every tool, by name, with no schema
{ }
// tool_search — the tools that match, with their schemas
{ "query": "payroll" }
// call_tool
{ "name": "run_payroll", "arguments": { "month": "2026-07" } }
```

Each tool does one thing. A model that does not know what is available calls
`list_tools`. A model that knows what it wants calls `tool_search`. A search
that matches nothing says so, and names `list_tools`.

A list and a search give each tool one name. `call_tool` takes that name back,
exactly as it was given. It is the same name the model calls directly when the
tool is not deferred.

A search covers every tool of the agent: from your worker or from a connection,
deferred or not. Therefore an answer of nothing means that the agent has
nothing.

## Any source

Deferral is a property of a tool. Each source sets the flag its own way.

| Source | How |
| --- | --- |
| A tool your worker declares | `defer: true` on the definition |
| One connection | `tools = { defer = true }` on the entry |
| Every tool of an agent | `defer_tools = true` |

`defer_tools` is the agent's default, and a tool or a connection that states
its own `defer` overrides it. See
[Connectors](./40-connectors.md#deferring-a-connection).

An agent can mix. A tool that does not defer stays in the request, beside the
three.

## What does not change

The wrapper stops at the engine. A `call_tool` becomes the call it names, with
that tool's own name, that tool's own arguments, and that tool's own route.

| | |
| --- | --- |
| Where it runs | Its own handler decides: your worker, the client, or the engine. |
| `tool.execute` | Arrives with the tool's own name. Your worker cannot tell. |
| `tool.finished` | Reports the tool's own name. |
| Schemas | The engine checks the arguments against the tool's own `input`, because the provider never received it. |
| Retries | The call's own policy. |

A `call_tool` that names a tool the agent cannot reach is refused. The error
names the tools it can reach.

A deferred name has no length limit. It never reaches a provider, so there is
nothing to fit.

## The cache

The three definitions are constant. Their names and their text say nothing about
which tools exist, so the request does not change when the set behind them
changes.

| What changes mid-session | What the request does |
| --- | --- |
| A connection is added | Nothing, if its tools defer. |
| A connection is removed | Nothing, if its tools deferred. |
| A connection's fetch settles | Nothing. |
| A tool that does not defer is added | It enters the request, and the cache behind it is lost. |

The engine decides from the config alone, and not from what a fetch has
answered. An agent that sets `defer_tools` thus carries the three tools from
its first turn, before it names one connection. A connection added in
turn 50 moves no definition.

The answers are where everything variable lives: which connections exist, how
many tools each one has, and what each server says it is for. An answer is a
tool result, at the end of the request, behind the cache.

## Next

- [Tool calls](./60-tools.md): the rules a deferred tool still follows.
- [Connectors](./40-connectors.md): a connection that defers its tools.
- [Sub-agents](./80-sub-agents.md): a third answer to a large tool set.
- [Async tools](./110-async-tools.md): answer a call later. A separate idea.
