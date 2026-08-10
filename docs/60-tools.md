---
title: Tool calls
group: Building agents
---

A tool is a function the model can call. Your worker declares it and runs it.

When the model calls a tool, the engine sends your worker a `tool.execute`
trigger. The worker answers with a `tool.result` or a `tool.error`. The engine
records the result and prompts the model again.

## Example

A tool with an input schema. The engine validates the model's arguments.

```javascript title="server.mjs"
const forecasts = { "San Francisco": "foggy", Tokyo: "clear" };

const tools = [
    {
        name: "get_weather",
        description: "Get the weather for a city.",
        input: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"]
        },
        exec: ({ city }) => {
            const sky = forecasts[city];
            if (!sky) throw new Error(`No forecast for ${city}.`);
            return `It's ${sky} in ${city}.`;
        }
    }
];

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5",
                tools: tools.map(({ name, description, input }) =>
                    ({ name, description, input }))
            }
        };
    }

    // The model called a tool. Run it, or say why it failed.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        try {
            return { actions: [{ type: "tool.result", result: tool.exec(trigger.input.value) }] };
        } catch (err) {
            return { actions: [{ type: "tool.error", error: err.message }] };
        }
    }

    return proposed;
}
```

## Declaring a tool

Tools go in the agent config. The model sees `name`, `description`, and `input`.

```typescript
type AgentTool = {
    name: string
    description?: string
    input?: unknown             // JSON Schema for the arguments
    output?: unknown            // JSON Schema the result must match
    handler?: "worker" | "client"  // where it runs. default worker
    defer?: boolean             // keep it out of the request. default: the agent's defer_tools
}
```

## Many tools

Above about forty tools, a model chooses badly, and each definition sits at the
front of the request where the provider keeps its cache.

Set `defer` on the tools an agent seldom needs. The request leaves them out, and
the agent gets `list_tools`, `tool_search`, and `call_tool` in their place. The
model searches for a tool and names it to `call_tool`. Your worker receives an
ordinary `tool.execute`, under the tool's own name.

See [Deferred tools](./65-deferred-tools.md).

## Schemas

The engine checks a tool's schemas in both directions. It only validates. A
value that passes is exactly the value that came in. The engine converts no
types and adds no defaults.

A tool with no schema is not checked.

### input

Before `tool.execute` reaches your worker, the engine checks the raw `arguments`
against `input`. It reports the result in `trigger.input`.

| `status` | Meaning |
| --- | --- |
| `valid` | The arguments are an object and match the schema. `value` holds them. |
| `invalid` | The arguments are an object. They do not match the schema. |
| `malformed` | The arguments are not a JSON object. |

Validation never stops a call. All three reach your worker, so you decide
whether to run the tool, correct the arguments, or refuse.

For `invalid` and `malformed`, the engine puts a `tool.error` in `proposed`. You
can return it unchanged.

### output

When a call ends with a result, the engine checks the result against `output`. A
result that does not match never reaches the model. The call ends with a
`tool.error` that cannot be retried.

The engine reads the result as JSON if it parses, and as a string if it does
not.

## The two triggers

Your worker answers `tool.execute`. It usually accepts the proposal for
`tool.finished`.

### `tool.execute`

The model called a tool. Run it.

For a valid call, `proposed` is empty. Answer with a `tool.result` or a
`tool.error`. If validation failed, or the model named a tool you did not
declare, `proposed` holds a `tool.error` you can return unchanged.

```typescript
{
    type: "tool.execute"
    id: string
    name: string
    arguments: string           // the raw argument string
    input: ToolInput            // the engine's validation
    attempt: number
    deadline?: string
}
```

### `tool.finished`

A tool call ended, after its result and after any retries.

`proposed` records the result as a tool message and prompts the model again. If
other calls are still in flight, `proposed` waits. Return it to continue.

```typescript
{
    type: "tool.finished"
    id: string
    ok: boolean
    name: string
    result?: string
    error?: ErrorInfo
}
```

## The three actions

### `tool.call`

Start a tool call. The engine proposes one for each call the model makes. Your
worker can also send one.

The tool's name decides where the call runs. A tool declared with `handler:
"client"` runs on the client. Everything else runs on your worker.

### `tool.result`

End a call with a result.

```typescript
{
    type: "tool.result"
    id?: string                 // id and attempt default to those of the
    attempt?: number            // tool.execute you answer
    result: unknown
}
```

### `tool.error`

End a call with a failure.

```typescript
{
    type: "tool.error"
    id?: string
    attempt?: number
    error: string
    retryable?: boolean         // default false
    code?: ErrorCode
    detail?: unknown
}
```

The engine does not retry by default. Set `retryable: true` to retry under the
call's policy. See [Retries](./210-retries.md).

The model reads `error` as the tool's result. Write it for the model to read.

## Where tools run

| Source | Runs on | Declared in |
| --- | --- | --- |
| Your code | Your worker | The config the worker returns |
| A connector | The engine | `mcp` on the agent |
| The browser | The client | `tools`, with `handler = "client"` |

## Next

- [Deferred tools](./65-deferred-tools.md): keep a large tool set out of the request.
- [Client-side tools](./150-client-tools.md): tools that run in the browser.
- [Async tools](./110-async-tools.md): answer a call later.
- [Connectors](./40-connectors.md): tools the engine runs on a service.
- [Sub-agents](./80-sub-agents.md): a tool call that starts another agent.
- [Retries](./210-retries.md): timeouts and backoff.
