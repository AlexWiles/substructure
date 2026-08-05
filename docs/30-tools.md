---
title: Tool calls
group: Building agents
---

A tool is a function the model can call. You declare it in the agent config.
When the model calls one, the engine sends your worker a `tool.execute` trigger.
The worker runs the tool and answers with a `tool.result` or a `tool.error`. The
engine then ends the call, sends a `tool.finished` trigger, and prompts the
model again with the result.

## Example

A tool with an input schema. The engine validates the model's arguments. The
worker returns a `tool.error` when the tool fails.

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
                model: "claude-haiku-4-5-20251001",
                tools: tools.map(({ name, description, input }) =>
                    ({ name, description, input }))
            }
        };
    }

    // The model called a tool. Run it, or say why it failed.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        try {
            const result = tool.exec(trigger.input.value);
            return { actions: [{ type: "tool.result", result }] };
        } catch (err) {
            return { actions: [{ type: "tool.error", error: err.message }] };
        }
    }

    return proposed;
}
```

## Declaration

Tools go in the agent config. The model sees `name`, `description`, and `input`.
The other fields are for the engine.

```typescript
type AgentTool = {
    name: string
    description?: string
    input?: unknown             // JSON Schema for arguments; omitted: none
    output?: unknown            // JSON Schema the result must satisfy
    handler?: "worker" | "client"  // where it runs; default worker
}
```

`input` and `output` are optional JSON Schemas. The engine applies them. See
[Schemas](#schemas).

## Schemas

The engine checks a tool's schemas in both directions. It only validates. It
never changes a value: a value that passes is exactly the value that came in.
The engine does not convert types or add defaults. A tool with no schema is not
checked.

### input

Before the `tool.execute` trigger reaches your worker, the engine checks the raw
`arguments` against `input`. It reports the result in `trigger.input`:

| `status` | Meaning |
| --- | --- |
| `valid` | The arguments are an object and match the schema. `value` holds them. |
| `invalid` | The arguments are an object, but they do not match the schema. |
| `malformed` | The arguments are not a JSON object. |

Validation never stops a call. All three cases reach your worker, so you decide
whether to run the tool, correct the arguments, or refuse. For `invalid` and
`malformed` arguments, the engine puts a `tool.error` in `proposed`. You can
return it unchanged.

### output

When a call ends with a result, the engine checks the result against `output`. A
result that does not match the schema does not reach the model. The call ends
with a `tool.error` that cannot be retried. The engine reads the result as JSON
if it parses, and as a string if it does not.

## Triggers

Two triggers are part of a tool call. Your worker answers `tool.execute`. It
usually accepts the proposal for `tool.finished`.

```typescript
type ToolExecute = {
    type: "tool.execute"
    id: string
    name: string
    arguments: string           // raw argument string
    input: ToolInput            // the engine's validation of arguments
    attempt: number
    deadline?: string
}

type ToolInput =
    | { status: "valid"; value: unknown }
    | { status: "invalid"; value: unknown; error: string }
    | { status: "malformed"; error: string }

type ToolFinished = {
    type: "tool.finished"
    id: string
    ok: boolean
    name: string
    result?: string
    error?: string
}
```

### `tool.execute`

The model called a tool. Run it. `input` holds the engine's validation of the
raw `arguments`. See [Schemas](#schemas). For a valid call, `proposed` is empty,
because only your worker can run the tool. Answer with a `tool.result` or a
`tool.error`. If validation failed, or if the model named a tool you did not
declare, `proposed` holds a `tool.error` that you can return unchanged.

### `tool.finished`

A tool call ended, after its result or error and after any retries. `proposed`
records the result as a tool message and prompts the model again. If other calls
are still in flight, `proposed` waits instead. Return it to continue the loop.

## Actions

```typescript
type ToolCall = {
    type: "tool.call"
    id?: string                 // omitted: the engine creates one
    name: string
    arguments: unknown
    retry?: RetryOverride       // applied over the agent config, else the engine default
}

type ToolResult = {
    type: "tool.result"
    id?: string                 // id and attempt default to those of the
    attempt?: number            // tool.execute trigger you answer
    result: unknown
}

type ToolError = {
    type: "tool.error"
    id?: string
    attempt?: number
    error: string
    retryable?: boolean         // default false: terminal
    code?: ErrorCode
    detail?: unknown
}
```

### `tool.call`

Start a tool call. The engine proposes one for each call the model makes. Your
worker can also send one. `retry` limits the attempts.

The tool's name decides where the call runs. A tool you declared with `handler:
"client"` runs on the client. All other tools run on your worker. The engine
reads this from the current config and writes it onto the call, so a later
config change cannot move a call that is already in flight.

### `tool.result`

End a call with a result. If the tool declares an `output` schema and the result
does not match it, the call ends with an error. See [Schemas](#schemas).

### `tool.error`

End a call with a failure. By default the engine does not retry it. Set
`retryable: true` to retry under the call's `retry` policy. `code` and `detail`
hold structured information. The model reads the error text as the tool's
result, so write it for the model to read.

For the full types, see [Protocol](./150-protocol.md).

## Next

- [Client-side tools](./90-client-tools.md): tools that run in the browser.
- [Sub-agents](./80-sub-agents.md): a tool call that starts another agent.
- [Connectors](./85-connectors.md): tools the engine runs on a service.
- [Retries](./120-retries.md): timeouts and backoff.
