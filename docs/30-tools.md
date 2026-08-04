---
title: Tool calls
group: Building agents
---

A tool is a function the model can call. You declare it in the agent config.
When the model calls one, the engine sends your worker a `tool.execute`
trigger; the worker runs it and answers with a `tool.result` or `tool.error`.
The engine settles the call, fires a `tool.finished` trigger, and re-prompts
the model with the outcome.

## Example

A tool with an input schema. The engine validates the model's arguments; the
worker returns a `tool.error` when the tool itself fails.

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

    // The model called a tool. Run it, or report why it failed.
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

Tools live in the agent config. The model sees `name`, `description`, and
`input`; the rest is for the engine.

```typescript
type AgentTool = {
    name: string
    description?: string
    input?: unknown             // JSON Schema for arguments; omitted: none
    output?: unknown            // JSON Schema the result must satisfy
    handler?: "worker" | "client"  // where it runs; default worker
}
```

`input` and `output` are optional JSON Schemas; the engine enforces them (see
[Schemas](#schemas)).

## Schemas

The engine enforces a tool's declared schemas in both directions. It only
validates and never mutates: a passing value is exactly what came in, with no
coercion or default-filling. Tools with no schema pass through unchecked.

### input

Before the `tool.execute` trigger reaches your worker, the engine classifies
the raw `arguments` against `input` and reports the outcome in `trigger.input`:

| `status` | Meaning |
| --- | --- |
| `valid` | Parsed to an object and conforms. `value` holds the arguments. |
| `invalid` | Parsed to an object but violates the schema. |
| `malformed` | Not a JSON object at all. |

Validation never blocks the call. Every case reaches your worker, so you
decide whether to run, coerce, or reject it. When the arguments are `invalid`
or `malformed`, the engine sets `proposed` to a ready-made `tool.error` you can
return as-is.

### output

When a call settles with a result, the engine checks it against `output`. A
result that violates the schema never reaches the model. It settles as a
terminal `tool.error` instead. The result is read as JSON when it parses, else
as a plain string.

## Triggers

Two triggers concern a tool call. Your worker answers `tool.execute`; it
usually accepts the proposal on `tool.finished`.

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

The model called a tool; run it. `input` carries the engine's validation of
the raw `arguments` (see [Schemas](#schemas)). `proposed` is empty on a clean
call, because only your worker can run the tool, so you answer with a
`tool.result` or `tool.error`. When validation fails or the model named a tool
you never declared, `proposed` is instead a ready-made `tool.error` to return
as-is.

### `tool.finished`

A tool call settled, after its result or error and any retries. `proposed`
records the outcome as a tool message, then re-prompts the model, or holds
when sibling calls are still in flight. Return it to continue the loop.

## Actions

```typescript
type ToolCall = {
    type: "tool.call"
    id?: string                 // omitted: the engine mints one
    name: string
    arguments: unknown
    retry?: RetryOverride       // layered over the agent config's, else the engine's
}

type ToolResult = {
    type: "tool.result"
    id?: string                 // id and attempt default to the answered
    attempt?: number            // tool.execute trigger
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

Dispatch a tool call. The engine proposes these for the model's calls, and
your worker can issue one directly. `retry` bounds the attempts.

Where the call runs follows from its name: a tool you declared `handler:
"client"` runs on the client, anything else runs on your worker. The engine
resolves this against the config in force and freezes it onto the call, so a
config change can't reroute a call already in flight.

### `tool.result`

Settle a call with a result. A result checked against a declared `output`
schema may settle as a terminal error instead (see [Schemas](#schemas)).

### `tool.error`

Settle a call with a failure. It's terminal by default; set `retryable: true`
to retry under the call's `retry` policy. `code` and `detail` carry structured
info. The error text reaches the model as the tool's result, so write it to be
read.

Full types in [Protocol](./150-protocol.md).

## Next

- [Client-side tools](./90-client-tools.md): tools that run in the browser.
- [Sub-agents](./80-sub-agents.md): delegation as a tool call.
- [Connectors](./85-connectors.md): tools the engine runs against a service.
- [Retries](./120-retries.md): timeouts and backoff.
