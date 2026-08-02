---
title: Protocol
group: Reference
---

The wire reference. The engine and your worker exchange JSON over HTTP.

Types are shown in TypeScript notation. `?` marks a field that may be
omitted. `unknown` is any JSON value. Timestamps are RFC 3339 strings.
Decimal money values are strings.

The machine-readable source of truth is
[`schemas/protocol.schema.json`](../schemas/protocol.schema.json) (all
types) and [`schemas/worker.openapi.json`](../schemas/worker.openapi.json)
(the worker endpoint). To generate bindings instead of writing them, see
[Typed bindings](./40-typed-bindings.md).

## Delivery

The engine POSTs a `DecisionRequest` to your worker's endpoint.

| Header | Value |
| --- | --- |
| `Content-Type` | `application/json` |
| `Accept` | `text/event-stream, application/json` |
| `traceparent` | W3C trace context |
| `X-Substructure-Signature` | `sha256=<hex HMAC-SHA256 of the body>`, when a signing secret is configured |

Respond with `application/json` (a `DecisionResponse`), or
`text/event-stream` to stream (see [Streaming](#streaming)).

## Decision request

```typescript
type DecisionRequest = {
    session_id: string
    decision_id: string
    agent_id: string
    identity: WorkerIdentity
    trigger: Trigger
    proposed: DecisionResponse      // empty when the engine has no default
    state: unknown              // your agent state, stored verbatim
    agent: AgentConfig | null
    calls: Call[]               // in-flight calls
    pending_calls: number       // in-flight tool and sub-agent calls
    messages: Message[]         // the active conversation path
    message_tree: MessageTree
    ancestry: string[]          // ancestor session ids, for sub-agents
    attempts: number            // delivery attempts for this decision
    deadline: string | null
    turn_id: string | null      // the turn's idempotency id
}

type WorkerIdentity = {   // the session owner, without the tenant
    id?: string
    metadata?: Record<string, string>
}

type Call = {
    id: string                  // sub-agents: the child session; fetches: the connection
    kind: "tool_call" | "llm_call" | "sub_agent" | "connector_sync"
    status: "pending" | "completed" | "failed" | "retry_scheduled" | "queued"
    attempt: number
    deadline?: string
    anchor?: string             // the tree node the call was requested at
    name?: string               // tool calls; fetches: the connection
    arguments?: string          // tool calls
    handler?: Handler
    stream?: boolean            // llm calls
    agent_id?: string           // sub-agents
    tool_call_id?: string       // sub-agents: the call the delegation answers
}

type Handler = "server" | "worker" | "client"
```

`proposed` is the engine's default continuation. Accept it by echoing it as
the decision. It is `null` when only the worker knows what to do, such as
running one of its own tools.

Every call and decision is queued first and dispatched when its
prerequisites settle, in arrival order. A `queued` call has not started: it
is still waiting on a fetch the agent's config owes, the live decision slot,
a parked branch, a running turn, or a blocked entry ahead of it in the
queue. Waits are bounded: every prerequisite carries a deadline and always
settles.

## Decision response

```typescript
type DecisionResponse = {
    messages?: DraftMessage[]   // messages to record
    actions?: Action[]          // what the engine should do next
    state?: unknown             // omitted or null keeps current state
    agent?: AgentConfig         // omitted keeps current config
}
```

To clear state, write a non-null empty value such as `{}`.

## Triggers

```typescript
type Trigger =
    | { type: "session.start" }
    | {
          type: "client.messages"
          messages: DraftMessage[]  // the client's full conversation view
          new_from: number          // index of the first new message
          client: ClientContext
      }
    | { type: "client.action"; name: string; args?: unknown }
    | {
          type: "tool.execute"
          id: string
          name: string
          arguments: string         // the raw argument string
          input: ToolInput          // the engine's validation of arguments
          attempt: number
          deadline?: string
      }
    | {
          type: "tool.finished"
          id: string
          ok: boolean
          name: string
          result?: string
          error?: string
      }
    | {
          type: "llm.execute"
          id: string
          request: unknown          // LlmRequest, or provider-native when format is set
          format?: "openai" | "anthropic"
          stream: boolean
          attempt: number
          deadline?: string
      }
    | {
          type: "llm.finished"
          id: string
          ok: boolean
          message?: DraftMessage
          truncated: boolean
          usage?: unknown
          cost?: string
          error?: string
          code?: ErrorCode
          detail?: unknown
      }
    | {
          type: "sub_agent.finished"
          id: string
          ok: boolean
          session_id: string
          agent_id: string
          result?: string
          error?: string
      }
    | { type: "interrupt.resumed"; interrupt_id: string; payload?: unknown }

type ToolInput =
    | { status: "valid"; value: unknown }     // parsed, conforms to the tool's input schema
    | { status: "invalid"; value: unknown; error: string }
    | { status: "malformed"; error: string }  // not a JSON object

type ErrorCode =
    | "provider_error"
    | "rate_limited"
    | "refused"
    | "budget_exceeded"
    | "deadline_exceeded"

type ClientContext = {
    tools?: AgentTool[]         // client-executed tools, layered onto the proposed config
    context?: unknown[]
    state?: unknown
    forwarded_props?: unknown
}
```

Answer `tool.execute` with `tool.result` or `tool.error`. Answer
`llm.execute` with `llm.result` or `llm.error`, or stream.

## Actions

```typescript
type Action =
    | {
          type: "llm.call"        // all fields optional; omitted fields fill
          id?: string             // from the agent config, then engine defaults
          model?: string
          messages?: DraftMessage[]  // explicit messages suppress system-prompt injection
          tools?: LlmTool[]
          temperature?: number
          max_completion_tokens?: number
          reasoning?: ReasoningConfig
          stream?: boolean        // default true
          retry?: RetryPolicy
          handler?: "server" | "worker"  // default server
      }
    | {
          type: "tool.call"
          id?: string             // omitted: the engine mints one
          name: string
          arguments: unknown
          retry?: RetryPolicy     // default: no retry
      }
    | {
          type: "tool.result"
          id?: string             // id and attempt omitted: taken from the
          attempt?: number        // answering tool.execute trigger
          result: unknown
      }
    | {
          type: "tool.error"
          id?: string
          attempt?: number
          error: string
          retryable?: boolean     // default false: terminal
          code?: ErrorCode
          detail?: unknown
      }
    | {
          type: "llm.result"
          id?: string
          attempt?: number
          response: unknown       // LlmResponse, or provider-native when the
      }                           // answered llm.execute carried a format
    | {
          type: "llm.error"
          id?: string
          attempt?: number
          error: string
          retryable?: boolean     // default false: terminal
          code?: ErrorCode
          detail?: unknown
      }
    | {
          type: "sub_agent.spawn"
          session_id: string
          agent_id: string
          tool_call_id: string    // the model tool call this delegation answers
          message?: DraftMessage  // the child's opening message, sent once it exists
          retry?: RetryPolicy
      }
    | { type: "message.send"; session_id: string; message: DraftMessage }
    | {
          type: "interrupt"
          interrupt_id?: string   // omitted: the engine mints one
          reason: string
          payload?: unknown
      }
    | { type: "done"; data?: unknown }
```

A bare `{ "type": "llm.call" }` prompts per the agent's identity over the
current conversation.

## Agent config

```typescript
type AgentConfig = {
    model: string               // the only required field
    system?: string
    handler?: "server" | "worker"  // where LLM calls run; default server
    format?: "openai" | "anthropic"  // wire format for worker-handled LLM
                                     // calls; requires handler worker
    retry?: RetryPolicy
    tools?: AgentTool[]
    sub_agents?: SubAgent[]
    mcp?: McpServer[]
}

type AgentTool = {
    name: string
    description?: string
    input?: unknown             // JSON Schema for arguments; omitted: no arguments
    output?: unknown            // JSON Schema results must satisfy; a violating
                                // result settles as a terminal error
    handler?: "worker" | "client"  // default worker
}

type SubAgent = {
    id: string                  // the agent to spawn, and the tool name the model sees
    description?: string
}

type McpServer = {
    id: string                  // a connection the engine holds; never a URL
    tools?: McpTools            // omitted: every tool the connection grants
}

type McpTools = {
    include?: string[]          // globs over the tool's name on the connection
    exclude?: string[]
    read_only?: boolean         // capability predicates read MCP annotations;
    non_destructive?: boolean   // an unannotated tool fails them
    idempotent?: boolean
}

type RetryPolicy = {
    timeout_secs: number | null
    max_retries: number
    backoff_base_secs: number
    backoff_max_secs: number
}
```

## Messages

```typescript
type Role = "system" | "user" | "assistant" | "tool"

type Content = string | ContentPart[]

type ContentPart =
    | { type: "text"; text: string }
    | { type: "image_url"; image_url: { url: string } }
    | { type: "file"; file: { filename: string; file_data: string } }
    | { type: "input_audio"; input_audio: { data: string; format: string } }
    | { type: "video_url"; video_url: { url: string } }

type ToolCall = {
    id: string
    type: string
    function: { name: string; arguments: string }
}

// A recorded message. DraftMessage is the same shape with an optional id,
// used wherever a message is not yet recorded.
type Message = {
    id: string
    role: Role
    content?: Content
    tool_calls?: ToolCall[]
    tool_call_id?: string
    name?: string
}

type MessageTree = {
    nodes: { message: Message; parent_id?: string }[]
    head_id?: string
}
```

## LLM requests and responses

The neutral shapes used when the agent config sets no `format`.

```typescript
type LlmRequest = {
    model: string
    messages: DraftMessage[]
    tools?: LlmTool[]
    temperature?: number
    max_completion_tokens?: number
    reasoning?: ReasoningConfig
}

type LlmTool = {
    name: string
    description: string
    input?: unknown             // JSON Schema; omitted: no arguments
    output?: unknown
}

type ReasoningConfig = {
    effort?: "xhigh" | "high" | "medium" | "low" | "minimal" | "none"
    max_tokens?: number
    exclude?: boolean
    enabled?: boolean
}

type LlmResponse = {
    model: string
    content?: string
    tool_calls?: ToolCall[]
    finish_reason?: string
    usage?: unknown
    cost?: string               // dollars, decimal string
    images?: { url: string }[]
}
```

## Client inputs

What clients submit. The target session rides the envelope: the CLI's
`--session`, or the client API's request body.

```typescript
type ClientInput =
    | {
          type: "client.message"
          agent_id: string
          turn_id?: string        // idempotency key
          message: DraftMessage
          stream?: boolean
          queue?: boolean         // hold for the next turn instead of refusing
      }
    | {
          type: "client.messages"
          agent_id: string
          turn_id?: string
          messages: DraftMessage[]  // the client's full conversation view
          stream?: boolean
          client?: ClientContext
      }
    | {
          type: "client.action"
          agent_id: string
          turn_id?: string
          name: string
          args?: unknown
      }
    | { type: "interrupt.resume"; interrupt_id: string; payload?: unknown }
    | { type: "tool.result"; id: string; attempt?: number; result?: unknown }
    | {
          type: "tool.error"
          id: string
          error: string
          retryable: boolean
          attempt?: number
      }
```

`agent_id` routes the turn and creates the session if new. Resubmitting a
completed `turn_id` returns the existing turn instead of running it again.

### Queuing a message

A session runs one turn at a time, so a submit that arrives while a turn is
running is refused with `turn_already_active`. `queue: true` asks for the other
behaviour: the engine takes the message, holds it, and starts it as the next
turn the moment the running one completes. The reply carries `queued: true`
when the turn has been taken but has not started.

Only `client.message` and `client.append` accept the flag. A full
`client.messages` view and a `client.action` are refused as before: a view
settles pending client tool calls as it arrives, so it cannot be held.

Queued turns run one at a time, in arrival order. A `turn_id` that is running,
queued, or completed is refused, so a retrying transport cannot ask the same
question twice.

## Streaming

A worker answering an `llm.execute` with `stream: true` may respond with
`text/event-stream` instead of JSON:

```
event: llm.token.delta
data: { "text": "hel" }

event: llm.token.delta
data: { "text": "lo" }

event: decision.result
data: { "actions": [{ "type": "llm.result", "response": … }] }
```

`llm.token.delta` frames carry a `StreamDelta`, or the provider's native
stream events when the `llm.execute` carried a `format`. The stream must end
with one `decision.result` frame (a `DecisionResponse`), or `decision.error`
with `message` and `retryable` (default `true`).

```typescript
type StreamDelta = {
    text?: string
    reasoning?: string
    tool_calls?: { id: string; name?: string; arguments?: string }[]
    finish_reason?: string
}
```
