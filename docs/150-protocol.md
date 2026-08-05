---
title: Protocol
group: Reference
---

The wire reference. The engine and your worker send JSON over HTTP.

The types below use TypeScript notation. `?` marks a field you can omit.
`unknown` is any JSON value. Timestamps are RFC 3339 strings. Money values are
decimal strings.

The machine-readable source of truth is
[`schemas/protocol.schema.json`](../schemas/protocol.schema.json) for all types,
and [`schemas/worker.openapi.json`](../schemas/worker.openapi.json) for the
worker endpoint. To generate types instead of writing them, see
[Typed bindings](./40-typed-bindings.md).

## Delivery

The engine POSTs a `DecisionRequest` to your worker's endpoint.

| Header | Value |
| --- | --- |
| `Content-Type` | `application/json` |
| `Accept` | `text/event-stream, application/json` |
| `traceparent` | W3C trace context |
| `X-Substructure-Signature` | `sha256=<hex HMAC-SHA256 of the body>`, when there is a signing secret |

Answer with `application/json`, which holds a `DecisionResponse`. To stream,
answer with `text/event-stream`. See [Streaming](#streaming).

## Decision request

```typescript
type DecisionRequest = {
    session_id: string
    decision_id: string
    agent_id: string
    identity: WorkerIdentity
    trigger: Trigger
    proposed: DecisionResponse      // empty when the engine has no plan
    state: unknown              // your agent state, stored exactly as it is
    agent: AgentConfig | null
    calls: Call[]               // calls in flight
    pending_calls: number       // tool and sub-agent calls in flight
    messages: Message[]         // the active conversation path
    message_tree: MessageTree
    ancestry: string[]          // parent session ids, for sub-agents
    attempts: number            // how many times this decision was delivered
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
    anchor?: string             // the tree node where the call was requested
    name?: string               // tool calls; fetches: the connection
    arguments?: string          // tool calls
    handler?: Handler
    stream?: boolean            // llm calls
    agent_id?: string           // sub-agents
    tool_call_id?: string       // sub-agents: the call this child answers
}

type Handler = "server" | "worker" | "client"
```

`proposed` is what the engine plans to do next. Return it unchanged to accept
it. It is `null` when only the worker knows what to do, such as when it must run
one of its own tools.

The engine queues every call and decision first. It starts each one when the
things it waits for are ready, in the order they arrived. A `queued` call has
not started. It is waiting for a tool fetch the agent's config needs, for the
decision slot, for a paused branch, for a running turn, or for an entry ahead of
it in the queue. Every wait has a limit, because each of those things has a
deadline and always ends.

## Decision response

```typescript
type DecisionResponse = {
    messages?: DraftMessage[]   // messages to record
    actions?: Action[]          // what the engine should do next
    state?: unknown             // omitted or null keeps the current state
    agent?: AgentConfig         // omitted keeps the current config
}
```

To clear the state, write an empty value that is not null, such as `{}`.

## Triggers

```typescript
type Trigger =
    | { type: "session.start" }
    | {
          type: "client.messages"
          messages: DraftMessage[]  // the client's full view of the conversation
          new_from: number          // the index of the first new message
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
          request: unknown          // LlmRequest, or the provider's own body when format is set
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
    | { status: "valid"; value: unknown }     // an object that matches the tool's input schema
    | { status: "invalid"; value: unknown; error: string }
    | { status: "malformed"; error: string }  // not a JSON object

type ErrorCode =
    | "provider_error"
    | "rate_limited"
    | "refused"
    | "budget_exceeded"
    | "deadline_exceeded"

type ClientContext = {
    tools?: AgentTool[]         // tools the client runs, added to the proposed config
    context?: unknown[]
    state?: unknown
    forwarded_props?: unknown
}
```

Answer `tool.execute` with `tool.result` or `tool.error`. Answer `llm.execute`
with `llm.result` or `llm.error`, or stream the answer.

## Actions

```typescript
type Action =
    | {
          type: "llm.call"        // every field is optional; the engine fills a
          id?: string             // missing field from the agent config, then
          model?: string          // from its own defaults
          messages?: DraftMessage[]  // if you set messages, the engine adds no system prompt
          tools?: LlmTool[]
          temperature?: number
          max_completion_tokens?: number
          reasoning?: ReasoningConfig
          stream?: boolean        // default true
          retry?: RetryOverride
          handler?: "server" | "worker"  // default server
      }
    | {
          type: "tool.call"
          id?: string             // omitted: the engine creates one
          name: string
          arguments: unknown
          retry?: RetryOverride     // default: from the agent config, else from the engine
      }
    | {
          type: "tool.result"
          id?: string             // if you omit id and attempt, they come from
          attempt?: number        // the tool.execute trigger you answer
          result: unknown
      }
    | {
          type: "tool.error"
          id?: string
          attempt?: number
          error: string
          retryable?: boolean     // default false: no retry
          code?: ErrorCode
          detail?: unknown
      }
    | {
          type: "llm.result"
          id?: string
          attempt?: number
          response: unknown       // LlmResponse, or the provider's own response
      }                           // when the llm.execute carried a format
    | {
          type: "llm.error"
          id?: string
          attempt?: number
          error: string
          retryable?: boolean     // default false: no retry
          code?: ErrorCode
          detail?: unknown
      }
    | {
          type: "sub_agent.spawn"
          session_id: string
          agent_id: string
          tool_call_id: string    // the model's tool call that this child answers
          message?: DraftMessage  // the child's first message, sent once it exists
          retry?: RetryOverride
      }
    | { type: "message.send"; session_id: string; message: DraftMessage }
    | {
          type: "interrupt"
          interrupt_id?: string   // omitted: the engine creates one
          reason: string
          payload?: unknown
      }
    | { type: "done"; data?: unknown }
```

`{ "type": "llm.call" }` on its own prompts the model with the agent's config
over the current conversation.

## Agent config

```typescript
type AgentConfig = {
    model: string               // the only required field
    system?: string
    handler?: "server" | "worker"  // where LLM calls run; default server
    format?: "openai" | "anthropic"  // wire format for LLM calls the worker
                                     // makes; needs handler = worker
    retry?: RetryConfig
    tools?: AgentTool[]
    sub_agents?: SubAgent[]
    mcp?: McpServer[]
}

type AgentTool = {
    name: string
    description?: string
    input?: unknown             // JSON Schema for the arguments; omitted: no arguments
    output?: unknown            // JSON Schema the result must match; a result that
                                // does not match ends the call with an error
    handler?: "worker" | "client"  // default worker
}

type SubAgent = {
    id: string                  // the agent to start, and the tool name the model sees
    description?: string
}

type McpServer = {
    id: string                  // a connection the engine holds; never a URL
    tools?: McpTools            // omitted: every tool the connection offers
}

type McpTools = {
    include?: string[]          // globs over the tool's name on the connection
    exclude?: string[]
    read_only?: boolean         // these read the MCP annotations; a tool with
    non_destructive?: boolean   // no annotation fails them
    idempotent?: boolean
}

type RetryPolicy = {
    attempt_timeout_secs: number | null  // one attempt; null waits forever
    total_timeout_secs: number | null    // the whole effect; null has no limit
    max_attempts: number                 // attempts, not retries
    backoff_base_secs: number
    backoff_max_secs: number
}

type RetryOverride = {            // names only the fields it changes
    attempt_timeout_secs?: number
    total_timeout_secs?: number
    max_attempts?: number
    backoff_base_secs?: number
    backoff_max_secs?: number
}
type RetryConfig = {              // one override per kind; they stack
    default?: RetryOverride
    llm?: RetryOverride
    tool?: RetryOverride
    sub_agent?: RetryOverride
    connector?: RetryOverride
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

// A recorded message. DraftMessage has the same shape with an optional id.
// Use DraftMessage for a message the engine has not recorded yet.
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

The engine uses these shapes when the agent config sets no `format`.

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
    input?: unknown             // JSON Schema; omitted: the tool takes no arguments
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

What a client submits. The session comes from outside the input itself: from the
CLI's `--session`, or from the client API's request body.

```typescript
type ClientInput =
    | {
          type: "client.message"
          agent_id: string
          turn_id?: string        // idempotency key
          message: DraftMessage
          stream?: boolean
          queue?: boolean         // wait for the next turn instead of being refused
      }
    | {
          type: "client.messages"
          agent_id: string
          turn_id?: string
          messages: DraftMessage[]  // the client's full view of the conversation
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

`agent_id` routes the turn. It also creates the session if the session is new.
If you submit a `turn_id` that is complete, the engine returns that turn instead
of running it again.

### Queuing a message

A session runs one turn at a time. A submit that arrives while a turn runs is
refused with `turn_already_active`. `queue: true` changes that. The engine takes
the message, holds it, and starts it as the next turn when the running turn
completes. The reply carries `queued: true` while the message waits.

Only `client.message` and `client.append` take this flag. The engine still
refuses a full `client.messages` view and a `client.action`. A view ends open
client tool calls as it arrives, so the engine cannot hold it.

Queued turns run one at a time, in the order they arrived. The engine refuses a
`turn_id` that is running, queued, or complete, so a transport that retries
cannot ask the same question twice.

## Streaming

A worker that answers an `llm.execute` with `stream: true` can reply with
`text/event-stream` instead of JSON:

```
event: llm.token.delta
data: { "text": "hel" }

event: llm.token.delta
data: { "text": "lo" }

event: decision.result
data: { "actions": [{ "type": "llm.result", "response": … }] }
```

Each `llm.token.delta` frame carries a `StreamDelta`, or a provider stream event
when the `llm.execute` carried a `format`. The stream must end with one
`decision.result` frame that holds a `DecisionResponse`. It can end with a
`decision.error` frame instead, which holds `message` and `retryable`. The
default for `retryable` is `true`.

```typescript
type StreamDelta = {
    text?: string
    reasoning?: string
    tool_calls?: { id: string; name?: string; arguments?: string }[]
    finish_reason?: string
}
```
