---
title: Protocol
group: Reference
---

The engine POSTs a decision request to your worker. Your worker returns a
decision response. Both are JSON.

The types use TypeScript notation. `?` marks an optional field. `unknown` is any
JSON value. Timestamps are RFC 3339 strings. Money is a decimal string.

The machine-readable source is
[`schemas/protocol.schema.json`](../schemas/protocol.schema.json). See
[Typed bindings](./270-typed-bindings.md) to generate types for your language.

## A decision request

The model called your `get_weather` tool.

```json
{
  "session_id": "0193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b",
  "decision_id": "0193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a61",
  "agent_id": "oncall",
  "identity": { "id": "user_42" },
  "trigger": {
    "type": "tool.execute",
    "id": "call_abc",
    "name": "get_weather",
    "arguments": "{\"city\":\"Tokyo\"}",
    "input": { "status": "valid", "value": { "city": "Tokyo" } },
    "attempt": 1
  },
  "proposed": {},
  "state": { "turns": 2 },
  "agent": {
    "model": "claude-sonnet-4-5",
    "system": "You are the on-call assistant.",
    "tools": [{ "name": "get_weather", "description": "Get the weather." }]
  },
  "calls": [
    { "id": "call_abc", "kind": "tool_call", "status": "pending", "attempt": 1, "name": "get_weather" }
  ],
  "pending_calls": 1,
  "messages": [
    { "id": "m1", "role": "user", "content": "weather in Tokyo?" },
    { "id": "m2", "role": "assistant", "tool_calls": [ { "id": "call_abc", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}" } } ] }
  ],
  "message_tree": { "nodes": [], "head_id": "m2" },
  "ancestry": [],
  "attempts": 1,
  "deadline": "2026-08-05T18:04:00Z",
  "turn_id": "0193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a60"
}
```

`proposed` is empty here. Only your worker can run the tool.

## A decision response

Run the tool and return its result.

```json
{
  "actions": [
    { "type": "tool.result", "result": "It is clear in Tokyo." }
  ],
  "state": { "turns": 3 }
}
```

The engine records the result, ends the call, and prompts the model again.

## Request fields

```typescript
type DecisionRequest = {
    session_id: string
    decision_id: string
    agent_id: string
    identity: WorkerIdentity
    trigger: Trigger
    proposed: DecisionResponse
    state: unknown
    agent: AgentConfig | null
    calls: Call[]
    pending_calls: number
    messages: Message[]
    message_tree: MessageTree
    ancestry: string[]
    attempts: number
    deadline: string | null
    turn_id: string | null
}
```

| Field | Meaning |
| --- | --- |
| `session_id` | The conversation this decision belongs to. |
| `decision_id` | This decision. Only one is live at a time. |
| `agent_id` | Which agent to act as. Route on this when one worker serves several agents. |
| `identity` | The session's owner. The engine sets it once and vouches for it. |
| `trigger` | Why the engine is asking. See [Triggers](#triggers). |
| `proposed` | What the engine plans to do. Return it unchanged to accept it. Empty when only your worker knows what to do. |
| `state` | Your agent state, stored exactly as you wrote it. `null` when the session has none. |
| `agent` | The config resolved for the active path. `null` when nothing has set one. |
| `calls` | Tool, model, sub-agent, and connector calls in flight. |
| `pending_calls` | How many tool and sub-agent calls are in flight. |
| `messages` | The active conversation, root to head. This is what the model sees. |
| `message_tree` | Every branch. See [Conversations](./120-conversations.md). |
| `ancestry` | Parent session ids, for a sub-agent. Empty for a root session. |
| `attempts` | How many times the engine has delivered this decision. |
| `deadline` | When this attempt expires. |
| `turn_id` | The turn this decision belongs to. |

```typescript
type WorkerIdentity = {
    id?: string
    metadata?: Record<string, string>
}

type Call = {
    id: string                  // sub-agents: the child session. connectors: the connection
    kind: "tool_call" | "llm_call" | "sub_agent" | "connector_sync"
    status: "pending" | "completed" | "failed" | "retry_scheduled" | "queued"
    attempt: number
    deadline?: string
    anchor?: string             // the tree node where the call was requested
    name?: string
    arguments?: string
    handler?: "server" | "worker" | "client"
    stream?: boolean
    agent_id?: string
    tool_call_id?: string       // sub-agents: the call this child answers
}
```

A `queued` call has not started yet. Every wait has a deadline.

## Response fields

```typescript
type DecisionResponse = {
    messages?: DraftMessage[]
    actions?: Action[]
    state?: unknown
    agent?: AgentConfig
    channels?: Record<string, unknown>
}
```

| Field | Meaning |
| --- | --- |
| `messages` | Messages to record on the active path. |
| `actions` | What the engine does next. See [Actions](#actions). |
| `state` | Replaces your state. Omit or send `null` to keep it. Send `{}` to clear it. |
| `agent` | Replaces the config. Omit to keep it. |
| `channels` | How a frontend shows this decision, keyed by kind. The engine passes it through. See [Slack](./130-slack.md#customizing-what-the-bot-shows). |

An empty response `{}` writes nothing. Use it to leave a tool call open. See
[Deferred tools](./110-deferred-tools.md).

## Triggers

What fires, what the engine proposes, and what you return.

| Trigger | Fires when | `proposed` holds | Return |
| --- | --- | --- | --- |
| `session.start` | The session was created. | The config from the file. Empty when the section sets none. | An `agent` config. |
| `client.messages` | A client sent or edited messages. | Record the view, then call the model. | `proposed` |
| `client.action` | A client called a named action. | Empty. A Slack prompt button proposes `interrupt.resolve`. | Your own actions. |
| `tool.execute` | The model called your tool. | Empty. A `tool.error` when the arguments failed validation or the tool is undeclared. | `tool.result` or `tool.error`. |
| `tool.finished` | A tool call ended, after retries. | Record the result, then call the model. Waits when other calls are in flight. | `proposed` |
| `llm.execute` | The agent's LLM block is `type = "worker"`. | Empty. | `llm.result` or `llm.error`, or a stream. |
| `llm.finished` | A model call ended. | Record the reply, then start its tool calls or end the turn. | `proposed` |
| `sub_agent.finished` | A child session's turn ended. | Record the child's result as the tool result, then call the model. | `proposed` |
| `interrupt.resumed` | Someone resumed a paused branch. | Call the model again over the transcript. | `proposed` |
| `turn.finished` | A turn completed. Carries its cost and output. | `done`. | `proposed` |

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
          error?: ErrorInfo
      }
    | {
          type: "llm.execute"
          id: string
          request: unknown          // LlmRequest, or the provider's body when format is set
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
          error?: ErrorInfo
      }
    | {
          type: "sub_agent.finished"
          id: string                // the tool call
          ok: boolean
          session_id: string        // the child
          agent_id: string
          result?: string
          error?: ErrorInfo
      }
    | { type: "interrupt.resumed"; interrupt_id: string; payload?: unknown }
    | {
          type: "turn.finished"
          turn_id: string
          data?: unknown            // the turn's final output
          cost: string
          usage: Record<string, number>
      }

type ToolInput =
    | { status: "valid"; value: unknown }
    | { status: "invalid"; value: unknown; error: string }
    | { status: "malformed"; error: string }  // not a JSON object

type ClientContext = {
    tools?: AgentTool[]         // tools the client runs, added to the proposed config
    context?: unknown[]
    state?: unknown
    forwarded_props?: unknown
}
```

## Actions

| Action | Does | Answers |
| --- | --- | --- |
| `llm.call` | Make a model call. | Any trigger. |
| `tool.call` | Start a tool call. | Any trigger. |
| `tool.result` | End a tool call with a result. | `tool.execute` |
| `tool.error` | End a tool call with a failure. | `tool.execute` |
| `llm.result` | End a model call your worker made. | `llm.execute` |
| `llm.error` | End a model call with a failure. | `llm.execute` |
| `sub_agent.spawn` | Start a child session. | Any trigger. |
| `message.send` | Write a message into a session. | Any trigger. |
| `interrupt` | Pause the active branch. | Any trigger. |
| `interrupt.resolve` | Clear an open interrupt and resume. | Any trigger. |
| `connector.sync` | Fetch a connection's tools again. | Any trigger. |
| `done` | End the turn. | Any trigger. |

```typescript
type Action =
    | {
          type: "llm.call"        // every field is optional. the engine fills a
          id?: string             // missing field from the agent config, then
          llm?: string            // from its own defaults
          model?: string
          messages?: DraftMessage[]  // set these and the engine adds no system prompt
          tools?: LlmTool[]
          temperature?: number
          max_completion_tokens?: number
          reasoning?: ReasoningConfig
          stream?: boolean
          retry?: RetryOverride
      }
    | {
          type: "tool.call"
          id?: string             // omitted: the engine creates one
          name: string            // the name decides where the call runs
          arguments: unknown
          retry?: RetryOverride
      }
    | {
          type: "tool.result"
          id?: string             // id and attempt default to those of the
          attempt?: number        // tool.execute you answer
          result: unknown
      }
    | {
          type: "tool.error"
          id?: string
          attempt?: number
          error: string           // the model reads this. write it for the model
          retryable?: boolean     // default false
          code?: ErrorCode
          detail?: unknown
      }
    | {
          type: "llm.result"
          id?: string
          attempt?: number
          response: unknown       // LlmResponse, or the provider's response
      }                           // when the llm.execute carried a format
    | {
          type: "llm.error"
          id?: string
          attempt?: number
          error: string
          retryable?: boolean     // default false
          code?: ErrorCode
          detail?: unknown
      }
    | {
          type: "sub_agent.spawn"
          session_id: string
          agent_id: string
          tool_call_id: string    // the model's tool call this child answers
          message?: DraftMessage  // the child's first message
          retry?: RetryOverride
      }
    | { type: "message.send"; session_id: string; message: DraftMessage }
    | {
          type: "interrupt"
          interrupt_id?: string   // omitted: the engine creates one
          reason: string
          payload?: unknown
      }
    | { type: "interrupt.resolve"; interrupt_id: string; payload?: unknown }
    | { type: "connector.sync"; id: string }
    | { type: "done"; data?: unknown }
```

`{ "type": "llm.call" }` on its own prompts the model with the agent's config
over the current conversation.

`llm.call` takes an `llm` to send one call to another block. The config stays as
it is. See [LLMs](./70-llms.md).

`connector.sync` names a connection the config in force already names. Use it
after a person corrects a credential: the fetch runs again with a full retry
budget, and the tools it returns replace the ones the session held. A decision
that waits on the connection is delivered after the fetch settles. See
[Connectors](./40-connectors.md#when-a-credential-stops-working).

## Errors

```typescript
type ErrorInfo = {
    message: string             // one sentence, safe to show a person
    code: ErrorCode
    param?: string              // the input to fix, such as `agent.llm`
    detail?: unknown
}

type ErrorCode =
    | "provider_error"
    | "rate_limited"
    | "refused"
    | "budget_exceeded"
    | "deadline_exceeded"
```

`code` describes a failure. `retryable` decides whether the engine tries again.
See [Retries](./210-retries.md).

## Agent config

```typescript
type AgentConfig = {
    model: string               // the only required field
    llm?: string                // the [llm.<id>] block calls run on
    system?: string
    handler?: "server" | "worker"    // where model calls run. default server
    format?: "openai" | "anthropic"  // wire format when handler is worker
    retry?: RetryConfig
    tools?: AgentTool[]
    sub_agents?: SubAgent[]
    mcp?: McpServer[]
}

type AgentTool = {
    name: string
    description?: string
    input?: unknown             // JSON Schema for the arguments
    output?: unknown            // JSON Schema the result must match
    handler?: "worker" | "client"  // default worker
}

type SubAgent = {
    id: string                  // the agent to start, and the tool name the model sees
    description?: string
}

type McpServer = {
    id: string                  // a connection the engine holds. never a URL
    tools?: McpTools            // omitted: every tool the connection offers
}

type McpTools = {
    include?: string[]          // globs over the tool's name on the connection
    exclude?: string[]
    read_only?: boolean         // these read the MCP annotations
    non_destructive?: boolean
    idempotent?: boolean
    discovery?: "all" | "search" // omitted: "all". search offers two tools for the connection
}
```

An `[agent.<id>]` section in `substructure.toml` uses these same names. See
[Config](./220-config.md).

## Retries

```typescript
type RetryPolicy = {
    attempt_timeout_secs: number | null  // one attempt. null waits forever
    total_timeout_secs: number | null    // the whole effect. null has no limit
    max_attempts: number                 // attempts, not retries
    backoff_base_secs: number
    backoff_max_secs: number
}

type RetryOverride = {          // names only the fields it changes
    attempt_timeout_secs?: number
    total_timeout_secs?: number
    max_attempts?: number
    backoff_base_secs?: number
    backoff_max_secs?: number
}

type RetryConfig = {            // one override per kind. they stack
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

// A recorded message. DraftMessage is the same shape with an optional id.
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

## Model requests and responses

The engine uses these shapes when the config sets no `format`.

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
    input?: unknown             // JSON Schema
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

What a client submits. The session comes from outside the input: from the CLI's
`--session`, or from the request body.

```typescript
type ClientInput =
    | {
          type: "client.message"
          agent_id: string
          turn_id?: string        // idempotency key
          message: DraftMessage
          stream?: boolean
          queue?: boolean         // wait for the next turn
      }
    | {
          type: "client.messages"
          agent_id: string
          turn_id?: string
          messages: DraftMessage[]  // the client's full view
          stream?: boolean
          client?: ClientContext
      }
    | {
          type: "client.append"
          agent_id: string
          turn_id?: string
          messages: DraftMessage[]  // added at the head, never branching
          stream?: boolean
          client?: ClientContext
          queue?: boolean
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

`agent_id` routes the turn. It creates the session when the session is new. A
`turn_id` that is already complete returns that turn.

### Three ways to send messages

| Input | Effect |
| --- | --- |
| `client.message` | Adds one message at the head. |
| `client.append` | Adds several messages at the head. Use it to sync an outside conversation. |
| `client.messages` | Replaces the conversation with your view, and branches where the two differ. |

`client.append` never branches. The engine composes it against the active path
when it delivers it, and drops any message whose id it already recorded.

### Queuing

A session runs one turn at a time. A submit that arrives during a turn is
refused with `turn_already_active`.

Set `queue: true` to wait instead. The engine holds the message and starts it
when the running turn completes. The reply carries `queued: true` while it
waits.

Only `client.message` and `client.append` take this flag. Queued turns run in
the order they arrived.

## Delivery

The engine POSTs to your worker's URL.

| Header | Value |
| --- | --- |
| `Content-Type` | `application/json` |
| `Accept` | `text/event-stream, application/json` |
| `traceparent` | W3C trace context |
| `X-Substructure-Signature` | `sha256=<hex HMAC-SHA256 of the body>`, when the agent has a signing secret |

Answer with `application/json` holding a `DecisionResponse`. To stream, answer
with `text/event-stream`.

## Streaming

A worker that answers an `llm.execute` with `stream: true` can reply with
`text/event-stream`.

```
event: llm.token.delta
data: { "text": "hel" }

event: llm.token.delta
data: { "text": "lo" }

event: decision.result
data: { "actions": [{ "type": "llm.result", "response": … }] }
```

Each `llm.token.delta` carries a `StreamDelta`, or a provider stream event when
the `llm.execute` carried a `format`. The stream ends with one `decision.result`
frame holding a `DecisionResponse`. It can end with a `decision.error` frame
instead, holding `message` and `retryable`. `retryable` defaults to `true`.

```typescript
type StreamDelta = {
    text?: string
    reasoning?: string
    tool_calls?: { id: string; name?: string; arguments?: string }[]
    finish_reason?: string
}
```

## Next

- [Workers](./50-workers.md): the code that answers these.
- [Config](./220-config.md): the same types, declared in the file.
- [Events](./240-events.md): what the engine streams to clients.
- [REST API](./250-api.md): the endpoints clients call.
