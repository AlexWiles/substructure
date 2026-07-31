// Code generated from JSON Schema using quicktype. DO NOT EDIT.
// To parse and unparse this JSON data, add this code to your project and do:
//
//    protocol, err := UnmarshalProtocol(bytes)
//    bytes, err = protocol.Marshal()

package main

import "bytes"
import "errors"
import "time"

import "encoding/json"

func UnmarshalProtocol(data []byte) (Protocol, error) {
	var r Protocol
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *Protocol) Marshal() ([]byte, error) {
	return json.Marshal(r)
}

// One entry point per wire surface; every protocol type is named under $defs.
type Protocol struct {
	ClientInput         *ClientInput         `json:"client_input,omitempty"`
	ClientPayload       *ClientPayload       `json:"client_payload,omitempty"`
	DecisionRequest     *DecisionRequest     `json:"decision_request,omitempty"`
	DecisionResponse    *DecisionResponse    `json:"decision_response,omitempty"`
	InterruptPayload    *InterruptPayload    `json:"interrupt_payload,omitempty"`
	InterruptResolution *InterruptResolution `json:"interrupt_resolution,omitempty"`
	StreamDelta         *StreamDelta         `json:"stream_delta,omitempty"`
	TokenDelta          *TokenDelta          `json:"token_delta,omitempty"`
}

// Everything a client can send on the input surface: submit a message / a full view / an
// append batch / a named action, resume an interrupt, or settle a client tool. A flat,
// internally-tagged union — its seven tags produce serde's "unknown variant, expected one
// of …" error for free. `Runtime::handle_client_input` is the single seam that dispatches
// it (mirroring `resolve_response` on the worker side).
//
// Addressing lives where it is meaningful, not in a shared envelope: `agent_id` (routes
// the turn, creating the session if new) and the optional idempotency `turn_id` are
// fields of the four submit variants only. A resume/settle addresses an interrupt/effect
// id and continues whatever turn is active, so it carries neither — misplacing them is
// unrepresentable rather than rejected. `session_id` is the one universal address and
// rides the envelope. A submit's body rebuilds a [`ClientPayload`] at the seam.
//
// The body of an interrupt resume: which interrupt, and the payload delivered
// to the worker. Shared by the [`ClientInput::InterruptResume`] input and the
// [`DecisionTrigger::InterruptResumed`] trigger.
type ClientInput struct {
	AgentID *string       `json:"agent_id,omitempty"`
	Message *DraftMessage `json:"message,omitempty"`
	// Hold this message for the next turn instead of refusing it when one
	// is already running. Off by default: rejection stays the contract for
	// a plain submitter, and queuing is declared intent.
	//
	// Hold this batch for the next turn instead of refusing it when one is
	// already running. Off by default: rejection stays the contract for a
	// plain submitter, and queuing is declared intent.
	Queue       *bool           `json:"queue,omitempty"`
	Stream      *bool           `json:"stream,omitempty"`
	TurnID      *string         `json:"turn_id"`
	Type        ClientInputType `json:"type"`
	Client      *ClientContext  `json:"client,omitempty"`
	Messages    []DraftMessage  `json:"messages,omitempty"`
	Args        interface{}     `json:"args"`
	Name        *string         `json:"name,omitempty"`
	InterruptID *string         `json:"interrupt_id,omitempty"`
	Payload     interface{}     `json:"payload"`
	Attempt     *int64          `json:"attempt"`
	ID          *string         `json:"id,omitempty"`
	Result      interface{}     `json:"result"`
	Error       *string         `json:"error,omitempty"`
	Retryable   *bool           `json:"retryable,omitempty"`
}

// Inputs a client declares on its run (the AG-UI `tools`/`context`/`state`/
// `forwardedProps`), forwarded to the worker on the `client.messages` decision.
// `tools` are the browser's frontend tools, normalized to client-handled
// [`AgentTool`]s; the engine layers them onto the proposed config by default, and
// the worker may override (e.g. whitelist) by returning its own `agent`.
//
// Inputs the client declared on its run; the engine layers `client.tools`
// onto the proposed config by default.
type ClientContext struct {
	Context        []interface{} `json:"context,omitempty"`
	ForwardedProps interface{}   `json:"forwarded_props"`
	State          interface{}   `json:"state"`
	Tools          []AgentTool   `json:"tools,omitempty"`
}

// A function tool the agent offers. The model-facing contract is
// `name`/`description`/`input`/`output`; `handler` selects where a call runs —
// `Some(Client)` ⇒ client-executed, absent ⇒ worker-executed (the default).
// `server` is invalid for tools: engine-executed tools come from a connector,
// which a worker declares by id rather than by tool.
type AgentTool struct {
	Description *string     `json:"description,omitempty"`
	Handler     *Handler    `json:"handler"`
	Input       interface{} `json:"input"`
	Name        string      `json:"name"`
	Output      interface{} `json:"output"`
}

// The wire form of a [`Message`]: `id` is optional because a client-submitted or
// worker-authored message is not yet recorded. `record`/`rerecord`
// (`runtime::session::wire`) are the seams that lower it to the internal
// [`Message`] (id always present) at recording time.
type DraftMessage struct {
	Content    *Content          `json:"content"`
	ID         *string           `json:"id"`
	Name       *string           `json:"name"`
	Role       Role              `json:"role"`
	ToolCallID *string           `json:"tool_call_id"`
	ToolCalls  []ToolCallElement `json:"tool_calls"`
}

type ContentPart struct {
	Text       *string         `json:"text,omitempty"`
	Type       ContentPartType `json:"type"`
	ImageURL   *ImageURLClass  `json:"image_url,omitempty"`
	File       *FileData       `json:"file,omitempty"`
	InputAudio *AudioData      `json:"input_audio,omitempty"`
	VideoURL   *VideoURLClass  `json:"video_url,omitempty"`
}

type FileData struct {
	FileData string `json:"file_data"`
	Filename string `json:"filename"`
}

type ImageURLClass struct {
	URL string `json:"url"`
}

type AudioData struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type VideoURLClass struct {
	URL string `json:"url"`
}

type ToolCallElement struct {
	Function ToolCallFunction `json:"function"`
	ID       string           `json:"id"`
	Type     string           `json:"type"`
}

type ToolCallFunction struct {
	Arguments string `json:"arguments"`
	Name      string `json:"name"`
}

// The client→engine inbound *submit* wire form: an untrusted client submits a message,
// its full conversation view, an append batch, or a named action. Lowered to domain events
// at the
// `SubmitClientPayload` command seam (`runtime::session::command`); never persisted
// as-is. Carried verbatim inside [`ClientInput`], which is the full client input
// surface.
//
// The body of a `client.message`: one message, optionally streamed.
//
// The body of a `client.messages`: the client's full conversation view, optionally
// streamed.
//
// The body of a `client.append`: messages appended at the session head. The
// view is composed against the active path at delivery, so a queued append
// lands after whatever turn beat it — it can never fork the tree. Messages
// whose ids are already recorded are dropped.
//
// The payload of a `client.action`: a named action with optional JSON args.
type ClientPayload struct {
	Message  *DraftMessage     `json:"message,omitempty"`
	Stream   *bool             `json:"stream,omitempty"`
	Type     ClientPayloadType `json:"type"`
	Client   *ClientContext    `json:"client,omitempty"`
	Messages []DraftMessage    `json:"messages,omitempty"`
	Args     interface{}       `json:"args"`
	Name     *string           `json:"name,omitempty"`
}

type DecisionRequest struct {
	// The agent config resolved for the active path (`null` when none is set).
	Agent       *AgentConfig   `json:"agent"`
	AgentID     string         `json:"agent_id"`
	Ancestry    []string       `json:"ancestry"`
	Attempts    int64          `json:"attempts"`
	Calls       []Effect       `json:"calls"`
	Deadline    *time.Time     `json:"deadline"`
	DecisionID  string         `json:"decision_id"`
	Identity    WorkerIdentity `json:"identity"`
	MessageTree MessageTree    `json:"message_tree"`
	Messages    []Message      `json:"messages"`
	// Count of in-flight `tool_call`/`sub_agent` calls.
	PendingCalls int64 `json:"pending_calls"`
	// The engine's default continuation for `trigger` (empty when it needs
	// worker knowledge). Advisory: accept by echoing it as the decision.
	Proposed  DecisionResponse `json:"proposed"`
	SessionID string           `json:"session_id"`
	State     interface{}      `json:"state"`
	Trigger   DecisionTrigger  `json:"trigger"`
	TurnID    *string          `json:"turn_id"`
}

// A declared agent identity — the same shape whether it is written in an
// `[agent.<id>]` section or returned by a worker.
//
// `llm` names the `[llm.*]` block every proposed call runs on, and so decides
// both the venue (the engine with a vendor key, or the agent's own worker) and
// the wire shape of a worker-run call. It is effectively required: a config
// that names none fails when the engine resolves a call against it.
type AgentConfig struct {
	// The `[llm.*]` block this agent's calls run on.
	Llm *string `json:"llm"`
	// MCP servers the agent draws tools from. The engine resolves each against
	// its connection registry into [`ConnectorTool`]s the model sees alongside
	// `tools`. Like `sub_agents`, these are never merged into `tools` — the
	// worker declares the server, not its tools.
	//
	// A second protocol gets its own field rather than a `type` tag here: its
	// filter would not be this one (MCP annotations mean nothing to an A2A
	// agent), and a union of conditionally-valid fields generates badly in the
	// Go and Python bindings.
	MCP    []MCPServer  `json:"mcp,omitempty"`
	Model  string       `json:"model"`
	Retry  *RetryPolicy `json:"retry"`
	Stream *bool        `json:"stream,omitempty"`
	// Sub-agents the model can delegate to. Presented to the model as tools (by
	// id) alongside `tools`, but each call spawns a child session rather than
	// executing a function.
	SubAgents []SubAgentElement `json:"sub_agents,omitempty"`
	System    *string           `json:"system"`
	// Worker- or client-executed tools the model can call.
	Tools []AgentTool `json:"tools,omitempty"`
}

// An MCP server the agent draws tools from. `id` resolves against the engine's
// connection registry — locally from `[mcp]` in `substructure.toml`, in the
// cloud from the connections an admin granted this app. The worker never names
// a URL or a credential.
type MCPServer struct {
	ID string `json:"id"`
	// Narrows what the model sees. Absent ⇒ every tool the connection grants.
	Tools *MCPTools `json:"tools"`
}

// An MCP server's tool filter. Applied in order — capability predicates, then
// `include`, then `exclude` — and only ever narrowing, so a filter can never
// widen what the connection grants.
//
// `include`/`exclude` are globs matched against the tool's name on the
// connection, the name its own documentation uses, not the prefixed name the
// model sees. Capability predicates read the MCP annotations; a tool that
// carries none fails the predicate, so an unannotated server yields nothing
// under `read_only` rather than silently passing everything through.
type MCPTools struct {
	Exclude        []string `json:"exclude,omitempty"`
	Idempotent     *bool    `json:"idempotent"`
	Include        []string `json:"include,omitempty"`
	NonDestructive *bool    `json:"non_destructive"`
	ReadOnly       *bool    `json:"read_only"`
}

// Fully-resolved retry policy — no optional fields. Stored on call state and
// read directly by retry logic.
type RetryPolicy struct {
	BackoffBaseSecs int64  `json:"backoff_base_secs"`
	BackoffMaxSecs  int64  `json:"backoff_max_secs"`
	MaxRetries      int64  `json:"max_retries"`
	TimeoutSecs     *int64 `json:"timeout_secs"`
}

// A sub-agent the model can delegate to. Named by `id` (the child agent to spawn,
// and the tool name the model calls); its model-facing input is the conventional
// single-`message` delegation schema.
type SubAgentElement struct {
	Description *string `json:"description,omitempty"`
	ID          string  `json:"id"`
}

// An in-flight effect (Pending or RetryScheduled) surfaced on each worker decision.
// A flat envelope plus kind-specific fields: a tool call's
// `name`/`arguments`/`handler`, an LLM call's `handler`/`stream`, a
// sub-agent's `agent_id`/`session_id`. A connector sync carries none — its
// `id` is the connection being fetched.
type Effect struct {
	AgentID *string `json:"agent_id"`
	// The tree node the effect was requested at.
	Anchor    *string      `json:"anchor"`
	Arguments *string      `json:"arguments"`
	Attempt   int64        `json:"attempt"`
	Deadline  *time.Time   `json:"deadline"`
	Handler   *Handler     `json:"handler"`
	ID        string       `json:"id"`
	Kind      EffectKind   `json:"kind"`
	Name      *string      `json:"name"`
	Status    EffectStatus `json:"status"`
	Stream    *bool        `json:"stream"`
	// The model tool call a delegation answers; its own `id` is the child session.
	ToolCallID *string `json:"tool_call_id"`
}

// The owner as delivered to the worker on `DecisionRequest.identity`: the
// subject and its metadata, without the tenant. The tenant scopes the session
// internally but is not sent to the worker.
type WorkerIdentity struct {
	ID       *string           `json:"id"`
	Metadata map[string]string `json:"metadata"`
}

type MessageTree struct {
	HeadID *string `json:"head_id"`
	Nodes  []Node  `json:"nodes"`
}

type Node struct {
	Message  Message `json:"message"`
	ParentID *string `json:"parent_id"`
}

type Message struct {
	Content    *Content          `json:"content"`
	ID         string            `json:"id"`
	Name       *string           `json:"name"`
	Role       Role              `json:"role"`
	ToolCallID *string           `json:"tool_call_id"`
	ToolCalls  []ToolCallElement `json:"tool_calls,omitempty"`
}

// The engine's default continuation for `trigger` (empty when it needs
// worker knowledge). Advisory: accept by echoing it as the decision.
//
// A decision: the messages/actions to author, plus optional state/agent writes.
// The worker returns one; the engine also proposes one as the default
// continuation (`DecisionRequest::proposed`), which the worker echoes or amends.
type DecisionResponse struct {
	Actions []DecisionAction `json:"actions,omitempty"`
	// A new agent config write; omitted keeps the current config.
	Agent    *AgentConfig   `json:"agent"`
	Messages []DraftMessage `json:"messages,omitempty"`
	// Omitted or `null` keeps the current state; clear with a non-null empty value.
	State interface{} `json:"state"`
}

// The action a worker authors on the wire. Mirrors the internal `Action`, but a
// settle's effect id may be omitted: on the sync/pull paths the answered
// `*.execute` trigger names it, so echoing it is redundant. `resolve_response`
// (`runtime::session::wire`) turns this into the internal `Action` (id always
// present) at the transport boundary.
//
// A flat, all-optional LLM request. `id` omitted ⇒ the engine mints one; it
// becomes the assistant node's id. Omitted fields are filled from the agent
// config (merge source), then engine defaults; `messages` omitted ⇒
// `[config.system?] + the decision's declared view`. Explicit `messages`
// suppress system injection. A bare `{"type":"llm.call"}` prompts per the
// agent's identity over the current view.
//
// `id` omitted ⇒ the engine mints one (LLM-driven tools carry the model's id).
//
// There is no `handler`: where a call runs follows from its name. A tool
// resolved from a connector runs on the engine, a tool declared
// `handler: client` runs on the client, and anything else runs on the
// worker. The engine already knows all three, so asking the worker to
// restate it only creates a way for the two to disagree.
//
// `id`/`attempt` omitted ⇒ taken from the answering `tool.execute` trigger,
// fencing the result to the attempt that ran.
//
// `id`/`attempt` omitted ⇒ taken from the answering `llm.execute` trigger,
// fencing the result to the attempt that ran.
//
// `interrupt_id` omitted ⇒ the engine mints one to correlate the later resume.
type DecisionAction struct {
	ID *string `json:"id"`
	// The `[llm.*]` block this call runs on; omitted ⇒ the merge source
	// config's `llm`. Naming a different block moves one call to another
	// venue or vendor.
	Llm                 *string            `json:"llm"`
	MaxCompletionTokens *int64             `json:"max_completion_tokens"`
	Messages            []DraftMessage     `json:"messages"`
	Model               *string            `json:"model"`
	Reasoning           *ReasoningConfig   `json:"reasoning"`
	Retry               *RetryPolicy       `json:"retry"`
	Stream              *bool              `json:"stream"`
	Temperature         *float64           `json:"temperature"`
	Tools               []LlmTool          `json:"tools"`
	Type                DecisionActionType `json:"type"`
	Arguments           interface{}        `json:"arguments"`
	Name                *string            `json:"name,omitempty"`
	Attempt             *int64             `json:"attempt"`
	Result              interface{}        `json:"result"`
	// A neutral `LlmResponse`, or the provider's native response when the
	// answered `llm.execute` carried a `format`.
	Response interface{} `json:"response"`
	Code     *ErrorCode  `json:"code"`
	Detail   interface{} `json:"detail"`
	Error    *string     `json:"error,omitempty"`
	// Omitted ⇒ terminal.
	Retryable *bool   `json:"retryable,omitempty"`
	AgentID   *string `json:"agent_id,omitempty"`
	SessionID *string `json:"session_id,omitempty"`
	// The model tool-call this delegation answers — always required.
	ToolCallID  *string       `json:"tool_call_id,omitempty"`
	Message     *DraftMessage `json:"message,omitempty"`
	InterruptID *string       `json:"interrupt_id"`
	Payload     interface{}   `json:"payload"`
	Reason      *string       `json:"reason,omitempty"`
	Data        interface{}   `json:"data"`
}

type ReasoningConfig struct {
	Effort    *ReasoningEffort `json:"effort"`
	Enabled   *bool            `json:"enabled"`
	Exclude   *bool            `json:"exclude"`
	MaxTokens *int64           `json:"max_tokens"`
}

// A tool's declared contract: flat on the wire. Providers that need
// OpenAI-style `{"type": "function", "function": {…}}` nesting re-wrap at
// their own boundary.
type LlmTool struct {
	Description string `json:"description"`
	// JSON Schema for the tool's arguments; omitted declares a no-argument
	// tool. The engine validates each call's arguments against it and hands
	// providers their native form.
	Input interface{} `json:"input"`
	Name  string      `json:"name"`
	// JSON Schema the settled result must satisfy; never sent to the model.
	// A violating result settles as a terminal tool error.
	Output interface{} `json:"output"`
}

// The trigger a worker sees on the wire — the materialized projection of the
// engine's internal decision trigger. It has no `ClientMessage`: a bare client
// message is always materialized to `ClientTranscript` by `to_wire_trigger`
// (`runtime::session::wire`) before delivery, so an unmaterialized message can
// never reach a worker.
//
// The first decision of every session; carries no proposal.
//
// Answer with `tool.result`/`tool.error`.
//
// Answer with `llm.result`/`llm.error`.
//
// The body of an interrupt resume: which interrupt, and the payload delivered
// to the worker. Shared by the [`ClientInput::InterruptResume`] input and the
// [`DecisionTrigger::InterruptResumed`] trigger.
//
// Fired after a turn completes, carrying its final output; blocks the session
// going idle until answered. Echo the proposed `done` to finalize.
type DecisionTrigger struct {
	Type DecisionTriggerType `json:"type"`
	// Inputs the client declared on its run; the engine layers `client.tools`
	// onto the proposed config by default.
	Client    *ClientContext `json:"client,omitempty"`
	Messages  []DraftMessage `json:"messages,omitempty"`
	NewFrom   *int64         `json:"new_from,omitempty"`
	Args      interface{}    `json:"args"`
	Name      *string        `json:"name,omitempty"`
	Arguments *string        `json:"arguments,omitempty"`
	Attempt   *int64         `json:"attempt,omitempty"`
	Deadline  *time.Time     `json:"deadline"`
	ID        *string        `json:"id,omitempty"`
	// The engine's classification of `arguments` against the tool's
	// declared `input` schema: `valid` (with the parsed `value`),
	// `invalid` (value plus the violation), or `malformed` (not a JSON
	// object). Always on the wire.
	Input  *ToolInput `json:"input,omitempty"`
	Error  *string    `json:"error"`
	Ok     *bool      `json:"ok,omitempty"`
	Result *string    `json:"result"`
	Format *LlmFormat `json:"format"`
	// The neutral `LlmRequest` JSON, or the provider's native request body
	// when `format` is set.
	Request     interface{}   `json:"request"`
	Stream      *bool         `json:"stream,omitempty"`
	Code        *ErrorCode    `json:"code"`
	Cost        *string       `json:"cost"`
	Detail      interface{}   `json:"detail"`
	Message     *DraftMessage `json:"message"`
	Truncated   *bool         `json:"truncated,omitempty"`
	Usage       interface{}   `json:"usage"`
	AgentID     *string       `json:"agent_id,omitempty"`
	SessionID   *string       `json:"session_id,omitempty"`
	InterruptID *string       `json:"interrupt_id,omitempty"`
	Payload     interface{}   `json:"payload"`
	Data        interface{}   `json:"data"`
	TurnID      *string       `json:"turn_id,omitempty"`
}

// The engine's classification of `arguments` against the tool's
// declared `input` schema: `valid` (with the parsed `value`),
// `invalid` (value plus the violation), or `malformed` (not a JSON
// object). Always on the wire.
//
// The engine's classification of a tool call's arguments, delivered on the
// `tool.execute` trigger alongside the raw `arguments` string. Always on the
// wire — absence never carries meaning.
//
// Parsed and, when the tool declares an `input` schema, conforming to it.
// `value` is exactly the parsed `arguments` — the engine never mutates it.
//
// Parsed to an object that violates the declared `input` schema.
//
// Not a JSON object: malformed JSON or a non-object value.
type ToolInput struct {
	Status Status      `json:"status"`
	Value  interface{} `json:"value"`
	Error  *string     `json:"error,omitempty"`
}

// An interrupt payload following the AG-UI Interrupt shape (spec spelling;
// `id` and `reason` live on the interrupt itself).
type InterruptPayload struct {
	// RFC 3339; display only until engine TTLs land.
	ExpiresAt *string `json:"expiresAt"`
	// Markdown; channels down-convert. Without it, channels fall back to
	// the interrupt's `reason`.
	Message *string `json:"message"`
	// Free-form, delivered to clients verbatim. `metadata.options`
	// ([`InterruptOption`] list) renders as Slack buttons.
	Metadata interface{} `json:"metadata"`
	// JSON Schema for the expected resolution payload.
	ResponseSchema interface{} `json:"responseSchema"`
	// Binds the interrupt to a prior tool call.
	ToolCallID *string `json:"toolCallId"`
}

// A channel-authored resume payload: the AG-UI resume shape
// (`{status, payload}`) plus a provenance stamp.
type InterruptResolution struct {
	Payload   interface{}         `json:"payload"`
	Responder *InterruptResponder `json:"responder"`
	Status    ResumeStatus        `json:"status"`
}

// Who resolved it, stamped by the channel — never by the requester.
type InterruptResponder struct {
	// The channel kind, e.g. `slack`, `ag-ui`.
	Channel string `json:"channel"`
	// The chosen option's label, when the resolution was a pick.
	Label *string `json:"label"`
	// Channel-native user id.
	User *string `json:"user"`
}

type StreamDelta struct {
	FinishReason *string         `json:"finish_reason"`
	Reasoning    *string         `json:"reasoning"`
	Text         *string         `json:"text"`
	ToolCalls    []ToolCallChunk `json:"tool_calls,omitempty"`
}

type ToolCallChunk struct {
	Arguments *string `json:"arguments"`
	ID        string  `json:"id"`
	Name      *string `json:"name"`
}

type TokenDelta struct {
	AgentID      string  `json:"agent_id"`
	Attempt      int64   `json:"attempt"`
	CallID       string  `json:"call_id"`
	FinishReason *string `json:"finish_reason"`
	Reasoning    *string `json:"reasoning"`
	// Transport routing key.
	RootSessionID string `json:"root_session_id"`
	// Per-call counter, distinct from event-store sequence.
	Seq int64 `json:"seq"`
	// May be a sub-agent of root.
	SessionID string `json:"session_id"`
	// Tenant isolation key — subscribers must match.
	TenantID  string          `json:"tenant_id"`
	Text      *string         `json:"text"`
	ToolCalls []ToolCallChunk `json:"tool_calls"`
	TurnID    *string         `json:"turn_id"`
}

// Server-side executor resolves the provider or connection and makes the call.
//
// Dispatched to the work queue for the worker to execute.
//
// Executed by the client. Session goes Idle while waiting (tools only).
type Handler string

const (
	Client Handler = "client"
	Server Handler = "server"
	Worker Handler = "worker"
)

type ContentPartType string

const (
	File       ContentPartType = "file"
	ImageURL   ContentPartType = "image_url"
	InputAudio ContentPartType = "input_audio"
	Text       ContentPartType = "text"
	VideoURL   ContentPartType = "video_url"
)

type Role string

const (
	Assistant Role = "assistant"
	System    Role = "system"
	Tool      Role = "tool"
	User      Role = "user"
)

type ClientInputType string

const (
	InterruptResume      ClientInputType = "interrupt.resume"
	PurpleClientAction   ClientInputType = "client.action"
	PurpleClientAppend   ClientInputType = "client.append"
	PurpleClientMessage  ClientInputType = "client.message"
	PurpleClientMessages ClientInputType = "client.messages"
	PurpleToolError      ClientInputType = "tool.error"
	PurpleToolResult     ClientInputType = "tool.result"
)

type ClientPayloadType string

const (
	FluffyClientAction   ClientPayloadType = "client.action"
	FluffyClientAppend   ClientPayloadType = "client.append"
	FluffyClientMessage  ClientPayloadType = "client.message"
	FluffyClientMessages ClientPayloadType = "client.messages"
)

// What kind of work an effect is. One enum for the wire and for the engine's
// own scheduling: a decision and a turn's end queue beside the calls and are
// swept the same way, so they are kinds too. Neither ever appears on an
// [`Effect`] — a decision rides the decision list, a turn end has no record.
//
// Fetching one connection's tool list. Its `id` is the connection id.
//
// A worker decision.
//
// The turn's completion, dependent on its `turn.finished` finalizer
// decision settling. Carries the turn id; the frozen output lives in the
// session's `finalizing`. Never swept: it has no deadline of its own.
type EffectKind string

const (
	ConnectorSync EffectKind = "connector_sync"
	Decision      EffectKind = "decision"
	LlmCall       EffectKind = "llm_call"
	SubAgent      EffectKind = "sub_agent"
	ToolCall      EffectKind = "tool_call"
	TurnEnd       EffectKind = "turn_end"
)

type EffectStatus string

const (
	Completed      EffectStatus = "completed"
	Failed         EffectStatus = "failed"
	Pending        EffectStatus = "pending"
	Queued         EffectStatus = "queued"
	RetryScheduled EffectStatus = "retry_scheduled"
)

type ErrorCode string

const (
	BudgetExceeded   ErrorCode = "budget_exceeded"
	DeadlineExceeded ErrorCode = "deadline_exceeded"
	ProviderError    ErrorCode = "provider_error"
	RateLimited      ErrorCode = "rate_limited"
	Refused          ErrorCode = "refused"
)

type ReasoningEffort string

const (
	High    ReasoningEffort = "high"
	Low     ReasoningEffort = "low"
	Medium  ReasoningEffort = "medium"
	Minimal ReasoningEffort = "minimal"
	None    ReasoningEffort = "none"
	Xhigh   ReasoningEffort = "xhigh"
)

type DecisionActionType string

const (
	Done             DecisionActionType = "done"
	FluffyToolError  DecisionActionType = "tool.error"
	FluffyToolResult DecisionActionType = "tool.result"
	Interrupt        DecisionActionType = "interrupt"
	LlmError         DecisionActionType = "llm.error"
	LlmResult        DecisionActionType = "llm.result"
	MessageSend      DecisionActionType = "message.send"
	SubAgentSpawn    DecisionActionType = "sub_agent.spawn"
	TypeLlmCall      DecisionActionType = "llm.call"
	TypeToolCall     DecisionActionType = "tool.call"
)

// OpenAI Chat Completions.
//
// Anthropic Messages API.
type LlmFormat string

const (
	Anthropic LlmFormat = "anthropic"
	Openai    LlmFormat = "openai"
)

type Status string

const (
	Invalid   Status = "invalid"
	Malformed Status = "malformed"
	Valid     Status = "valid"
)

type DecisionTriggerType string

const (
	InterruptResumed        DecisionTriggerType = "interrupt.resumed"
	LlmExecute              DecisionTriggerType = "llm.execute"
	LlmFinished             DecisionTriggerType = "llm.finished"
	SessionStart            DecisionTriggerType = "session.start"
	SubAgentFinished        DecisionTriggerType = "sub_agent.finished"
	TentacledClientAction   DecisionTriggerType = "client.action"
	TentacledClientMessages DecisionTriggerType = "client.messages"
	ToolExecute             DecisionTriggerType = "tool.execute"
	ToolFinished            DecisionTriggerType = "tool.finished"
	TurnFinished            DecisionTriggerType = "turn.finished"
)

type ResumeStatus string

const (
	Cancelled ResumeStatus = "cancelled"
	Resolved  ResumeStatus = "resolved"
)

type Content struct {
	ContentPartArray []ContentPart
	String           *string
}

func (x *Content) UnmarshalJSON(data []byte) error {
	x.ContentPartArray = nil
	object, err := unmarshalUnion(data, nil, nil, nil, &x.String, true, &x.ContentPartArray, false, nil, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
	}
	return nil
}

func (x *Content) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, nil, nil, x.String, x.ContentPartArray != nil, x.ContentPartArray, false, nil, false, nil, false, nil, true)
}

func unmarshalUnion(data []byte, pi **int64, pf **float64, pb **bool, ps **string, haveArray bool, pa interface{}, haveObject bool, pc interface{}, haveMap bool, pm interface{}, haveEnum bool, pe interface{}, nullable bool) (bool, error) {
	if pi != nil {
		*pi = nil
	}
	if pf != nil {
		*pf = nil
	}
	if pb != nil {
		*pb = nil
	}
	if ps != nil {
		*ps = nil
	}

	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return false, err
	}

	switch v := tok.(type) {
	case json.Number:
		if pi != nil {
			i, err := v.Int64()
			if err == nil {
				*pi = &i
				return false, nil
			}
		}
		if pf != nil {
			f, err := v.Float64()
			if err == nil {
				*pf = &f
				return false, nil
			}
			return false, errors.New("Unparsable number")
		}
		return false, errors.New("Union does not contain number")
	case float64:
		return false, errors.New("Decoder should not return float64")
	case bool:
		if pb != nil {
			*pb = &v
			return false, nil
		}
		return false, errors.New("Union does not contain bool")
	case string:
		if haveEnum {
			return false, json.Unmarshal(data, pe)
		}
		if ps != nil {
			*ps = &v
			return false, nil
		}
		return false, errors.New("Union does not contain string")
	case nil:
		if nullable {
			return false, nil
		}
		return false, errors.New("Union does not contain null")
	case json.Delim:
		if v == '{' {
			if haveObject {
				return true, json.Unmarshal(data, pc)
			}
			if haveMap {
				return false, json.Unmarshal(data, pm)
			}
			return false, errors.New("Union does not contain object")
		}
		if v == '[' {
			if haveArray {
				return false, json.Unmarshal(data, pa)
			}
			return false, errors.New("Union does not contain array")
		}
		return false, errors.New("Cannot handle delimiter")
	}
	return false, errors.New("Cannot unmarshal union")
}

func marshalUnion(pi *int64, pf *float64, pb *bool, ps *string, haveArray bool, pa interface{}, haveObject bool, pc interface{}, haveMap bool, pm interface{}, haveEnum bool, pe interface{}, nullable bool) ([]byte, error) {
	if pi != nil {
		return json.Marshal(*pi)
	}
	if pf != nil {
		return json.Marshal(*pf)
	}
	if pb != nil {
		return json.Marshal(*pb)
	}
	if ps != nil {
		return json.Marshal(*ps)
	}
	if haveArray {
		return json.Marshal(pa)
	}
	if haveObject {
		return json.Marshal(pc)
	}
	if haveMap {
		return json.Marshal(pm)
	}
	if haveEnum {
		return json.Marshal(pe)
	}
	if nullable {
		return json.Marshal(nil)
	}
	return nil, errors.New("Union must not be null")
}
