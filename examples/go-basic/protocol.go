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
	ClientInput      *ClientInput           `json:"client_input,omitempty"`
	ClientPayload    *ClientPayload         `json:"client_payload,omitempty"`
	DecisionRequest  *DecisionRequest       `json:"decision_request,omitempty"`
	DecisionResponse *DecisionResponseClass `json:"decision_response,omitempty"`
	StreamDelta      *StreamDelta           `json:"stream_delta,omitempty"`
	TokenDelta       *TokenDelta            `json:"token_delta,omitempty"`
}

// Everything a client can send on the input surface: submit a message / a full view / a
// named action, resume an interrupt, or settle a client tool. A flat, internally-tagged
// union — its six tags produce serde's "unknown variant, expected one of …" error for
// free. `Runtime::handle_client_input` is the single seam that dispatches it (mirroring
// `resolve_response` on the worker side).
//
// Addressing lives where it is meaningful, not in a shared envelope: `agent_id` (routes
// the turn, creating the session if new) and the optional idempotency `turn_id` are
// fields of the three submit variants only. A resume/settle addresses an interrupt/effect
// id and continues whatever turn is active, so it carries neither — misplacing them is
// unrepresentable rather than rejected. `session_id` is the one universal address and
// rides the envelope. A submit's body rebuilds a [`ClientPayload`] at the seam.
type ClientInput struct {
	AgentID     *string              `json:"agent_id,omitempty"`
	Message     *ClientInputMessage  `json:"message,omitempty"`
	Stream      *bool                `json:"stream,omitempty"`
	TurnID      *string              `json:"turn_id"`
	Type        ClientInputType      `json:"type"`
	Messages    []ClientInputMessage `json:"messages,omitempty"`
	Args        interface{}          `json:"args"`
	Name        *string              `json:"name,omitempty"`
	InterruptID *string              `json:"interrupt_id,omitempty"`
	Payload     interface{}          `json:"payload"`
	Attempt     *int64               `json:"attempt"`
	ID          *string              `json:"id,omitempty"`
	Result      interface{}          `json:"result"`
	Error       *string              `json:"error,omitempty"`
	Retryable   *bool                `json:"retryable,omitempty"`
}

// The wire form of a [`Message`]: `id` is optional because a client-submitted or
// worker-authored message is not yet recorded. `record`/`rerecord`
// (`runtime::session::wire`) are the seams that lower it to the internal
// [`Message`] (id always present) at recording time.
type ClientInputMessage struct {
	Content    *Content          `json:"content"`
	ID         *string           `json:"id"`
	Name       *string           `json:"name"`
	Role       Role              `json:"role"`
	ToolCallID *string           `json:"tool_call_id"`
	ToolCalls  []MessageToolCall `json:"tool_calls"`
}

type ContentElement struct {
	Text       *string          `json:"text,omitempty"`
	Type       ContentType      `json:"type"`
	ImageURL   *ImageURLClass   `json:"image_url,omitempty"`
	File       *FileClass       `json:"file,omitempty"`
	InputAudio *InputAudioClass `json:"input_audio,omitempty"`
	VideoURL   *VideoURLClass   `json:"video_url,omitempty"`
}

type FileClass struct {
	FileData string `json:"file_data"`
	Filename string `json:"filename"`
}

type ImageURLClass struct {
	URL string `json:"url"`
}

type InputAudioClass struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type VideoURLClass struct {
	URL string `json:"url"`
}

type MessageToolCall struct {
	Function Function `json:"function"`
	ID       string   `json:"id"`
	Type     string   `json:"type"`
}

type Function struct {
	Arguments string `json:"arguments"`
	Name      string `json:"name"`
}

// The client→engine inbound *submit* wire form: an untrusted client submits a message,
// its full conversation view, or a named action. Lowered to domain events at the
// `SubmitClientPayload` command seam (`runtime::session::command`); never persisted
// as-is. Carried verbatim inside [`ClientInput`], which is the full client input
// surface.
//
// The body of a `client.message`: one message, optionally streamed.
//
// The body of a `client.messages`: the client's full conversation view, optionally
// streamed.
//
// The payload of a `client.action`: a named action with optional JSON args.
type ClientPayload struct {
	Message  *ClientInputMessage  `json:"message,omitempty"`
	Stream   *bool                `json:"stream,omitempty"`
	Type     ClientPayloadType    `json:"type"`
	Messages []ClientInputMessage `json:"messages,omitempty"`
	Args     interface{}          `json:"args"`
	Name     *string              `json:"name,omitempty"`
}

type DecisionRequest struct {
	// The agent config resolved for the active path (`null` when none is set).
	Agent       *AgentClass   `json:"agent"`
	AgentID     string        `json:"agent_id"`
	Ancestry    []string      `json:"ancestry"`
	Attempts    int64         `json:"attempts"`
	Calls       []CallElement `json:"calls"`
	Deadline    *time.Time    `json:"deadline"`
	DecisionID  string        `json:"decision_id"`
	Identity    Identity      `json:"identity"`
	MessageTree MessageTree   `json:"message_tree"`
	Messages    []NodeMessage `json:"messages"`
	// Count of in-flight `tool_call`/`sub_agent` calls.
	PendingCalls int64 `json:"pending_calls"`
	// The engine's default continuation for `trigger` (`null` when it needs
	// worker knowledge). Advisory: accept by echoing it as the decision.
	Proposed  *DecisionResponseClass `json:"proposed"`
	SessionID string                 `json:"session_id"`
	State     interface{}            `json:"state"`
	Trigger   Trigger                `json:"trigger"`
	TurnID    *string                `json:"turn_id"`
}

// A declared agent identity. `model` is the only required field; everything else
// refines the proposed LLM request the engine derives for `client.messages`.
type AgentClass struct {
	Model  string      `json:"model"`
	Retry  *RetryClass `json:"retry"`
	Stream *bool       `json:"stream,omitempty"`
	// Sub-agents the model can delegate to. Presented to the model as tools (by
	// id) alongside `tools`, but each call spawns a child session rather than
	// executing a function.
	SubAgents []SubAgentElement `json:"sub_agents,omitempty"`
	System    *string           `json:"system"`
	// Worker- or client-executed tools the model can call.
	Tools []AgentTool `json:"tools,omitempty"`
}

// Fully-resolved retry policy — no optional fields. Stored on call state and
// read directly by retry logic.
type RetryClass struct {
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

// A function tool the agent offers. The model-facing contract is
// `name`/`description`/`input`/`output`; `handler` selects where a call runs —
// `Some(Client)` ⇒ client-executed, absent ⇒ worker-executed (the default).
// `server` is invalid for tools.
type AgentTool struct {
	Description *string      `json:"description,omitempty"`
	Handler     *HandlerEnum `json:"handler"`
	Input       interface{}  `json:"input"`
	Name        string       `json:"name"`
	Output      interface{}  `json:"output"`
}

// An in-flight effect (Pending or RetryScheduled) surfaced on each worker decision.
// A flat envelope plus kind-specific fields: a tool call's
// `name`/`arguments`/`handler`, an LLM call's `handler`/`stream`, a
// sub-agent's `agent_id`/`session_id`.
type CallElement struct {
	AgentID *string `json:"agent_id"`
	// The tree node the effect was requested at.
	Anchor    *string      `json:"anchor"`
	Arguments *string      `json:"arguments"`
	Attempt   int64        `json:"attempt"`
	Deadline  *time.Time   `json:"deadline"`
	Handler   *HandlerEnum `json:"handler"`
	ID        string       `json:"id"`
	Kind      CallKind     `json:"kind"`
	Name      *string      `json:"name"`
	SessionID *string      `json:"session_id"`
	Status    CallStatus   `json:"status"`
	Stream    *bool        `json:"stream"`
}

type Identity struct {
	ID       *string           `json:"id"`
	Metadata map[string]string `json:"metadata,omitempty"`
	TenantID string            `json:"tenant_id"`
}

type MessageTree struct {
	HeadID *string       `json:"head_id"`
	Nodes  []NodeElement `json:"nodes,omitempty"`
}

type NodeElement struct {
	Kind     NodeKind      `json:"kind"`
	Message  *NodeMessage  `json:"message,omitempty"`
	ParentID *string       `json:"parent_id"`
	Control  *ControlClass `json:"control,omitempty"`
}

// A non-conversational tree marker (interrupt/resume); filtered out of LLM prompts.
type ControlClass struct {
	ID          string      `json:"id"`
	InterruptID string      `json:"interrupt_id"`
	Kind        ControlKind `json:"kind"`
	Origin      Origin      `json:"origin"`
	Payload     interface{} `json:"payload"`
	Reason      *string     `json:"reason,omitempty"`
}

type NodeMessage struct {
	Content    *Content          `json:"content"`
	ID         string            `json:"id"`
	Name       *string           `json:"name"`
	Role       Role              `json:"role"`
	ToolCallID *string           `json:"tool_call_id"`
	ToolCalls  []MessageToolCall `json:"tool_calls"`
}

// A decision: the messages/actions to author, plus optional state/agent writes.
// The worker returns one; the engine also proposes one as the default
// continuation (`DecisionRequest::proposed`), which the worker echoes or amends.
type DecisionResponseClass struct {
	Actions []ActionElement `json:"actions,omitempty"`
	// A new agent config write; omitted keeps the current config.
	Agent    *AgentClass          `json:"agent"`
	Messages []ClientInputMessage `json:"messages,omitempty"`
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
// `id` omitted ⇒ the effect named by the answering `tool.execute` trigger.
//
// `id` omitted ⇒ the effect named by the answering `llm.execute` trigger.
//
// `interrupt_id` omitted ⇒ the engine mints one to correlate the later resume.
type ActionElement struct {
	// `server` or `worker`; omitted ⇒ `server`.
	//
	// `worker` or `client`; omitted ⇒ `worker`.
	Handler             *HandlerEnum         `json:"handler,omitempty"`
	ID                  *string              `json:"id"`
	MaxCompletionTokens *int64               `json:"max_completion_tokens"`
	Messages            []ClientInputMessage `json:"messages"`
	Model               *string              `json:"model"`
	Reasoning           *ReasoningClass      `json:"reasoning"`
	Retry               *RetryClass          `json:"retry"`
	Stream              *bool                `json:"stream"`
	Temperature         *float64             `json:"temperature"`
	Tools               []ActionTool         `json:"tools"`
	Type                ActionType           `json:"type"`
	Arguments           interface{}          `json:"arguments"`
	Name                *string              `json:"name,omitempty"`
	Attempt             *int64               `json:"attempt"`
	Result              interface{}          `json:"result"`
	Response            *Response            `json:"response,omitempty"`
	Code                *CodeEnum            `json:"code"`
	Detail              interface{}          `json:"detail"`
	Error               *string              `json:"error,omitempty"`
	// Omitted ⇒ terminal.
	Retryable *bool   `json:"retryable,omitempty"`
	AgentID   *string `json:"agent_id,omitempty"`
	SessionID *string `json:"session_id,omitempty"`
	// The model tool-call this delegation answers — always required.
	ToolCallID  *string             `json:"tool_call_id,omitempty"`
	Message     *ClientInputMessage `json:"message,omitempty"`
	InterruptID *string             `json:"interrupt_id"`
	Payload     interface{}         `json:"payload"`
	Reason      *string             `json:"reason,omitempty"`
	Data        interface{}         `json:"data"`
}

type ReasoningClass struct {
	Effort    *EffortEnum `json:"effort"`
	Enabled   *bool       `json:"enabled"`
	Exclude   *bool       `json:"exclude"`
	MaxTokens *int64      `json:"max_tokens"`
}

// Normalized LLM response. Provider adapters convert their raw responses
// into this type at the boundary.
type Response struct {
	Content *string `json:"content"`
	// Cost in dollars for this call, if the provider reports it. A decimal
	// string on the wire.
	Cost         *string `json:"cost"`
	FinishReason *string `json:"finish_reason"`
	// Images generated by the model.
	Images    []ImageElement    `json:"images,omitempty"`
	Model     string            `json:"model"`
	ToolCalls []MessageToolCall `json:"tool_calls,omitempty"`
	Usage     interface{}       `json:"usage"`
}

// An image returned by the model in the response.
type ImageElement struct {
	URL string `json:"url"`
}

// A tool's declared contract: flat on the wire. Providers that need
// OpenAI-style `{"type": "function", "function": {…}}` nesting re-wrap at
// their own boundary.
type ActionTool struct {
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
type Trigger struct {
	Type      TriggerType          `json:"type"`
	Messages  []ClientInputMessage `json:"messages,omitempty"`
	NewFrom   *int64               `json:"new_from,omitempty"`
	Args      interface{}          `json:"args"`
	Name      *string              `json:"name,omitempty"`
	Arguments *string              `json:"arguments,omitempty"`
	Attempt   *int64               `json:"attempt,omitempty"`
	Deadline  *time.Time           `json:"deadline"`
	ID        *string              `json:"id,omitempty"`
	// The engine's classification of `arguments` against the tool's
	// declared `input` schema: `valid` (with the parsed `value`),
	// `invalid` (value plus the violation), or `malformed` (not a JSON
	// object). Always on the wire.
	Input       *Input              `json:"input,omitempty"`
	Error       *string             `json:"error"`
	Ok          *bool               `json:"ok,omitempty"`
	Result      *string             `json:"result"`
	Request     *Request            `json:"request,omitempty"`
	Stream      *bool               `json:"stream,omitempty"`
	Code        *CodeEnum           `json:"code"`
	Cost        *string             `json:"cost"`
	Detail      interface{}         `json:"detail"`
	Message     *ClientInputMessage `json:"message"`
	Truncated   *bool               `json:"truncated,omitempty"`
	Usage       interface{}         `json:"usage"`
	AgentID     *string             `json:"agent_id,omitempty"`
	SessionID   *string             `json:"session_id,omitempty"`
	InterruptID *string             `json:"interrupt_id,omitempty"`
	Payload     interface{}         `json:"payload"`
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
type Input struct {
	Status InputStatus `json:"status"`
	Value  interface{} `json:"value"`
	Error  *string     `json:"error,omitempty"`
}

type Request struct {
	MaxCompletionTokens *int64               `json:"max_completion_tokens"`
	Messages            []ClientInputMessage `json:"messages"`
	Model               string               `json:"model"`
	Reasoning           *ReasoningClass      `json:"reasoning"`
	Temperature         *float64             `json:"temperature"`
	Tools               []ActionTool         `json:"tools"`
}

type StreamDelta struct {
	FinishReason *string               `json:"finish_reason"`
	Reasoning    *string               `json:"reasoning"`
	Text         *string               `json:"text"`
	ToolCalls    []StreamDeltaToolCall `json:"tool_calls,omitempty"`
}

type StreamDeltaToolCall struct {
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
	TenantID  string                `json:"tenant_id"`
	Text      *string               `json:"text"`
	ToolCalls []StreamDeltaToolCall `json:"tool_calls,omitempty"`
	TurnID    *string               `json:"turn_id"`
}

type ContentType string

const (
	File       ContentType = "file"
	ImageURL   ContentType = "image_url"
	InputAudio ContentType = "input_audio"
	Text       ContentType = "text"
	VideoURL   ContentType = "video_url"
)

type Role string

const (
	Assistant  Role = "assistant"
	RoleSystem Role = "system"
	Tool       Role = "tool"
	User       Role = "user"
)

type ClientInputType string

const (
	InterruptResume      ClientInputType = "interrupt.resume"
	PurpleClientAction   ClientInputType = "client.action"
	PurpleClientMessage  ClientInputType = "client.message"
	PurpleClientMessages ClientInputType = "client.messages"
	PurpleToolError      ClientInputType = "tool.error"
	PurpleToolResult     ClientInputType = "tool.result"
)

type ClientPayloadType string

const (
	FluffyClientAction   ClientPayloadType = "client.action"
	FluffyClientMessage  ClientPayloadType = "client.message"
	FluffyClientMessages ClientPayloadType = "client.messages"
)

// Server-side executor resolves the provider and makes the call (LLM only).
//
// Dispatched to the work queue for the worker to execute.
//
// Executed by the client. Session goes Idle while waiting (tools only).
//
// Where a call runs — one wire enum so `handler` has a single type on every
// surface. Tool calls accept `worker` (default) or `client`; LLM calls accept
// `server` (default) or `worker`. The invalid pairing (a `server` tool, a
// `client` LLM call) is rejected at the decision seam.
//
// `server` or `worker`; omitted ⇒ `server`.
//
// `worker` or `client`; omitted ⇒ `worker`.
type HandlerEnum string

const (
	Client HandlerEnum = "client"
	Server HandlerEnum = "server"
	Worker HandlerEnum = "worker"
)

type CallKind string

const (
	LlmCall  CallKind = "llm_call"
	SubAgent CallKind = "sub_agent"
	ToolCall CallKind = "tool_call"
)

type CallStatus string

const (
	Completed      CallStatus = "completed"
	Failed         CallStatus = "failed"
	Pending        CallStatus = "pending"
	Queued         CallStatus = "queued"
	RetryScheduled CallStatus = "retry_scheduled"
)

type ControlKind string

const (
	KindInterrupt ControlKind = "interrupt"
	Resume        ControlKind = "resume"
)

// Privilege level of the caller that issued an interrupt. Derived from the
// authenticated `Caller`, never from request data; resuming requires a
// caller at or above the origin's privilege.
type Origin string

const (
	Frontend     Origin = "frontend"
	Machine      Origin = "machine"
	OriginSystem Origin = "system"
)

type NodeKind string

const (
	Control NodeKind = "control"
	Message NodeKind = "message"
)

type CodeEnum string

const (
	BudgetExceeded   CodeEnum = "budget_exceeded"
	DeadlineExceeded CodeEnum = "deadline_exceeded"
	ProviderError    CodeEnum = "provider_error"
	RateLimited      CodeEnum = "rate_limited"
	Refused          CodeEnum = "refused"
)

type EffortEnum string

const (
	High    EffortEnum = "high"
	Low     EffortEnum = "low"
	Medium  EffortEnum = "medium"
	Minimal EffortEnum = "minimal"
	None    EffortEnum = "none"
	Xhigh   EffortEnum = "xhigh"
)

type ActionType string

const (
	Done             ActionType = "done"
	FluffyToolError  ActionType = "tool.error"
	FluffyToolResult ActionType = "tool.result"
	LlmError         ActionType = "llm.error"
	LlmResult        ActionType = "llm.result"
	MessageSend      ActionType = "message.send"
	SubAgentSpawn    ActionType = "sub_agent.spawn"
	TypeInterrupt    ActionType = "interrupt"
	TypeLlmCall      ActionType = "llm.call"
	TypeToolCall     ActionType = "tool.call"
)

type InputStatus string

const (
	Invalid   InputStatus = "invalid"
	Malformed InputStatus = "malformed"
	Valid     InputStatus = "valid"
)

type TriggerType string

const (
	InterruptResumed        TriggerType = "interrupt.resumed"
	LlmExecute              TriggerType = "llm.execute"
	LlmFinished             TriggerType = "llm.finished"
	SessionStart            TriggerType = "session.start"
	SubAgentFinished        TriggerType = "sub_agent.finished"
	TentacledClientAction   TriggerType = "client.action"
	TentacledClientMessages TriggerType = "client.messages"
	ToolExecute             TriggerType = "tool.execute"
	ToolFinished            TriggerType = "tool.finished"
)

type Content struct {
	ContentElementArray []ContentElement
	String              *string
}

func (x *Content) UnmarshalJSON(data []byte) error {
	x.ContentElementArray = nil
	object, err := unmarshalUnion(data, nil, nil, nil, &x.String, true, &x.ContentElementArray, false, nil, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
	}
	return nil
}

func (x *Content) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, nil, nil, x.String, x.ContentElementArray != nil, x.ContentElementArray, false, nil, false, nil, false, nil, true)
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
