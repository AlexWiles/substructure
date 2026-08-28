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

// Everything a client can send: submit a message, a full view, an append
// batch, or a named action; resume an interrupt; or settle a client tool.
//
// Each variant carries only the addressing it needs. The four submit variants
// carry `agent_id`, which routes the turn and starts the session if it is new,
// and an optional `turn_id`. A resume or settle names an interrupt or effect
// and continues whatever turn is running, so it carries neither.
// `session_id` is on the envelope.
//
// The body of an interrupt resume: which interrupt, and the payload delivered
// to the worker. Shared by the [`ClientInput::InterruptResume`] input and the
// [`DecisionTrigger::InterruptResumed`] trigger.
type ClientInput struct {
	AgentID                                                              *string         `json:"agent_id,omitempty"`
	Message                                                              *DraftMessage   `json:"message,omitempty"`
	// Hold this message for the next turn instead of refusing it while a                
	// turn is running. Off by default.                                                  
	//                                                                                   
	// Hold this batch for the next turn instead of refusing it while a                  
	// turn is running. Off by default.                                                  
	Queue                                                                *bool           `json:"queue,omitempty"`
	Stream                                                               *bool           `json:"stream,omitempty"`
	TurnID                                                               *string         `json:"turn_id"`
	Type                                                                 ClientInputType `json:"type"`
	Client                                                               *ClientContext  `json:"client,omitempty"`
	Messages                                                             []DraftMessage  `json:"messages,omitempty"`
	Args                                                                 interface{}     `json:"args"`
	Name                                                                 *string         `json:"name,omitempty"`
	InterruptID                                                          *string         `json:"interrupt_id,omitempty"`
	Payload                                                              interface{}     `json:"payload"`
	Attempt                                                              *int64          `json:"attempt"`
	Content                                                              []ToolContent   `json:"content"`
	ID                                                                   *string         `json:"id,omitempty"`
	IsError                                                              *bool           `json:"is_error,omitempty"`
	Result                                                               interface{}     `json:"result"`
	StructuredContent                                                    interface{}     `json:"structured_content"`
	Error                                                                *string         `json:"error,omitempty"`
	Retryable                                                            *bool           `json:"retryable,omitempty"`
}

// Inputs a client declares on its run, passed to the worker on the
// `client.messages` decision. `tools` are the browser's own tools, read as
// client-handled [`AgentTool`]s. The engine adds them to the proposed config.
// A worker can override that by returning its own `agent`.
//
// Inputs the client declared on its run. The engine adds
// `client.tools` to the proposed config.
type ClientContext struct {
	Context        []interface{} `json:"context,omitempty"`
	ForwardedProps interface{}   `json:"forwarded_props"`
	State          interface{}   `json:"state"`
	Tools          []AgentTool   `json:"tools,omitempty"`
}

// A function tool the agent offers. `handler` says where a call runs:
// `client` for the client, absent for the worker. `server` is invalid here.
// The engine runs only connector tools, and a worker declares those by
// connection id.
type AgentTool struct {
	// Keep this tool out of the request. See [`LlmTool::defer`]. Absent ⇒            
	// the agent's `defer_tools`.                                                     
	Defer                                                                 *bool       `json:"defer"`
	Description                                                           *string     `json:"description,omitempty"`
	Handler                                                               *Handler    `json:"handler"`
	Input                                                                 interface{} `json:"input"`
	Name                                                                  string      `json:"name"`
	Output                                                                interface{} `json:"output"`
}

type ToolContent struct {
	Text     *string           `json:"text,omitempty"`
	Type     ToolContentType   `json:"type"`
	Data     *string           `json:"data,omitempty"`
	MIMEType *string           `json:"mimeType"`
	Resource *ResourceContents `json:"resource,omitempty"`
	Name     *string           `json:"name"`
	URI      *string           `json:"uri,omitempty"`
}

type ResourceContents struct {
	Blob     *string `json:"blob"`
	MIMEType *string `json:"mimeType"`
	Text     *string `json:"text"`
	URI      string  `json:"uri"`
}

// The wire form of a [`Message`]. `id` is absent until the message is
// recorded.
type DraftMessage struct {
	Content    *Content          `json:"content"`
	ID         *string           `json:"id"`
	Name       *string           `json:"name"`
	Reasoning  *Reasoning        `json:"reasoning"`
	Role       Role              `json:"role"`
	ToolCallID *string           `json:"tool_call_id"`
	ToolCalls  []ToolCallElement `json:"tool_calls"`
}

type StoredContent struct {
	Text     *string           `json:"text,omitempty"`
	Type     StoredContentType `json:"type"`
	URI      *string           `json:"uri,omitempty"`
	MIMEType *string           `json:"mimeType"`
	Name     *string           `json:"name"`
}

// What the model thought before it answered. `text` is for a reader.
// `blocks` are the provider's own and stay unchanged. Anthropic requires the
// thinking before a tool call back with its signature.
type Reasoning struct {
	Blocks   []interface{}     `json:"blocks,omitempty"`
	Provider ReasoningProvider `json:"provider"`
	Text     *string           `json:"text"`
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

// What a client submits: a message, its full conversation view, an append
// batch, or a named action. The engine turns it into events and never stores
// it as it arrived.
//
// The body of a `client.message`: one message, optionally streamed.
//
// The body of a `client.messages`: the client's full conversation view, optionally
// streamed.
//
// The body of a `client.append`. Messages are added at the session head,
// against the active path at delivery. A queued append lands after whatever
// turn beat it, so it cannot fork the tree. A message whose id is already
// recorded is dropped.
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
	// The agent config for the active path. `null` when none is set.                       
	Agent                                                                  *AgentConfig     `json:"agent"`
	AgentID                                                                string           `json:"agent_id"`
	Ancestry                                                               []string         `json:"ancestry"`
	Attempts                                                               int64            `json:"attempts"`
	Calls                                                                  []Effect         `json:"calls"`
	Deadline                                                               *time.Time       `json:"deadline"`
	DecisionID                                                             string           `json:"decision_id"`
	Identity                                                               WorkerIdentity   `json:"identity"`
	MessageTree                                                            MessageTree      `json:"message_tree"`
	Messages                                                               []Message        `json:"messages"`
	// Count of in-flight `tool_call`/`subagent` calls.                                    
	PendingCalls                                                           int64            `json:"pending_calls"`
	// The engine's default continuation for `trigger`. Empty when only the                 
	// worker can decide. Accept it by echoing it back.                                     
	Proposed                                                               DecisionResponse `json:"proposed"`
	SessionID                                                              string           `json:"session_id"`
	State                                                                  interface{}      `json:"state"`
	Trigger                                                                DecisionTrigger  `json:"trigger"`
	TurnID                                                                 *string          `json:"turn_id"`
}

// A declared agent. The same shape whether a file writes it or a worker
// returns it.
//
// `llm` names the `[llm.*]` block every call runs on. That block decides where
// the call runs and what shape it takes. A config that names no block fails
// when the engine resolves a call.
type AgentConfig struct {
	// Defer every tool this agent offers, whatever its source. A tool or a                      
	// connection overrides this with its own `defer`. Absent, the agent defers                  
	// nothing; a connection can still defer on its own.                                         
	DeferTools                                                                 *DeferToolsWire   `json:"defer_tools"`
	// How hard the model thinks. Unset leaves the provider's own default.                       
	Effort                                                                     *ReasoningEffort  `json:"effort"`
	// The `[llm.*]` block this agent's calls run on.                                            
	Llm                                                                        *string           `json:"llm"`
	// MCP servers this agent draws tools from.                                                  
	MCP                                                                        []MCPServer       `json:"mcp,omitempty"`
	// Whether the engine tells the model that an MCP server is available, and                   
	// what that server says it is for.                                                          
	MCPAnnounce                                                                *MCPAnnounce      `json:"mcp_announce,omitempty"`
	Model                                                                      string            `json:"model"`
	// Plugins this agent uses.                                                                  
	Plugins                                                                    []AgentPlugin     `json:"plugins,omitempty"`
	// Boxed. Five per-kind overrides are too many bytes to carry inline.                        
	Retry                                                                      *RetryConfig      `json:"retry"`
	// Subagents the model can delegate to. The model sees them as tools.                       
	// Each call starts a child session.                                                         
	Subagents                                                                  []SubagentElement `json:"subagents,omitempty"`
	System                                                                     *string           `json:"system"`
	// Worker- or client-executed tools the model can call.                                      
	Tools                                                                      []AgentTool       `json:"tools,omitempty"`
}

// How an agent's deferred tools reach the model.
type DeferTools struct {
	// The most matches one search answers with. Never zero: a search that can                    
	// answer with nothing is a search the model cannot use.                                      
	MaxMatches                                                                *int64              `json:"max_matches,omitempty"`
	// Which tools the agent gets to reach the ones it defers.                                    
	Strategy                                                                  *DeferToolsStrategy `json:"strategy,omitempty"`
}

// An MCP server the agent draws tools from. `path` names a connection the
// engine holds. A worker never writes a URL or a credential.
type MCPServer struct {
	Approve                                                                   *Approve            `json:"approve,omitempty"`
	AuthFailure                                                               *MCPAuthFailure     `json:"auth_failure,omitempty"`
	Path                                                                      string              `json:"path"`
	ToolSyncFailure                                                           *MCPToolSyncFailure `json:"tool_sync_failure,omitempty"`
	// Narrows what the model sees. Absent ⇒ every tool the connection grants.                    
	Tools                                                                     *MCPTools           `json:"tools"`
}

// Which of a connection's tools the model sees, and how they reach it.
//
// The filter runs in order: capability predicates, then `include`, then
// `exclude`. Each step only removes, so a filter cannot widen what the
// connection grants. `defer` runs last and removes nothing.
//
// `include` and `exclude` are globs over the tool's name on the connection,
// not the prefixed name the model sees.
//
// Capability predicates read the MCP annotations. A tool with no annotation
// fails the predicate, so a server that annotates nothing yields nothing under
// `read_only`.
type MCPTools struct {
	// Keep every surviving tool out of the request. See [`LlmTool::defer`].         
	// Absent ⇒ the agent's `defer_tools`.                                           
	Defer                                                                   *bool    `json:"defer"`
	Exclude                                                                 []string `json:"exclude,omitempty"`
	Idempotent                                                              *bool    `json:"idempotent"`
	Include                                                                 []string `json:"include,omitempty"`
	NonDestructive                                                          *bool    `json:"non_destructive"`
	ReadOnly                                                                *bool    `json:"read_only"`
}

// A plugin an agent uses. The skills and servers come from the bundle when the
// config loads.
type AgentPlugin struct {
	Approve                                            *Approve            `json:"approve,omitempty"`
	AuthFailure                                        *MCPAuthFailure     `json:"auth_failure,omitempty"`
	Description                                        *string             `json:"description,omitempty"`
	ID                                                 string              `json:"id"`
	// Where each of this plugin's servers is declared.                    
	Servers                                            []string            `json:"servers,omitempty"`
	Skills                                             []SkillMeta         `json:"skills,omitempty"`
	ToolSyncFailure                                    *MCPToolSyncFailure `json:"tool_sync_failure,omitempty"`
	// Applied to each of the plugin's servers.                            
	Tools                                              *MCPTools           `json:"tools"`
}

// What the model sees of a skill before it loads it.
type SkillMeta struct {
	Description *string `json:"description,omitempty"`
	Name        string  `json:"name"`
}

// Retry overrides, one for each effect kind. `default` covers the kinds that
// name nothing. A kind layers on top of `default`.
type RetryConfig struct {
	Connector *RetryOverride `json:"connector"`
	Default   *RetryOverride `json:"default"`
	Llm       *RetryOverride `json:"llm"`
	Subagent  *RetryOverride `json:"subagent"`
	Tool      *RetryOverride `json:"tool"`
}

// Only the fields it names change. An override cannot make a timeout
// unbounded.
type RetryOverride struct {
	BackoffBaseSecs  *int64 `json:"backoff_base_secs"`
	BackoffMaxSecs   *int64 `json:"backoff_max_secs"`
	MaxAttempts      *int64 `json:"max_attempts"`
	QueueTimeoutSecs *int64 `json:"queue_timeout_secs"`
	RunTimeoutSecs   *int64 `json:"run_timeout_secs"`
	TotalTimeoutSecs *int64 `json:"total_timeout_secs"`
}

// A subagent the model can delegate to. `id` is both the child agent and the
// tool name the model calls. Its input is one `message`.
type SubagentElement struct {
	Description *string `json:"description,omitempty"`
	ID          string  `json:"id"`
}

// An effect still running, shown on each worker decision. A flat envelope
// plus the fields of its kind. A connector sync carries none. Its `id` is the
// connection being fetched.
type Effect struct {
	AgentID                                                           *string      `json:"agent_id"`
	// The tree node the effect was requested at.                                  
	Anchor                                                            *string      `json:"anchor"`
	Arguments                                                         *string      `json:"arguments"`
	Attempt                                                           int64        `json:"attempt"`
	Deadline                                                          *time.Time   `json:"deadline"`
	Handler                                                           *Handler     `json:"handler"`
	ID                                                                string       `json:"id"`
	Kind                                                              EffectKind   `json:"kind"`
	Name                                                              *string      `json:"name"`
	Status                                                            EffectStatus `json:"status"`
	Stream                                                            *bool        `json:"stream"`
	// The model tool call a delegation answers. Its `id` is the child             
	// session.                                                                    
	ToolCallID                                                        *string      `json:"tool_call_id"`
}

// The owner as the worker receives it, without the tenant. Read `kind` with
// `id`. Only `frontend` is an end user.
type WorkerIdentity struct {
	Metadata   map[string]string `json:"metadata"`
	Subject    *Subject          `json:"subject"`
	Visibility *Visibility       `json:"visibility,omitempty"`
}

// One identity, as the source that authenticated it named it. An id is
// unique only within its issuer.
type Subject struct {
	ID     string `json:"id"`
	Issuer string `json:"issuer"`
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
	Reasoning  *Reasoning        `json:"reasoning"`
	Role       Role              `json:"role"`
	ToolCallID *string           `json:"tool_call_id"`
	ToolCalls  []ToolCallElement `json:"tool_calls,omitempty"`
}

// The engine's default continuation for `trigger`. Empty when only the
// worker can decide. Accept it by echoing it back.
//
// The messages and actions to author, plus optional state and agent writes.
// A worker returns one. The engine proposes one too, which the worker echoes
// or changes.
type DecisionResponse struct {
	Actions                                                                   []DecisionAction       `json:"actions,omitempty"`
	// A new agent config write; omitted keeps the current config.                                   
	Agent                                                                     *AgentConfig           `json:"agent"`
	// How each channel shows this decision, keyed by channel kind. The engine                       
	// does not read it.                                                                             
	Channels                                                                  map[string]interface{} `json:"channels,omitempty"`
	Messages                                                                  []DraftMessage         `json:"messages,omitempty"`
	// Absent or `null` keeps the current state. Send an empty value to                              
	// clear it.                                                                                     
	State                                                                     interface{}            `json:"state"`
}

// The action a worker writes on the wire. A settle can leave out the effect
// id, because the `*.execute` trigger it answers already names it.
//
// A flat LLM request. Every field is optional.
//
// Without `id`, the engine mints one, and it becomes the assistant node's
// id. A field left out comes from the agent config, then from the engine's
// default. Without `messages`, the request carries the config's system
// message and the decision's view. Given `messages`, no system message is
// added.
//
// Without `id`, the engine mints one. A tool the model called carries the
// model's id.
//
// There is no `handler`. The name says where the call runs: a connector
// tool on the engine, a `handler: client` tool on the client, anything
// else on the worker. The engine knows all three already.
//
// Without `id` and `attempt`, both come from the `tool.execute` trigger
// this answers. That ties the result to the attempt that ran.
//
// Without `id` and `attempt`, both come from the `llm.execute` trigger
// this answers. That ties the result to the attempt that ran.
//
// Without `interrupt_id`, the engine mints one to match the later
// resume.
//
// Resolve an open interrupt and resume the session.
//
// Fetch a connection's tools again, after a person replaced its
// credential.
type DecisionAction struct {
	ID                                                                   *string            `json:"id"`
	// The `[llm.*]` block this call runs on. Absent uses the config's                      
	// `llm`. Naming another block moves this one call elsewhere.                           
	Llm                                                                  *string            `json:"llm"`
	MaxCompletionTokens                                                  *int64             `json:"max_completion_tokens"`
	Messages                                                             []DraftMessage     `json:"messages"`
	Model                                                                *string            `json:"model"`
	Reasoning                                                            *ReasoningConfig   `json:"reasoning"`
	// Layered over the agent config's `llm` policy, or over the engine's                   
	// default.                                                                             
	//                                                                                      
	// Layered over the agent config's policy for this kind, or over the                    
	// engine's default for where the tool runs.                                            
	//                                                                                      
	// Layered over the agent config's `subagent` policy, or over the                      
	// engine's default.                                                                    
	Retry                                                                *RetryOverride     `json:"retry"`
	Stream                                                               *bool              `json:"stream"`
	Temperature                                                          *float64           `json:"temperature"`
	Tools                                                                []LlmTool          `json:"tools"`
	Type                                                                 DecisionActionType `json:"type"`
	Arguments                                                            interface{}        `json:"arguments"`
	Name                                                                 *string            `json:"name,omitempty"`
	Attempt                                                              *int64             `json:"attempt"`
	Content                                                              []ToolContent      `json:"content"`
	IsError                                                              *bool              `json:"is_error,omitempty"`
	Result                                                               interface{}        `json:"result"`
	StructuredContent                                                    interface{}        `json:"structured_content"`
	// An `LlmResponse`, or the provider's own response when the                            
	// `llm.execute` this answers carried a `format`.                                       
	Response                                                             interface{}        `json:"response"`
	Code                                                                 *ErrorCode         `json:"code"`
	Detail                                                               interface{}        `json:"detail"`
	Error                                                                *string            `json:"error,omitempty"`
	// Omitted ⇒ terminal.                                                                  
	Retryable                                                            *bool              `json:"retryable,omitempty"`
	AgentID                                                              *string            `json:"agent_id,omitempty"`
	// The child's opening message. It travels with the spawn, so it                        
	// cannot arrive before the session exists.                                             
	Message                                                              *DraftMessage      `json:"message"`
	SessionID                                                            *string            `json:"session_id,omitempty"`
	// The model tool call this delegation answers. Required.                               
	ToolCallID                                                           *string            `json:"tool_call_id,omitempty"`
	InterruptID                                                          *string            `json:"interrupt_id"`
	Payload                                                              interface{}        `json:"payload"`
	Reason                                                               *string            `json:"reason,omitempty"`
	Path                                                                 *string            `json:"path,omitempty"`
	Data                                                                 interface{}        `json:"data"`
}

type ReasoningConfig struct {
	Effort    *ReasoningEffort `json:"effort"`
	Enabled   *bool            `json:"enabled"`
	Exclude   *bool            `json:"exclude"`
	MaxTokens *int64           `json:"max_tokens"`
}

// A tool's declared contract. Flat on the wire. A provider that needs it
// nested re-wraps it at its own boundary.
type LlmTool struct {
	// Keep this definition out of the request.                                            
	//                                                                                     
	// The engine still records it, still routes a call to it, and still finds             
	// it in a search. Only the request leaves it out. That keeps a large tool             
	// set out of the model's context and out of the cached prefix.                        
	//                                                                                     
	// Any source can set it. Deferral belongs to a tool, not to where the tool            
	// came from.                                                                          
	Defer                                                                      *bool       `json:"defer,omitempty"`
	Description                                                                string      `json:"description"`
	// JSON Schema for the arguments. Absent declares a tool with no                       
	// arguments. The engine checks every call against it.                                 
	Input                                                                      interface{} `json:"input"`
	Name                                                                       string      `json:"name"`
	// JSON Schema the result must satisfy. The model never sees it. A result              
	// that breaks it becomes a terminal tool error.                                       
	Output                                                                     interface{} `json:"output"`
}

// The trigger a worker sees on the wire. There is no `ClientMessage`: the
// engine turns a bare client message into `ClientTranscript` before it sends
// it.
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
// Sent after a turn completes, with its final output. The session stays
// busy until it is answered. Echo the proposed `done` to finish.
type DecisionTrigger struct {
	Type                                                                   DecisionTriggerType `json:"type"`
	// Inputs the client declared on its run. The engine adds                                  
	// `client.tools` to the proposed config.                                                  
	Client                                                                 *ClientContext      `json:"client,omitempty"`
	Messages                                                               []DraftMessage      `json:"messages,omitempty"`
	NewFrom                                                                *int64              `json:"new_from,omitempty"`
	Args                                                                   interface{}         `json:"args"`
	Name                                                                   *string             `json:"name,omitempty"`
	Arguments                                                              *string             `json:"arguments,omitempty"`
	Attempt                                                                *int64              `json:"attempt,omitempty"`
	Deadline                                                               *time.Time          `json:"deadline"`
	ID                                                                     *string             `json:"id,omitempty"`
	// What the engine made of `arguments` against the tool's `input`                          
	// schema: `valid`, `invalid`, or `malformed`. Always on the wire.                         
	Input                                                                  *ToolInput          `json:"input,omitempty"`
	Error                                                                  *ErrorInfo          `json:"error"`
	Ok                                                                     *bool               `json:"ok,omitempty"`
	Result                                                                 *Result             `json:"result"`
	Format                                                                 *LlmFormat          `json:"format"`
	// The neutral `LlmRequest` JSON, or the provider's native request body                    
	// when `format` is set.                                                                   
	Request                                                                interface{}         `json:"request"`
	Stream                                                                 *bool               `json:"stream,omitempty"`
	Cost                                                                   *string             `json:"cost"`
	Message                                                                *DraftMessage       `json:"message"`
	// True when the model declined the request. Without it, a refusal                         
	// looks like a turn that ended well and said nothing.                                     
	Refused                                                                *bool               `json:"refused,omitempty"`
	Truncated                                                              *bool               `json:"truncated,omitempty"`
	Usage                                                                  *Usage              `json:"usage"`
	AgentID                                                                *string             `json:"agent_id,omitempty"`
	SessionID                                                              *string             `json:"session_id,omitempty"`
	InterruptID                                                            *string             `json:"interrupt_id,omitempty"`
	Payload                                                                interface{}         `json:"payload"`
	Data                                                                   interface{}         `json:"data"`
	TurnID                                                                 *string             `json:"turn_id,omitempty"`
}

// Why something failed. One shape on every event and on the wire.
//
// There is no `retryable` field. Whether to try again is a decision about one
// attempt, not a fact about the failure. The events that settle an attempt
// carry it instead.
type ErrorInfo struct {
	Code                                                                ErrorCode   `json:"code"`
	// Small structured details, such as a status or the llm blocks that            
	// exist.                                                                       
	Detail                                                              interface{} `json:"detail"`
	// One sentence the engine wrote, safe to show a human. Never a raw             
	// document. An unbounded body belongs in the log.                              
	Message                                                             string      `json:"message"`
	// The one input to fix, when the failure names one. For example                
	// `agent.llm` or `actions[0].type`.                                            
	Param                                                               *string     `json:"param"`
}

// What the engine made of `arguments` against the tool's `input`
// schema: `valid`, `invalid`, or `malformed`. Always on the wire.
//
// What the engine made of a tool call's arguments, sent with the raw
// `arguments` string. Always on the wire.
//
// Parsed, and valid against the `input` schema if the tool declares one.
// `value` is the parsed `arguments`. The engine never changes it.
//
// Parsed to an object that violates the declared `input` schema.
//
// Not a JSON object. Either malformed JSON or another type.
type ToolInput struct {
	Status Status      `json:"status"`
	Value  interface{} `json:"value"`
	Error  *string     `json:"error,omitempty"`
}

type StoredResult struct {
	Content           []StoredContent `json:"content,omitempty"`
	IsError           *bool           `json:"isError,omitempty"`
	StructuredContent interface{}     `json:"structuredContent"`
}

// What one call read and wrote. Every provider means these counts the same
// way.
//
// Vendors report different things. Anthropic gives the part of the prompt it
// did not read from the cache. OpenAI gives the whole prompt. Each adapter
// converts to this shape, because these counts get added together.
type Usage struct {
	// The part of `input` the provider read from the cache.                            
	CacheRead                                                               int64       `json:"cache_read"`
	// The part of `input` the provider wrote to the cache.                             
	CacheWrite                                                              int64       `json:"cache_write"`
	// Every input token of the call, cached or not.                                    
	Input                                                                   int64       `json:"input"`
	Output                                                                  int64       `json:"output"`
	// The counts as the provider reported them, for a number this type does            
	// not name.                                                                        
	Provider                                                                interface{} `json:"provider"`
	// `input` and `output` together.                                                   
	Total                                                                   int64       `json:"total"`
	// The part of `input` the provider read fresh.                                     
	UncachedInput                                                           int64       `json:"uncached_input"`
}

// An interrupt payload in the AG-UI shape. `id` and `reason` live on the
// interrupt itself.
type InterruptPayload struct {
	// RFC 3339. Display only.                                                       
	ExpiresAt                                                            *string     `json:"expiresAt"`
	// Markdown. A channel converts it as it needs. Without it, a channel            
	// shows the interrupt's `reason`.                                               
	Message                                                              *string     `json:"message"`
	// Free-form, delivered to clients unchanged. `metadata.options` is a            
	// list of [`InterruptOption`], which Slack shows as buttons.                    
	Metadata                                                             interface{} `json:"metadata"`
	// JSON Schema for the expected resolution payload.                              
	ResponseSchema                                                       interface{} `json:"responseSchema"`
	// Binds the interrupt to a prior tool call.                                     
	ToolCallID                                                           *string     `json:"toolCallId"`
}

// A resume payload a channel wrote: the AG-UI shape plus who resolved it.
type InterruptResolution struct {
	Payload   interface{}         `json:"payload"`
	Responder *InterruptResponder `json:"responder"`
	Status    ResumeStatus        `json:"status"`
}

// Who resolved it. The channel sets this, never the requester.
type InterruptResponder struct {
	// The channel kind, e.g. `slack`, `ag-ui`.                            
	Channel                                                        string  `json:"channel"`
	// The chosen option's label, when the resolution was a pick.          
	Label                                                          *string `json:"label"`
	// The chosen option's `style`, when the resolution was a pick.        
	Style                                                          *string `json:"style"`
	// Channel-native user id.                                             
	User                                                           *string `json:"user"`
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
	AgentID                                                 string          `json:"agent_id"`
	Attempt                                                 int64           `json:"attempt"`
	CallID                                                  string          `json:"call_id"`
	FinishReason                                            *string         `json:"finish_reason"`
	Reasoning                                               *string         `json:"reasoning"`
	// Transport routing key.                                               
	RootSessionID                                           string          `json:"root_session_id"`
	// Per-call counter, distinct from event-store sequence.                
	Seq                                                     int64           `json:"seq"`
	// May be a subagent of root.                                          
	SessionID                                               string          `json:"session_id"`
	// Tenant isolation key — subscribers must match.                       
	TenantID                                                string          `json:"tenant_id"`
	Text                                                    *string         `json:"text"`
	ToolCalls                                               []ToolCallChunk `json:"tool_calls"`
	TurnID                                                  *string         `json:"turn_id"`
}

// The engine makes the call.
//
// The worker executes it.
//
// The client executes it. The session goes idle until it answers.
// Tools only.
type Handler string

const (
	Client Handler = "client"
	Server Handler = "server"
	Worker Handler = "worker"
)

type ToolContentType string

const (
	Audio        ToolContentType = "audio"
	Image        ToolContentType = "image"
	PurpleText   ToolContentType = "text"
	Resource     ToolContentType = "resource"
	ResourceLink ToolContentType = "resource_link"
)

type StoredContentType string

const (
	Blob       StoredContentType = "blob"
	FluffyText StoredContentType = "text"
	Link       StoredContentType = "link"
)

// Which provider wrote the blocks. They go back only to that provider.
// Anthropic rejects blocks it did not sign.
type ReasoningProvider string

const (
	Openrouter                 ReasoningProvider = "openrouter"
	ReasoningProviderAnthropic ReasoningProvider = "anthropic"
	ReasoningProviderOpenai    ReasoningProvider = "openai"
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

// Which tools the agent gets to reach the ones it defers.
//
// How the tools an agent defers reach the model.
//
// The engine holds every deferred definition whatever this says. This chooses
// which tools the request advertises, and whether the request carries the
// deferred definitions.
//
// `tool_search` and `call_tool`. A search answers with the schema, so one
// search is enough to make a call.
type DeferToolsStrategy string

const (
	Search DeferToolsStrategy = "search"
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

// Which of a connection's calls stop for a person.
//
// A tool that the connection marks `destructiveHint`.
type Approve string

const (
	Always       Approve = "always"
	ApproveNever Approve = "never"
	Destructive  Approve = "destructive"
)

// What a session does when a connection needs a person to authorize it.
//
// Stop and ask. A channel that cannot show the question degrades.
//
// Go on without this connection's tools.
type MCPAuthFailure string

const (
	Degrade                 MCPAuthFailure = "degrade"
	MCPAuthFailureInterrupt MCPAuthFailure = "interrupt"
)

// Whether the model is told that a connection's tool fetch failed. The turn
// goes ahead without those tools either way.
//
// Name the connection wherever its tools would have been.
//
// Say nothing. For a connection the agent does not need.
type MCPToolSyncFailure string

const (
	Silent MCPToolSyncFailure = "silent"
	Warn   MCPToolSyncFailure = "warn"
)

// Whether the engine tells the model that an MCP server is available, and
// what that server says it is for.
//
// Where an MCP announcement lands.
//
// The system prompt while no call has run. Then a block on the last user
// message. Then a message of its own. The engine takes the first place it
// can use.
//
// Nowhere. For a server whose own description does not help.
type MCPAnnounce string

const (
	Auto             MCPAnnounce = "auto"
	MCPAnnounceNever MCPAnnounce = "never"
)

// What kind of work an effect is. One enum for the wire and for scheduling. A
// decision and a turn's end queue beside the calls and are swept the same way,
// so they are kinds too. Neither appears on an [`Effect`].
//
// Fetching one connection's tool list. Its `id` is the connection id.
//
// A worker decision.
//
// The turn's completion, which waits for the `turn.finished` decision to
// settle. Carries the turn id. Never swept, because it has no deadline.
type EffectKind string

const (
	ConnectorSync EffectKind = "connector_sync"
	Decision      EffectKind = "decision"
	LlmCall       EffectKind = "llm_call"
	Subagent      EffectKind = "subagent"
	ToolCall      EffectKind = "tool_call"
	TurnEnd       EffectKind = "turn_end"
)

// Running, waiting for its result. Off the deadline clock. A delegation
// stays here for as long as its child turn takes.
type EffectStatus string

const (
	Completed      EffectStatus = "completed"
	Failed         EffectStatus = "failed"
	Pending        EffectStatus = "pending"
	Queued         EffectStatus = "queued"
	RetryScheduled EffectStatus = "retry_scheduled"
	Running        EffectStatus = "running"
)

// Who can read what a session says. The transport sets it once, when the
// session starts. Absent or unknown reads as `shared`. `shared` never selects
// a personal credential.
//
// More than one person can read the answer.
//
// One person only.
type Visibility string

const (
	Private Visibility = "private"
	Shared  Visibility = "shared"
)

// What kind of failure, so a consumer can branch on it instead of reading the
// sentence. A closed set, required on every [`ErrorInfo`].
//
// `provider_error`, `rate_limited`, `refused`, `budget_exceeded`, and
// `deadline_exceeded` mean a call ran and went wrong.
// `invalid_response` means a document did not parse, or parsed into something
// unusable. `handler_error` means the worker or client reported its own
// failure. `worker_unreachable` means it was never reached. `unroutable`
// means nothing could decide. `internal` means the engine's own fault.
type ErrorCode string

const (
	BudgetExceeded    ErrorCode = "budget_exceeded"
	DeadlineExceeded  ErrorCode = "deadline_exceeded"
	HandlerError      ErrorCode = "handler_error"
	Internal          ErrorCode = "internal"
	InvalidResponse   ErrorCode = "invalid_response"
	ProviderError     ErrorCode = "provider_error"
	RateLimited       ErrorCode = "rate_limited"
	Refused           ErrorCode = "refused"
	Unroutable        ErrorCode = "unroutable"
	WorkerUnreachable ErrorCode = "worker_unreachable"
)

type DecisionActionType string

const (
	Done              DecisionActionType = "done"
	FluffyToolError   DecisionActionType = "tool.error"
	FluffyToolResult  DecisionActionType = "tool.result"
	InterruptResolve  DecisionActionType = "interrupt.resolve"
	LlmError          DecisionActionType = "llm.error"
	LlmResult         DecisionActionType = "llm.result"
	MessageSend       DecisionActionType = "message.send"
	SubagentSpawn     DecisionActionType = "subagent.spawn"
	TypeConnectorSync DecisionActionType = "connector.sync"
	TypeInterrupt     DecisionActionType = "interrupt"
	TypeLlmCall       DecisionActionType = "llm.call"
	TypeToolCall      DecisionActionType = "tool.call"
)

// OpenAI Chat Completions.
//
// Anthropic Messages API.
type LlmFormat string

const (
	LlmFormatAnthropic LlmFormat = "anthropic"
	LlmFormatOpenai    LlmFormat = "openai"
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
	SubagentFinished        DecisionTriggerType = "subagent.finished"
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
	StoredContentArray []StoredContent
	String             *string
}

func (x *Content) UnmarshalJSON(data []byte) error {
	x.StoredContentArray = nil
	object, err := unmarshalUnion(data, nil, nil, nil, &x.String, true, &x.StoredContentArray, false, nil, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
	}
	return nil
}

func (x *Content) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, nil, nil, x.String, x.StoredContentArray != nil, x.StoredContentArray, false, nil, false, nil, false, nil, true)
}

// Defer every tool this agent offers, whatever its source. A tool or a
// connection overrides this with its own `defer`. Absent, the agent defers
// nothing; a connection can still defer on its own.
type DeferToolsWire struct {
	Bool       *bool
	DeferTools *DeferTools
}

func (x *DeferToolsWire) UnmarshalJSON(data []byte) error {
	x.DeferTools = nil
	var c DeferTools
	object, err := unmarshalUnion(data, nil, nil, &x.Bool, nil, false, nil, true, &c, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
		x.DeferTools = &c
	}
	return nil
}

func (x *DeferToolsWire) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, nil, x.Bool, nil, false, nil, x.DeferTools != nil, x.DeferTools, false, nil, false, nil, true)
}

type Result struct {
	StoredResult *StoredResult
	String       *string
}

func (x *Result) UnmarshalJSON(data []byte) error {
	x.StoredResult = nil
	var c StoredResult
	object, err := unmarshalUnion(data, nil, nil, nil, &x.String, false, nil, true, &c, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
		x.StoredResult = &c
	}
	return nil
}

func (x *Result) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, nil, nil, x.String, false, nil, x.StoredResult != nil, x.StoredResult, false, nil, false, nil, true)
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
