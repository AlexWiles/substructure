// ── Primitives ──────────────────────────────────────────────────────────────

/** UUID v7 string */
export type Uuid = string;

/** ISO 8601 datetime string (UTC) */
export type DateTime = string;

/** Decimal as string (e.g. "0.0012") */
export type Decimal = string;

/** Hex-encoded 16-byte trace ID */
export type TraceId = string;

/** Hex-encoded 8-byte span ID */
export type SpanId = string;

// ── Identity ────────────────────────────────────────────────────────────────

export interface ClientIdentity {
  tenant_id: string;
  sub?: string;
  attrs?: Record<string, string>;
}

// ── Retry ───────────────────────────────────────────────────────────────────

export interface RetryPolicy {
  timeout_secs: number | null;
  max_retries: number;
  backoff_base_secs: number;
  backoff_max_secs: number;
}

export interface RetryState {
  attempts: number;
  next_at?: DateTime;
}

// ── Tracing ─────────────────────────────────────────────────────────────────

export interface SpanContext {
  trace_id: TraceId;
  span_id: SpanId;
  parent_span_id?: SpanId;
  trace_flags: number;
  trace_state?: string;
  name?: string;
}

// ── Messages ────────────────────────────────────────────────────────────────

export type Role = "system" | "user" | "assistant" | "tool";

export interface ToolCallFunction {
  name: string;
  arguments: string;
}

export interface ToolCall {
  id: string;
  type: string;
  function: ToolCallFunction;
}

export interface Message {
  role: Role;
  content?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
}

// ── LLM ─────────────────────────────────────────────────────────────────────

export interface LlmToolFunction {
  name: string;
  description: string;
  parameters: unknown;
}

export interface LlmTool {
  function: LlmToolFunction;
}

export interface LlmRequest {
  model: string;
  messages: Message[];
  tools?: LlmTool[];
  temperature?: number;
  max_completion_tokens?: number;
}

export interface LlmResponse {
  model: string;
  content?: string;
  tool_calls: ToolCall[];
  finish_reason?: string;
  usage?: Record<string, unknown>;
  cost?: Decimal;
}

// ── Artifacts ───────────────────────────────────────────────────────────────

export type Part =
  | { kind: "text"; text: string }
  | { kind: "data"; data: unknown };

export interface Artifact {
  name?: string;
  description?: string;
  parts: Part[];
}

// ── Tool Handler ────────────────────────────────────────────────────────────

export type ToolHandler = "worker" | "client";

// ── Decision Triggers ───────────────────────────────────────────────────────

export interface ToolResult {
  tool_call_id: string;
  name: string;
  content: string;
  is_error: boolean;
}

export type DecisionTrigger =
  | { type: "user_message"; stream: boolean; message: Message }
  | { type: "llm_response"; call_id: string; message: Message; truncated: boolean }
  | { type: "llm_error"; call_id: string; error: string }
  | {
      type: "tool_execute";
      tool_call_id: string;
      name: string;
      arguments: string;
      attempt: number;
      deadline?: DateTime;
    }
  | { type: "tool_result"; result: ToolResult }
  | { type: "sub_agent_error"; session_id: Uuid; agent_id: string; error: string }
  | { type: "interrupt_resumed"; interrupt_id: string }
  | { type: "stall" };

// ── Worker Actions ──────────────────────────────────────────────────────────

export type WorkerAction =
  | {
      type: "call_llm";
      request: LlmRequest;
      stream: boolean;
      llm_client: string;
      retry: RetryPolicy;
    }
  | {
      type: "call_tool";
      tool_call_id: string;
      name: string;
      arguments: string;
      handler: ToolHandler;
      retry: RetryPolicy;
    }
  | { type: "return_tool_result"; tool_call_id: string; result: string; attempt: number }
  | {
      type: "return_tool_error";
      tool_call_id: string;
      error: string;
      retryable: boolean;
      attempt: number;
    }
  | { type: "resolve_remote_tool"; session_id: Uuid; tool_call_id: string; result: string }
  | { type: "spawn_sub_agent"; session_id: Uuid; agent_id: string; retry: RetryPolicy }
  | { type: "send_message"; session_id: Uuid; message: Message }
  | { type: "done"; artifacts: Artifact[] };

// ── Event Payloads ──────────────────────────────────────────────────────────

export interface SessionCreated {
  type: "session.created";
  agent_id: string;
  auth: ClientIdentity;
  ancestry?: Uuid[];
  worker_retry: RetryPolicy;
}

export interface SessionDone {
  type: "session.done";
  artifacts?: Artifact[];
}

export interface SessionCancelled {
  type: "session.cancelled";
}

export interface NewMessage {
  type: "message.new";
  message: Message;
}

export interface LlmCallRequested {
  type: "llm.call.requested";
  call_id: string;
  request: LlmRequest;
  stream: boolean;
  llm_client: string;
  retry: RetryPolicy;
}

export interface LlmCallCompleted {
  type: "llm.call.completed";
  call_id: string;
  response: LlmResponse;
}

export interface LlmCallErrored {
  type: "llm.call.errored";
  call_id: string;
  error: string;
  retryable: boolean;
  source?: unknown;
}

export interface ToolCallRequested {
  type: "tool.call.requested";
  tool_call_id: string;
  name: string;
  arguments: string;
  handler: ToolHandler;
  retry: RetryPolicy;
}

export interface ToolCallCompleted {
  type: "tool.call.completed";
  tool_call_id: string;
  name: string;
  result: string;
}

export interface ToolCallErrored {
  type: "tool.call.errored";
  tool_call_id: string;
  name: string;
  error: string;
  retryable: boolean;
}

export interface SubAgentRequested {
  type: "sub_agent.requested";
  session_id: Uuid;
  agent_id: string;
  retry: RetryPolicy;
}

export interface SubAgentStarted {
  type: "sub_agent.started";
  session_id: Uuid;
}

export interface SubAgentErrored {
  type: "sub_agent.errored";
  session_id: Uuid;
  error: string;
  retryable: boolean;
}

export interface SessionInterrupted {
  type: "session.interrupted";
  interrupt_id: string;
  reason: string;
  payload: unknown;
}

export interface InterruptResumed {
  type: "session.interrupt_resumed";
  interrupt_id: string;
  payload: unknown;
}

export interface WorkerDecisionRequested {
  type: "worker.decision.requested";
  decision_id: string;
  trigger: DecisionTrigger;
}

export interface WorkerDecisionCompleted {
  type: "worker.decision.completed";
  decision_id: string;
  /** Base64-encoded opaque worker state */
  state: string;
}

export interface WorkerDecisionErrored {
  type: "worker.decision.errored";
  decision_id: string;
  error: string;
  retryable: boolean;
}

export interface SessionMessageRequested {
  type: "session.message_requested";
  target_session_id: Uuid;
  message: Message;
}

export interface ToolCallResolutionRequested {
  type: "tool_call.resolution_requested";
  target_session_id: Uuid;
  tool_call_id: string;
  result: string;
}

export interface WorkerStateUpdated {
  type: "worker.state.updated";
  /** Base64-encoded opaque worker state */
  state: string;
}

// ── Event (tagged union) ────────────────────────────────────────────────────

export type EventPayload =
  | SessionCreated
  | NewMessage
  | LlmCallRequested
  | LlmCallCompleted
  | LlmCallErrored
  | ToolCallRequested
  | ToolCallCompleted
  | ToolCallErrored
  | SubAgentRequested
  | SubAgentStarted
  | SubAgentErrored
  | SessionInterrupted
  | InterruptResumed
  | WorkerDecisionRequested
  | WorkerDecisionCompleted
  | WorkerDecisionErrored
  | SessionMessageRequested
  | ToolCallResolutionRequested
  | WorkerStateUpdated
  | SessionCancelled
  | SessionDone;

// ── Event Envelope ──────────────────────────────────────────────────────────

export interface Event {
  id: Uuid;
  tenant_id: string;
  aggregate_type: string;
  aggregate_id: Uuid;
  sequence: number;
  span: SpanContext;
  occurred_at: DateTime;
  payload: EventPayload;
  derived?: unknown;
  metadata?: Record<string, string>;
  start_time: DateTime;
  end_time: DateTime;
}

// ── Session State ───────────────────────────────────────────────────────────

export type SessionStatus =
  | "idle"
  | { interrupted: { interrupt_id: string } }
  | "done";

export type EffectStatus = "pending" | "completed" | "failed" | "retry_scheduled";

export interface EffectTracking {
  status: EffectStatus;
  retry: RetryState;
  retry_policy: RetryPolicy;
  deadline?: DateTime;
}

export interface LlmCallState {
  call_id: string;
  tracking: EffectTracking;
  request: LlmRequest;
  stream: boolean;
  llm_client: string;
}

export interface ToolCallState {
  tool_call_id: string;
  name: string;
  tracking: EffectTracking;
  handler: ToolHandler;
  arguments: string;
  result?: string;
  is_error: boolean;
}

export interface SubAgentCallState {
  session_id: Uuid;
  agent_id: string;
  tracking: EffectTracking;
}

export interface WorkerDecisionState {
  decision_id: string;
  tracking: EffectTracking;
  trigger: DecisionTrigger;
}

export interface SessionState {
  session_id: Uuid;
  status: SessionStatus;
  agent_id?: string;
  auth?: ClientIdentity;
  token_usage: Record<string, number>;
  cost: Decimal;
  /** Base64-encoded opaque worker state */
  worker_state: string;
  ancestry?: Uuid[];
  artifacts?: Artifact[];
  worker_retry?: RetryPolicy;
  llm_calls: Record<string, LlmCallState>;
  tool_calls: Record<string, ToolCallState>;
  sub_agent_calls: Record<string, SubAgentCallState>;
  worker_decisions: Record<string, WorkerDecisionState>;
}

// ── Client HTTP API ─────────────────────────────────────────────────────────

export interface SendMessageRequest {
  agent_id: string;
  message: string;
  tenant_id?: string;
  session_id?: Uuid;
}

// ── Worker HTTP API ─────────────────────────────────────────────────────────

export interface WorkerDecisionRequestWire {
  session_id: Uuid;
  tenant_id: string;
  decision_id: string;
  agent_id: string;
  auth: ClientIdentity;
  trigger: DecisionTrigger;
  /** Base64-encoded opaque worker state */
  worker_state: string;
  ancestry?: Uuid[];
  span: SpanContext;
  attempts: number;
  deadline?: DateTime;
}

export interface SubmitRequest {
  session_id: Uuid;
  tenant_id: string;
  decision_id: string;
  actions: WorkerAction[];
  /** Base64-encoded opaque worker state */
  state: string;
  span?: SpanContext;
}

export interface SubmitResponse {
  ok: boolean;
  error?: string;
}

export interface RegisterRequest {
  tenant_id: string;
  agent_ids: string[];
  transport_type: string;
  config: unknown;
}

export interface RegisterResponse {
  ok: boolean;
}
