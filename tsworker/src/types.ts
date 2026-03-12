/** Wire format types matching the substructure2 Rust runtime JSON format. */

// --- Messages ---

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

// --- LLM request ---

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

// --- Tool results ---

export interface ToolResult {
  tool_call_id: string;
  name: string;
  content: string;
  is_error?: boolean;
}

// --- Artifacts ---

export interface TextPart {
  kind: "text";
  text: string;
}

export interface DataPart {
  kind: "data";
  data: unknown;
}

export type Part = TextPart | DataPart;

export interface Artifact {
  name?: string;
  description?: string;
  parts: Part[];
}

// --- Retry policy ---

export interface RetryPolicy {
  timeout_secs?: number;
  max_retries: number;
  backoff_base_secs: number;
  backoff_max_secs: number;
}

export const RetryPolicies = {
  noRetry(): RetryPolicy {
    return { max_retries: 0, backoff_base_secs: 0, backoff_max_secs: 0 };
  },
  default(): RetryPolicy {
    return {
      timeout_secs: 30,
      max_retries: 3,
      backoff_base_secs: 2,
      backoff_max_secs: 60,
    };
  },
} as const;

// --- Tool handler ---

export type ToolHandler = "worker" | "client" | "sub_agent";

// --- Decision triggers (discriminated union on "type") ---

export interface UserMessageTrigger {
  type: "user_message";
  stream: boolean;
  message: Message;
}

export interface LlmCompletedTrigger {
  type: "llm_completed";
  call_id: string;
  message: Message;
  truncated?: boolean;
}

export interface LlmFailedTrigger {
  type: "llm_failed";
  call_id: string;
  error: string;
}

export interface ToolResolvedTrigger {
  type: "tool_resolved";
  result: ToolResult;
}

export interface InterruptResumedTrigger {
  type: "interrupt_resumed";
  interrupt_id: string;
}

export interface StallTrigger {
  type: "stall";
}

export type DecisionTrigger =
  | UserMessageTrigger
  | LlmCompletedTrigger
  | LlmFailedTrigger
  | ToolResolvedTrigger
  | InterruptResumedTrigger
  | StallTrigger;

// --- Worker actions (discriminated union on "type") ---

export interface RequestLlmAction {
  type: "request_llm";
  request: LlmRequest;
  stream: boolean;
  llm_client: string;
  retry: RetryPolicy;
}

export interface RequestToolCallAction {
  type: "request_tool_call";
  tool_call_id: string;
  name: string;
  arguments: string;
  handler: ToolHandler;
  retry: RetryPolicy;
}

export interface RequestSubAgentAction {
  type: "request_sub_agent";
  agent_id: string;
  message: string;
  retry: RetryPolicy;
}

export interface DoneAction {
  type: "done";
  artifacts: Artifact[];
}

export type WorkerAction =
  | RequestLlmAction
  | RequestToolCallAction
  | RequestSubAgentAction
  | DoneAction;

// --- Span context ---

export interface SpanContext {
  trace_id: string;
  span_id: string;
  parent_span_id?: string;
  trace_flags?: number;
  trace_state?: string;
  name?: string;
}

// --- HTTP wire types ---

export interface PollRequest {
  tenant_id: string;
  agent_ids: string[];
  timeout_ms?: number;
}

export interface PollResponse {
  session_id: string;
  tenant_id: string;
  decision_id: string;
  agent_id: string;
  trigger: DecisionTrigger;
  worker_state: string; // base64-encoded
  span: SpanContext;
  attempts?: number;
  deadline?: string;
}

export interface SubmitRequest {
  session_id: string;
  tenant_id: string;
  decision_id: string;
  actions: WorkerAction[];
  state: string; // base64-encoded
  span?: SpanContext;
}

export interface SubmitResponse {
  ok: boolean;
  error?: string;
}

// --- Client API types ---

export interface SendMessageRequest {
  agent_id: string;
  message: string;
  tenant_id?: string;
  session_id?: string;
}

// --- Session event types (NDJSON stream from /sessions/send) ---

export interface SessionCreatedPayload {
  type: "session.created";
  agent_id: string;
}

export interface NewMessagePayload {
  type: "message.new";
  message: Message;
}

export interface LlmCallRequestedPayload {
  type: "llm.call.requested";
  call_id: string;
  request: LlmRequest;
  stream: boolean;
  llm_client?: string;
  retry: RetryPolicy;
}

export interface LlmCallCompletedPayload {
  type: "llm.call.completed";
  call_id: string;
  response: {
    model: string;
    content?: string;
    tool_calls?: ToolCall[];
    finish_reason?: string;
    usage?: unknown;
    cost?: string;
  };
}

export interface LlmCallErroredPayload {
  type: "llm.call.errored";
  call_id: string;
  error: string;
  retryable: boolean;
}

export interface ToolCallRequestedPayload {
  type: "tool.call.requested";
  tool_call_id: string;
  name: string;
  arguments: string;
  handler?: ToolHandler;
  retry: RetryPolicy;
}

export interface ToolCallCompletedPayload {
  type: "tool.call.completed";
  tool_call_id: string;
  name: string;
  result: string;
}

export interface ToolCallErroredPayload {
  type: "tool.call.errored";
  tool_call_id: string;
  name: string;
  error: string;
  retryable: boolean;
}

export interface WorkerDecisionRequestedPayload {
  type: "worker.decision.requested";
  decision_id: string;
  trigger: DecisionTrigger;
}

export interface WorkerDecisionCompletedPayload {
  type: "worker.decision.completed";
  decision_id: string;
  state: string;
}

export interface WorkerDecisionErroredPayload {
  type: "worker.decision.errored";
  decision_id: string;
  error: string;
  retryable: boolean;
}

export interface SessionDonePayload {
  type: "session.done";
  artifacts?: Artifact[];
}

export interface SessionCancelledPayload {
  type: "session.cancelled";
}

export type EventPayload =
  | SessionCreatedPayload
  | NewMessagePayload
  | LlmCallRequestedPayload
  | LlmCallCompletedPayload
  | LlmCallErroredPayload
  | ToolCallRequestedPayload
  | ToolCallCompletedPayload
  | ToolCallErroredPayload
  | WorkerDecisionRequestedPayload
  | WorkerDecisionCompletedPayload
  | WorkerDecisionErroredPayload
  | SessionDonePayload
  | SessionCancelledPayload;

export interface SessionEvent {
  id: string;
  tenant_id: string;
  aggregate_type: string;
  aggregate_id: string;
  sequence: number;
  span: SpanContext;
  occurred_at: string;
  payload: EventPayload;
}
