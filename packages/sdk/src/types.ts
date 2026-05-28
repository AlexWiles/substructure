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

/**
 * The full identity shape as seen in the event log and admin views. `id` is
 * optional because the underlying engine technically permits it. In practice
 * every public entry point (HTTP, embedded) rejects a missing `id` before a
 * session is created.
 */
export interface ClientIdentity {
    tenant_id: string;
    id?: string;
    metadata?: Record<string, string>;
}

/**
 * Narrowed identity for the worker-facing surface. `id` is required because
 * any decision your worker receives came from a session created via an entry
 * point that enforced a non-empty `id`.
 */
export interface WorkerIdentity {
    tenant_id: string;
    id: string;
    metadata?: Record<string, string>;
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

// ── Multimodal content parts (OpenAI/OpenRouter wire format) ──────────

export interface ImageUrlPart {
    type: "image_url";
    image_url: { url: string };
}

export interface FilePart {
    type: "file";
    file: { filename: string; file_data: string };
}

export interface InputAudioPart {
    type: "input_audio";
    input_audio: { data: string; format: string };
}

export interface VideoUrlPart {
    type: "video_url";
    video_url: { url: string };
}

export interface TextPart {
    type: "text";
    text: string;
}

export type ContentPart = TextPart | ImageUrlPart | FilePart | InputAudioPart | VideoUrlPart;

/** Message content: plain string or array of typed parts. */
export type Content = string | ContentPart[];

export interface Message {
    role: Role;
    content?: Content;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
    name?: string;
}

/** Extract concatenated text from message content. */
export function contentText(content: Content | undefined): string {
    if (content === undefined) return "";
    if (typeof content === "string") return content;
    return content
        .filter((p): p is TextPart => p.type === "text")
        .map((p) => p.text)
        .join("\n");
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

export interface ResponseImage {
    url: string;
}

export interface LlmResponse {
    model: string;
    content?: string;
    tool_calls: ToolCall[];
    finish_reason?: string;
    usage?: Record<string, unknown>;
    cost?: Decimal;
    images?: ResponseImage[];
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

export interface ClientAction {
    name: string;
    args?: unknown;
}

export type ClientPayload =
    | { type: "message"; message: Message; stream?: boolean }
    | ({ type: "action" } & ClientAction);

export type DecisionTrigger =
    | { type: "user.message"; stream: boolean; message: Message }
    | ({ type: "client.action" } & ClientAction)
    | {
          type: "llm.response";
          call_id: string;
          message: Message;
          truncated: boolean;
          usage?: Record<string, unknown>;
          cost?: Decimal;
      }
    | { type: "llm.error"; call_id: string; error: string; code?: string; detail?: unknown }
    | {
          type: "tool.execute";
          tool_call_id: string;
          name: string;
          arguments: string;
          attempt: number;
          deadline?: DateTime;
      }
    | { type: "tool.result"; result: ToolResult }
    | { type: "sub_agent.turn.complete"; session_id: Uuid; agent_id: string; turn_id: string; data: unknown }
    | { type: "sub_agent.error"; session_id: Uuid; agent_id: string; error: string }
    | { type: "interrupt.resumed"; interrupt_id: string }
    | { type: "stall" };

// ── Worker Actions ──────────────────────────────────────────────────────────

export type WorkerAction =
    | {
          type: "call.llm";
          request: LlmRequest;
          stream: boolean;
          retry: RetryPolicy;
      }
    | {
          type: "call.tool";
          tool_call_id: string;
          name: string;
          arguments: string;
          handler: ToolHandler;
          retry: RetryPolicy;
      }
    | { type: "return.tool.result"; tool_call_id: string; result: string; attempt: number }
    | {
          type: "return.tool.error";
          tool_call_id: string;
          error: string;
          retryable: boolean;
          attempt: number;
      }
    | { type: "spawn.sub_agent"; session_id: Uuid; agent_id: string; retry: RetryPolicy }
    | { type: "send.message"; session_id: Uuid; message: Message }
    | { type: "done"; data: unknown };

export type SubmitToolCallResultRequest = Extract<WorkerAction, { type: "return.tool.result" | "return.tool.error" }>;

export interface SubmitToolCallResultResponse {
    ok: boolean;
    error?: string;
}

/** Identifies a single deferred tool call. */
export interface SubmitToolCallResultTarget {
    sessionId: string;
    toolCallId: string;
    attempt: number;
}

/** Successful completion of a deferred tool call. */
export interface SubmitToolCallSuccess {
    result: string;
    error?: never;
    retryable?: never;
}

/** Failed completion of a deferred tool call. */
export interface SubmitToolCallFailure {
    error: string;
    retryable?: boolean;
    result?: never;
}

/**
 * Outcome of a deferred tool call: either a successful result or a
 * failure. The `never`-typed alternates make this a discriminated union,
 * so passing both — or neither — is a compile error.
 */
export type SubmitToolCallResultOutcome = SubmitToolCallSuccess | SubmitToolCallFailure;

/** Arguments for completing a deferred tool call. */
export type SubmitToolCallResultArgs = SubmitToolCallResultTarget & SubmitToolCallResultOutcome;

/** Translate the ergonomic args object into the wire shape. */
export function toSubmitToolCallResultRequest(args: SubmitToolCallResultArgs): SubmitToolCallResultRequest {
    if (args.result !== undefined) {
        return {
            type: "return.tool.result",
            tool_call_id: args.toolCallId,
            result: args.result,
            attempt: args.attempt,
        };
    }
    return {
        type: "return.tool.error",
        tool_call_id: args.toolCallId,
        error: args.error,
        retryable: args.retryable ?? false,
        attempt: args.attempt,
    };
}

// ── Event Payloads ──────────────────────────────────────────────────────────

export interface SessionCreated {
    type: "session.created";
    agent_id: string;
    identity: ClientIdentity;
    ancestry?: Uuid[];
    worker_retry: RetryPolicy;
}

/** Pure lifecycle marker — no data. Turn output lives in TurnCompleted. */
export interface SessionDone {
    type: "session.done";
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
    code?: string;
    detail?: unknown;
}

/** Transient LLM token delta. Not persisted — clients that connect mid-call
 *  will not see deltas already emitted for that call. The full assembled
 *  content always arrives via the persisted `llm.call.completed` event,
 *  which is followed by a `message.new` event with the canonical assistant
 *  message. Use `(call_id, attempt)` to join back to the originating
 *  `llm.call.requested` event; order chunks within a call by `seq`. */
export interface LlmTokenDelta {
    type: "llm.token.delta";
    call_id: string;
    attempt: number;
    /** Monotonic per-call sequence (0-based). */
    seq: number;
    agent_id: string;
    turn_id?: string;
    text?: string;
    finish_reason?: string;
}

export interface ToolCallRequested {
    type: "tool.call.requested";
    tool_call_id: string;
    attempt: number;
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

export interface SubAgentTurnCompleted {
    type: "sub_agent.turn_completed";
    session_id: Uuid;
    cost: Decimal;
    token_usage?: Record<string, number>;
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

export interface WorkerStateUpdated {
    type: "worker.state.updated";
    /** Base64-encoded opaque worker state */
    state: string;
}

export interface TurnStarted {
    type: "turn.started";
    turn_id: string;
}

export interface TurnCompleted {
    type: "turn.completed";
    turn_id: string;
    data?: unknown;
    turn_cost?: Decimal;
    turn_token_usage?: Record<string, number>;
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
    | SubAgentTurnCompleted
    | SessionInterrupted
    | InterruptResumed
    | WorkerDecisionRequested
    | WorkerDecisionCompleted
    | WorkerDecisionErrored
    | SessionMessageRequested
    | WorkerStateUpdated
    | SessionCancelled
    | SessionDone
    | TurnStarted
    | TurnCompleted;

// ── Event Envelope ──────────────────────────────────────────────────────────

export interface DerivedState {
    status: SessionStatus;
    agent_id?: string;
    cost: Decimal;
    sub_agent_cost: Decimal;
    turn_cost: Decimal;
    turn_token_usage?: Record<string, number>;
    token_usage?: Record<string, number>;
    sub_agent_token_usage?: Record<string, number>;
    turn_id?: string;
}

/** A persisted session event from the event store. Carries a monotonic
 *  `sequence` usable with `sequence_after` for resume. */
export interface PersistedEvent {
    id: Uuid;
    tenant_id: string;
    aggregate_type: string;
    aggregate_id: Uuid;
    sequence: number;
    event_type: string;
    span: SpanContext;
    occurred_at: DateTime;
    payload: EventPayload;
    derived?: DerivedState;
    metadata?: Record<string, string>;
    start_time: DateTime;
    end_time: DateTime;
}

/** A transient LLM token delta envelope. Same envelope shape as a persisted
 *  event but slimmer — not persisted, no `sequence`, not replayed on
 *  reconnect. */
export interface TokenDeltaEvent {
    aggregate_type: string;
    aggregate_id: Uuid;
    event_type: "llm.token.delta";
    occurred_at: DateTime;
    payload: LlmTokenDelta;
}

/** Discriminate on `payload.type`: `"llm.token.delta"` ⇒ `TokenDeltaEvent`,
 *  anything else ⇒ `PersistedEvent`. */
export type Event = PersistedEvent | TokenDeltaEvent;

// ── Turns ───────────────────────────────────────────────────────────────────

/** Scope of events to observe within a session. With `turnId` omitted, the
 *  scope is the whole session. `startTurn` always returns one with `turnId`
 *  set; `turnResult` requires it at runtime. */
export interface SessionScope {
    sessionId: string;
    turnId?: string;
}

export interface TurnResult {
    turnId: string;
    data: unknown;
    cost: Decimal;
    tokenUsage: Record<string, number>;
}

/** Drain a turn event stream to completion and return the turn result.
 *  Throws if the stream ends without a `turn.completed` event. */
export async function drainToTurnResult(stream: AsyncIterable<Event>): Promise<TurnResult> {
    let completed: TurnCompleted | undefined;
    for await (const event of stream) {
        if (event.payload.type === "turn.completed") {
            completed = event.payload;
        }
    }
    if (!completed) {
        throw new Error("stream ended without turn.completed");
    }
    return {
        turnId: completed.turn_id,
        data: completed.data,
        cost: completed.turn_cost ?? "0",
        tokenUsage: completed.turn_token_usage ?? {},
    };
}

// ── Session State ───────────────────────────────────────────────────────────

export type SessionStatus = "idle" | { interrupted: { interrupt_id: string } } | "done";

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
    identity?: ClientIdentity;
    token_usage: Record<string, number>;
    cost: Decimal;
    sub_agent_cost: Decimal;
    turn_cost: Decimal;
    turn_token_usage?: Record<string, number>;
    sub_agent_token_usage?: Record<string, number>;
    /** Base64-encoded opaque worker state */
    worker_state: string;
    ancestry?: Uuid[];
    data?: unknown;
    worker_retry?: RetryPolicy;
    llm_calls: Record<string, LlmCallState>;
    tool_calls: Record<string, ToolCallState>;
    sub_agent_calls: Record<string, SubAgentCallState>;
    worker_decisions: Record<string, WorkerDecisionState>;
}

// ── Client HTTP API ─────────────────────────────────────────────────────────

export interface SubmitPayloadRequest {
    agent_id: string;
    payload: ClientPayload;
    /** Required for embedded runtime; ignored/forbidden on remote HTTP submit. */
    identity?: ClientIdentity;
    tenant_id?: string;
    session_id?: Uuid;
    turn_id?: string;
}

export interface MintClientTokenRequest {
    identity: {
        id: string;
        metadata?: Record<string, string>;
    };
    ttl_seconds?: number;
}

export interface MintClientTokenResponse {
    token: string;
    expires_at: number;
}

export interface MachineSubmitPayloadRequest {
    agent_id: string;
    payload: ClientPayload;
    session_id?: Uuid;
    turn_id?: string;
    identity: {
        id: string;
        metadata?: Record<string, string>;
    };
}

export interface SubmitClientPayloadResponse {
    session_id: Uuid;
    turn_id: string;
}

export interface StreamSessionEventsParams {
    turn_id?: string;
    sequence_after?: number;
}

// ── Worker HTTP API ─────────────────────────────────────────────────────────

export interface WorkerAuthOptions {
    bearerToken: string;
}

export interface WorkerDecisionRequestWire {
    session_id: Uuid;
    tenant_id: string;
    decision_id: string;
    agent_id: string;
    identity: WorkerIdentity;
    trigger: DecisionTrigger;
    worker_state: string;
    ancestry?: Uuid[];
    span: SpanContext;
    attempts: number;
    deadline?: DateTime;
    turn_id?: string;
}

export interface SubmitRequest {
    session_id: Uuid;
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
