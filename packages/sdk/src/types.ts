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

export interface SpanContext {
    trace_id: TraceId;
    span_id: SpanId;
    parent_span_id?: SpanId;
    trace_flags: number;
    trace_state?: string;
    name?: string;
}

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

export interface ToolCallChunk {
    id: string;
    name?: string;
    arguments?: string;
}

// Multimodal content parts use the OpenAI/OpenRouter wire format.
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

export type Content = string | ContentPart[];

export interface Message {
    /** Node id. The engine stamps one on every message it records and hands back,
     *  so messages you read — transcript, triggers, tree — always carry it. */
    id: string;
    role: Role;
    content?: Content;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
    name?: string;
}

/** A message you construct — to submit, prompt, or seed a session. The engine
 *  assigns the node id when it records it, so `id` is optional here; omit it on
 *  messages you create and the engine fills it in. A read `Message` is also a
 *  valid `MessageInput`. */
export type MessageInput = Omit<Message, "id"> & { id?: string };

export function contentText(content: Content | undefined): string {
    if (content === undefined) return "";
    if (typeof content === "string") return content;
    return content
        .filter((p): p is TextPart => p.type === "text")
        .map((p) => p.text)
        .join("\n");
}

export interface LlmToolFunction {
    name: string;
    description: string;
    parameters: unknown;
}

export interface LlmTool {
    function: LlmToolFunction;
}

/** Model id and per-call parameters: the reusable core of an LLM call, shared by
 *  `LlmRequest` (the wire payload) and `Llm` (the loop's config). */
export interface LlmParams {
    model: string;
    temperature?: number;
    max_completion_tokens?: number;
    /** Reasoning / thinking controls, passed through to providers that support
     *  it. `effort` and `max_tokens` are mutually exclusive. */
    reasoning?: ReasoningConfig;
}

export interface LlmRequest extends LlmParams {
    messages: MessageInput[];
    tools?: LlmTool[];
}

/** Mirrors OpenRouter's unified `reasoning` parameter. */
export interface ReasoningConfig {
    /** OpenAI / Grok style. Mutually exclusive with `max_tokens`. */
    effort?: "xhigh" | "high" | "medium" | "low" | "minimal" | "none";
    /** Anthropic / Gemini / Qwen style token budget. Mutually exclusive with `effort`. */
    max_tokens?: number;
    /** Reason internally but omit reasoning from the response (default false). */
    exclude?: boolean;
    /** Shorthand to enable reasoning at the provider's default (medium) effort. */
    enabled?: boolean;
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

export type ToolHandler = "worker" | "client";

export type LlmHandler = "server" | "worker";

export interface ToolResult {
    tool_call_id: string;
    name: string;
    content: string;
}

/** The message's `id` is the node id; `parent_id` is absent only for the thread root. */
export interface MessageNode {
    parent_id?: string;
    message: Message;
}

/** A control marker (interrupt/resume); filtered out of LLM prompts, surfaced to clients as run outcome. */
export interface Control {
    id: string;
    interrupt_id: string;
    kind: "interrupt" | "resume";
    reason?: string;
    payload?: unknown;
    origin: InterruptOrigin;
}

export interface ControlNode {
    parent_id?: string;
    control: Control;
}

/** A tree node: a conversation message or a control marker. */
export type Node = ({ kind: "message" } & MessageNode) | ({ kind: "control" } & ControlNode);

/** Tree nodes plus `head_id`, the active leaf; the active transcript is its path to root. */
export interface MessageTree {
    nodes: Node[];
    head_id?: string;
}

/** The node id, regardless of node kind. */
export function nodeId(node: Node): string | undefined {
    return node.kind === "message" ? node.message.id : node.control.id;
}

export interface ClientAction {
    name: string;
    args?: unknown;
}

export type ClientPayload =
    | { type: "message"; message: MessageInput; stream?: boolean }
    | { type: "messages"; messages: MessageInput[]; stream?: boolean }
    | ({ type: "action" } & ClientAction);

/** How a finished llm call landed: the response fields when the trigger's
 *  `ok` is true, the error fields when it is false. */
export type LlmOutcome =
    | {
          message: Message;
          truncated: boolean;
          usage?: Record<string, unknown>;
          cost?: Decimal;
          error?: never;
      }
    | { error: string; code?: string; detail?: unknown; message?: never };

/** What the engine sends in `trigger`: news that hasn't been written into the
 *  conversation yet. `*.finished` triggers are the final word on a call —
 *  uniform across subjects: the payload when `ok`, `error` when not. A
 *  `sub_agent.finished`'s `id` is the model tool call the delegation answers,
 *  like a tool's; `session_id` is the child session. */
export type DecisionTrigger =
    | { type: "client.message"; message: Message }
    | { type: "client.transcript"; messages: Message[] }
    | ({ type: "client.action" } & ClientAction)
    | { type: "tool.execute"; id: string; name: string; arguments: string; attempt: number; deadline?: DateTime }
    | { type: "llm.execute"; id: string; request: LlmRequest; stream: boolean; attempt: number; deadline?: DateTime }
    | { type: "tool.finished"; id: string; ok: boolean; name: string; result?: string; error?: string }
    | {
          type: "sub_agent.finished";
          id: string;
          ok: boolean;
          session_id: Uuid;
          agent_id: string;
          result?: string;
          error?: string;
      }
    | ({ type: "llm.finished"; id: string; ok: boolean } & LlmOutcome)
    | { type: "interrupt.resumed"; interrupt_id: string; payload?: unknown };

/** The call kinds a worker can be delegated via `*.execute`. */
export type WorkKind = "tool_call" | "llm_call";

export type WorkerAction =
    | {
          /** `id` names the call; its outcome returns as an `llm.finished` trigger
           *  with the same id. Reuse of a pending/completed id is an idempotent no-op. */
          type: "llm.call";
          id: string;
          request: LlmRequest;
          stream?: boolean;
          retry?: RetryPolicy;
          handler: LlmHandler;
      }
    | {
          /** `id` names the call (the model's tool call id); its outcome
           *  comes back as a `tool.finished` trigger with the same id. */
          type: "tool.call";
          id: string;
          name: string;
          arguments: string;
          handler: ToolHandler;
          retry?: RetryPolicy;
      }
    | { type: "tool.result"; id: string; result: string; attempt?: number }
    | { type: "llm.result"; id: string; response: LlmResponse; attempt?: number }
    | {
          type: "tool.error";
          id: string;
          error: string;
          retryable: boolean;
          attempt?: number;
          code?: string;
          detail?: unknown;
      }
    | {
          type: "llm.error";
          id: string;
          error: string;
          retryable: boolean;
          attempt?: number;
          code?: string;
          detail?: unknown;
      }
    | { type: "sub_agent.spawn"; session_id: Uuid; agent_id: string; tool_call_id: string; retry?: RetryPolicy }
    | { type: "message.send"; session_id: Uuid; message: MessageInput }
    | { type: "interrupt"; interrupt_id?: string; reason: string; payload?: unknown }
    | { type: "done"; data: unknown };

/** The settle body: a `tool.result`/`llm.result` or `tool.error`/`llm.error`.
 *  The worker surface accepts all four; the client surface only the `tool.*` pair. */
export type SettleEffectRequest =
    | { type: "tool.result"; id: string; result: string; attempt?: number }
    | { type: "llm.result"; id: string; response: LlmResponse; attempt?: number }
    | {
          type: "tool.error";
          id: string;
          error: string;
          retryable: boolean;
          attempt?: number;
          code?: string;
          detail?: unknown;
      }
    | {
          type: "llm.error";
          id: string;
          error: string;
          retryable: boolean;
          attempt?: number;
          code?: string;
          detail?: unknown;
      };

export interface SettleEffectResponse {
    ok: boolean;
    error?: string;
}

export interface InterruptSessionRequest {
    interrupt_id?: string;
    reason?: string;
    payload?: unknown;
}

export interface InterruptSessionResponse {
    ok: boolean;
    interrupt_id: string;
}

export interface ResumeInterruptRequest {
    interrupt_id: string;
    payload?: unknown;
}

export interface ResumeInterruptResponse {
    ok: boolean;
}

/** Which effect to settle, on which session. `id` is the effect id (a tool
 *  call id or an llm call id). `attempt` is optional: omitted settles the
 *  current attempt; echo the trigger's `attempt` to fence a stale executor. */
export interface SettleEffectTarget {
    sessionId: string;
    id: string;
    attempt?: number;
}

/** Settle a `tool_call` with its result string. */
export interface SettleToolResult {
    kind?: "tool_call";
    result: string;
    response?: never;
    error?: never;
    retryable?: never;
}

/** Settle an `llm_call` with its response. */
export interface SettleLlmResult {
    kind: "llm_call";
    response: LlmResponse;
    result?: never;
    error?: never;
    retryable?: never;
}

/** Fail an effect of either kind. */
export interface SettleEffectFailure {
    kind?: WorkKind;
    error: string;
    retryable?: boolean;
    result?: never;
    response?: never;
}

/**
 * Outcome of an out-of-band settlement: a tool result, an llm response, or a
 * failure. The `never`-typed alternates make this a discriminated union, so
 * mixing fields from more than one shape is a compile error.
 */
export type SettleEffectOutcome = SettleToolResult | SettleLlmResult | SettleEffectFailure;

export type SettleEffectArgs = SettleEffectTarget & SettleEffectOutcome;

/** The tool-call-only outcome the frontend/user surface accepts: clients
 *  answer client tools, never model calls, so `kind: "llm_call"` is unexpressible. */
export type SettleToolCallOutcome = SettleToolResult | (SettleEffectFailure & { kind?: "tool_call" });
export type SettleToolCallArgs = SettleEffectTarget & SettleToolCallOutcome;

export function toSettleEffectRequest(args: SettleEffectArgs): SettleEffectRequest {
    if (args.result !== undefined) {
        return {
            type: "tool.result",
            id: args.id,
            result: args.result,
            attempt: args.attempt,
        };
    }
    if (args.response !== undefined) {
        return {
            type: "llm.result",
            id: args.id,
            response: args.response,
            attempt: args.attempt,
        };
    }
    return {
        type: (args.kind ?? "tool_call") === "llm_call" ? "llm.error" : "tool.error",
        id: args.id,
        error: args.error,
        retryable: args.retryable ?? false,
        attempt: args.attempt,
    };
}

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

export interface NewMessage extends MessageNode {
    type: "message.new";
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

/** Transient LLM token delta. Not persisted — reconnecting mid-call skips
 *  ahead to `llm.call.completed` / `message.new` for the canonical content.
 *  Join to `llm.call.requested` via `(call_id, attempt)`; order chunks by `seq`. */
export interface LlmTokenDelta {
    type: "llm.token.delta";
    session_id: Uuid;
    agent_id: string;
    turn_id?: string;
    call_id: string;
    attempt: number;
    /** Monotonic per-call sequence (0-based). */
    seq: number;
    text?: string;
    reasoning?: string;
    tool_calls?: ToolCallChunk[];
    finish_reason?: string;
}

export interface LlmTokenDeltaInput {
    text?: string;
    reasoning?: string;
    tool_calls?: ToolCallChunk[];
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
    tool_call_id?: string;
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
    data?: unknown;
}

export type InterruptOrigin = "system" | "machine" | "frontend";

export interface SessionInterrupted {
    type: "session.interrupted";
    interrupt_id: string;
    origin: InterruptOrigin;
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
    message: MessageInput;
}

export interface WorkerStateUpdated {
    type: "worker.state.updated";
    /** Opaque worker state as raw JSON; the engine stores it without interpreting it. */
    state: unknown;
    /** The node id that was the active head when this version was written; absent if the tree was empty. */
    anchor?: string;
}

/** An outstanding call abandoned because its branch was forked away (or the
 *  session was interrupted or cancelled). */
export interface CallVoided {
    type: "call.voided";
    kind: "tool_call" | "llm_call" | "sub_agent";
    /** The call id — for a sub-agent, the model tool call the delegation answers. */
    id: string;
    /** The child session, present on `sub_agent` voids; the cancellation cascades to it. */
    session_id?: Uuid;
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
    | CallVoided
    | SessionCancelled
    | SessionDone
    | TurnStarted
    | TurnCompleted;

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

/** A persisted event. `sequence` is monotonic and resumable via `sequence_after`. */
export interface PersistedEvent {
    id: Uuid;
    tenant_id: string;
    aggregate_type: string;
    aggregate_id: Uuid;
    sequence: number;
    span: SpanContext;
    occurred_at: DateTime;
    payload: EventPayload;
    derived?: DerivedState;
    metadata?: Record<string, string>;
    start_time: DateTime;
    end_time: DateTime;
}

/** SSE stream item. Persisted events arrive wrapped in `PersistedEvent`;
 *  transient token deltas arrive bare. Discriminate with `isTokenDelta`. */
export type Event = PersistedEvent | LlmTokenDelta;

export function isTokenDelta(event: Event): event is LlmTokenDelta {
    return (event as LlmTokenDelta).type === "llm.token.delta";
}

/** Project a raw event stream down to persisted events, dropping transient
 *  token deltas. This is what `stream()` applies unless you pass
 *  `{ tokens: true }`. Exposed for callers wiring streams together by hand. */
export async function* persistedOnly(stream: AsyncIterable<Event>): AsyncGenerator<PersistedEvent> {
    for await (const event of stream) {
        if (!isTokenDelta(event)) yield event;
    }
}

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
        if (!isTokenDelta(event) && event.payload.type === "turn.completed") {
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

export type SessionStatus =
    | "idle"
    | { interrupted: { interrupt_id: string; origin: InterruptOrigin; reason: string } }
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

/** A worker-state write anchored to the tree position it was made at. The current
 *  state is the newest version whose anchor lies on the active path. */
export interface StateVersion {
    state: unknown;
    anchor?: string;
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
    state_versions: StateVersion[];
    ancestry?: Uuid[];
    data?: unknown;
    worker_retry?: RetryPolicy;
    llm_calls: Record<string, LlmCallState>;
    tool_calls: Record<string, ToolCallState>;
    sub_agent_calls: Record<string, SubAgentCallState>;
    worker_decisions: Record<string, WorkerDecisionState>;
}

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

export interface WorkerAuthOptions {
    bearerToken: string;
}

// In-flight calls surfaced on each worker decision as a flat, tagged list. `kind`
// is open, so a worker ignores kinds it doesn't handle.

export interface InFlightCallBase {
    id: string;
    status: EffectStatus;
    attempt: number;
    deadline?: DateTime;
    /** The tree node the call was requested at. A fork that leaves it off
     *  the active path voids the call (`call.voided`). */
    anchor?: string;
}

export interface InFlightToolCall extends InFlightCallBase {
    kind: "tool_call";
    name: string;
    arguments: string;
    handler: ToolHandler;
}

export interface InFlightSubAgent extends InFlightCallBase {
    kind: "sub_agent";
    agent_id: string;
    session_id: Uuid;
}

export interface InFlightLlmCall extends InFlightCallBase {
    kind: "llm_call";
    handler: LlmHandler;
    stream: boolean;
}

/** A call kind this SDK doesn't know; ignore it. The `never` fields keep
 *  narrowing on `kind` usable for the known variants. */
export interface UnknownInFlightCall extends InFlightCallBase {
    kind: string & {};
    name?: never;
    arguments?: never;
    agent_id?: never;
    session_id?: never;
    handler?: never;
    stream?: never;
}

export type InFlightCall = InFlightToolCall | InFlightSubAgent | InFlightLlmCall | UnknownInFlightCall;

export interface WorkerDecisionRequestWire {
    session_id: Uuid;
    decision_id: string;
    agent_id: string;
    identity: WorkerIdentity;
    trigger: DecisionTrigger;
    /** Your state as raw JSON, resolved from the active branch; `null` when the session has none. */
    state: unknown;
    calls: InFlightCall[];
    /** How many `tool_call`/`sub_agent` calls are still in flight: the step gate as a number (prompt at 0). */
    pending_calls: number;
    /** The active conversation as a flat list*/
    transcript?: Message[];
    /** The full tree, for clients that need branch structure. */
    message_tree?: MessageTree;
    ancestry?: Uuid[];
    attempts: number;
    deadline?: DateTime;
    turn_id?: string;
}

export interface SubmitRequest {
    session_id: Uuid;
    decision_id: string;
    actions: WorkerAction[];
    /** Flat conversation the engine reconciles into the message tree: known ids
     *  continue, id-less/unknown messages are appended (forking automatically). */
    transcript: MessageInput[];
    /** Opaque worker state as raw JSON; the engine stores it without interpreting it.
     *  Omitted (or `null`) = keep the current state; the engine can't tell a `null`
     *  from an omitted field. Clear with a present non-null empty value (e.g. `{}`). */
    state?: unknown;
}

export interface SubmitResponse {
    ok: boolean;
    error?: string;
}
