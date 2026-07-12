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
    /** Node id; present on every message the engine hands you. */
    id: string;
    role: Role;
    content?: Content;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
    name?: string;
}

/** A message you construct; `id` is optional (the engine assigns it). */
export type MessageInput = Omit<Message, "id"> & { id?: string };

export function contentText(content: Content | undefined): string {
    if (content === undefined) return "";
    if (typeof content === "string") return content;
    return content
        .filter((p): p is TextPart => p.type === "text")
        .map((p) => p.text)
        .join("\n");
}

/** A tool's declared contract: flat on the wire. */
export interface LlmTool {
    name: string;
    description: string;
    /** JSON Schema for the tool's arguments; omitted declares a no-argument
     *  tool. The engine validates each call's arguments against it and hands
     *  providers their native form. */
    input?: unknown;
    /** JSON Schema the settled result must satisfy; never sent to the model.
     *  A violating result settles as a terminal tool error. */
    output?: unknown;
}

/** The engine's classification of a tool call's arguments against the tool's
 *  declared `input` schema. Always present on a `tool.execute` trigger. */
export type ToolInput =
    | { status: "valid"; value: Record<string, unknown> }
    | { status: "invalid"; value: Record<string, unknown>; error: string }
    | { status: "malformed"; error: string };

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
    | { type: "client.message"; message: MessageInput; stream?: boolean }
    | { type: "client.messages"; messages: MessageInput[]; stream?: boolean }
    | ({ type: "client.action" } & ClientAction);

/** Everything a client can send to the input endpoint. The three submit shapes are a
 *  {@link ClientPayload} carrying its own addressing — `agent_id` (routes the turn,
 *  creating the session if new) and the optional idempotency `turn_id`; a resume/settle
 *  addresses an interrupt/effect id. `session_id` is the one universal address and rides
 *  the request envelope, so it is unmisplaceable. */
export type ClientInput =
    | (ClientPayload & { agent_id: string; turn_id?: string })
    | { type: "interrupt.resume"; interrupt_id: string; payload?: unknown }
    | { type: "tool.result"; id: string; result?: unknown; attempt?: number }
    | { type: "tool.error"; id: string; error: string; retryable: boolean; attempt?: number };

/** The one client input request body: the {@link ClientInput} plus the universal
 *  `session_id` address (minted when absent). */
export interface ClientInputRequest {
    session_id?: Uuid;
    input: ClientInput;
}

export interface ClientInputResponse {
    session_id: Uuid;
    turn_id: string;
}

/** Result of a finished llm call: response fields on success, error fields on failure. */
export type LlmOutcome =
    | {
          message: MessageInput;
          truncated: boolean;
          usage?: Record<string, unknown>;
          cost?: Decimal;
          error?: never;
      }
    | { error: string; code?: string; detail?: unknown; message?: never };

/** A tool a client declares on its run (AG-UI `tools`, normalized to the
 *  client-handled wire tool shape). */
export interface ClientTool {
    name: string;
    description?: string;
    input?: unknown;
    output?: unknown;
    handler?: ToolHandler;
}

/** Inputs a client declares on its run (the AG-UI `tools`/`context`/`state`/
 *  `forwardedProps`), forwarded to the worker on `client.messages`. The engine
 *  layers `tools` onto the proposed config by default; a worker may override. */
export interface ClientContext {
    tools?: ClientTool[];
    context?: unknown[];
    state?: unknown;
    forwarded_props?: unknown;
}

/** What a decision's `trigger` carries. A `sub_agent.finished`'s `id` is the
 *  tool call it answers; `session_id` is the child session. */
export type DecisionTrigger =
    | {
          /** The full proposed conversation; `messages[new_from..]` is unrecorded (ids optional there). */
          type: "client.messages";
          messages: MessageInput[];
          new_from: number;
          /** Inputs the client declared on its run; the engine layers `client.tools`
           *  onto the proposal by default. */
          client?: ClientContext;
      }
    | ({ type: "client.action" } & ClientAction)
    | {
          type: "tool.execute";
          id: string;
          name: string;
          arguments: string;
          /** The engine's classification of `arguments`: `valid` (with the
           *  parsed `value`), `invalid` (value plus the schema violation), or
           *  `malformed` (not a JSON object; empty parses as `{}`). */
          input: ToolInput;
          attempt: number;
          deadline?: DateTime;
      }
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

/** Kinds of call a worker can execute. */
export type WorkKind = "tool_call" | "llm_call";

export type WorkerAction =
    | {
          /** A flat, all-optional LLM request. Omit `id` and the engine mints one;
           *  it becomes the assistant node's id. Omitted fields are filled from the
           *  agent config, then engine defaults; omit `messages` and the engine
           *  prompts with `[config.system?] + the decision's declared view`.
           *  Explicit `messages` suppress system injection; a bare
           *  `{ type: "llm.call" }` prompts per the agent's identity over the view. */
          type: "llm.call";
          id?: string;
          model?: string;
          messages?: MessageInput[];
          tools?: LlmTool[];
          temperature?: number;
          max_completion_tokens?: number;
          reasoning?: ReasoningConfig;
          stream?: boolean;
          retry?: RetryPolicy;
          /** Omitted ⇒ `"server"`. */
          handler?: LlmHandler;
      }
    | {
          /** `id` is the model's tool call id; omit it for an ad-hoc worker tool and the engine mints one. */
          type: "tool.call";
          id?: string;
          name: string;
          /** A non-string value is canonicalized to its JSON text engine-side. */
          arguments: string | Record<string, unknown>;
          /** Omitted ⇒ `"worker"`. */
          handler?: ToolHandler;
          retry?: RetryPolicy;
      }
    // On the sync/pull paths a settle's `id` may be omitted — the answered
    // `*.execute` trigger names the effect the engine resolves it against.
    | {
          type: "tool.result";
          id?: string;
          /** A non-string value is canonicalized to its JSON text engine-side. */
          result: unknown;
          attempt?: number;
      }
    | { type: "llm.result"; id?: string; response: LlmResponse; attempt?: number }
    | {
          type: "tool.error";
          id?: string;
          error: string;
          /** Omitted ⇒ terminal. */
          retryable?: boolean;
          attempt?: number;
          code?: string;
          detail?: unknown;
      }
    | {
          type: "llm.error";
          id?: string;
          error: string;
          /** Omitted ⇒ terminal. */
          retryable?: boolean;
          attempt?: number;
          code?: string;
          detail?: unknown;
      }
    | { type: "sub_agent.spawn"; session_id: Uuid; agent_id: string; tool_call_id: string; retry?: RetryPolicy }
    | { type: "message.send"; session_id: Uuid; message: MessageInput }
    | { type: "interrupt"; interrupt_id?: string; reason: string; payload?: unknown }
    | { type: "done"; data: unknown };

/** Body settling a call: a result or an error. */
export type SettleEffectRequest =
    | {
          type: "tool.result";
          id: string;
          /** A non-string value is canonicalized to its JSON text engine-side. */
          result: unknown;
          attempt?: number;
      }
    | { type: "llm.result"; id: string; response: LlmResponse; attempt?: number }
    | {
          type: "tool.error";
          id: string;
          error: string;
          /** Omitted ⇒ terminal. */
          retryable?: boolean;
          attempt?: number;
          code?: string;
          detail?: unknown;
      }
    | {
          type: "llm.error";
          id: string;
          error: string;
          /** Omitted ⇒ terminal. */
          retryable?: boolean;
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

/** Which effect to settle, on which session; optional `attempt` fences a stale executor. */
export interface SettleEffectTarget {
    sessionId: string;
    id: string;
    attempt?: number;
}

/** Settle a `tool_call` with its result; any non-string value is canonicalized to its
 *  JSON text engine-side. */
export interface SettleToolResult {
    kind?: "tool_call";
    result: unknown;
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
    // Discriminate on `response`/`error` first: a tool `result` is now `unknown`, which
    // includes `undefined`, so it can't be the presence check — a failure is.
    if (args.response !== undefined) {
        return {
            type: "llm.result",
            id: args.id,
            response: args.response,
            attempt: args.attempt,
        };
    }
    if (args.error !== undefined) {
        return {
            type: (args.kind ?? "tool_call") === "llm_call" ? "llm.error" : "tool.error",
            id: args.id,
            error: args.error,
            retryable: args.retryable ?? false,
            attempt: args.attempt,
        };
    }
    return {
        type: "tool.result",
        id: args.id,
        result: args.result,
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
    /** Opaque worker state as raw JSON. */
    state: unknown;
    /** Node id this version was anchored to. */
    anchor?: string;
}

/** An outstanding call abandoned (branch forked away, or session interrupted/cancelled). */
export interface CallVoided {
    type: "call.voided";
    kind: "tool_call" | "llm_call" | "sub_agent";
    /** Call id; for a sub-agent, the tool call it answers. */
    id: string;
    /** Child session, present on `sub_agent` voids. */
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

/** A worker-state write anchored to its tree position. */
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

// In-flight calls surfaced on each decision; `kind` is open, ignore unknown kinds.

export interface InFlightCallBase {
    id: string;
    status: EffectStatus;
    attempt: number;
    deadline?: DateTime;
    /** Tree node the call was requested at. */
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

/** A call kind this SDK doesn't know; ignore it. `never` fields keep `kind` narrowing usable. */
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

/** The engine-derived default continuation for a trigger: the decision an SDK
 *  agent loop would author. Advisory — the engine never applies it; a worker
 *  accepts by returning it (amended or verbatim) as its decision. */
export interface ProposedDecision {
    messages: MessageInput[];
    actions: WorkerAction[];
}

/** Exactly what a worker receives when the engine asks for a decision. Every
 *  field is always present on the wire — the contract is total, not shaped by
 *  what happens to be empty; `deadline`/`turn_id` are `null` when unset. */
export interface WireDecisionRequest {
    session_id: Uuid;
    decision_id: string;
    agent_id: string;
    identity: WorkerIdentity;
    trigger: DecisionTrigger;
    /** The default continuation for `trigger`; `null` when only the worker can
     *  answer (`client.messages`, a well-formed `tool.execute`, `llm.execute`)
     *  or there is nothing to do (`client.action`, `interrupt.resumed`). */
    proposed: ProposedDecision | null;
    /** Your state as raw JSON; `null` when the session has none. */
    state: unknown;
    calls: InFlightCall[];
    /** Outstanding parallel work for this step: in-flight `tool_call`/`sub_agent`
     *  calls plus sibling results not yet recorded. The default step gate — the
     *  built-in loop prompts at 0. Inspect `calls` for finer control. */
    pending_calls: number;
    /** The active conversation as a flat list. */
    messages: Message[];
    /** The full tree, for clients that need branch structure. */
    message_tree: MessageTree;
    ancestry: Uuid[];
    attempts: number;
    deadline: DateTime | null;
    turn_id: string | null;
}

/** The decision a worker authors in reply to a decision request: the updated
 *  conversation, the next actions, and new state. The whole sync HTTP response
 *  body, and the streamed `decision.result` frame. */
export interface WireDecisionResponse {
    actions: WorkerAction[];
    /** Flat conversation the engine reconciles into the message tree: known ids
     *  continue, id-less/unknown messages are appended (forking automatically). */
    messages: MessageInput[];
    /** Opaque worker state as raw JSON; omit or `null` keeps current, `{}` clears. */
    state?: unknown;
}

/** A decision pushed to the engine out-of-band via the submit route: the
 *  `session_id`/`decision_id` that route it, wrapping the worker-authored
 *  decision. */
export interface SubmitDecisionRequest extends WireDecisionResponse {
    session_id: Uuid;
    decision_id: string;
}

export interface SubmitResponse {
    ok: boolean;
    error?: string;
}
