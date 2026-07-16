/**
 * One entry point per wire surface; every protocol type is named under $defs.
 */
export interface Protocol {
    client_input?: ClientInput;
    client_payload?: ClientPayload;
    decision_request?: DecisionRequest;
    decision_response?: DecisionResponse;
    stream_delta?: StreamDelta;
    token_delta?: TokenDelta;
    [property: string]: any;
}

/**
 * Everything a client can send on the input surface: submit a message / a full view / a
 * named action, resume an interrupt, or settle a client tool. A flat, internally-tagged
 * union — its six tags produce serde's "unknown variant, expected one of …" error for
 * free. `Runtime::handle_client_input` is the single seam that dispatches it (mirroring
 * `resolve_response` on the worker side).
 *
 * Addressing lives where it is meaningful, not in a shared envelope: `agent_id` (routes
 * the turn, creating the session if new) and the optional idempotency `turn_id` are
 * fields of the three submit variants only. A resume/settle addresses an interrupt/effect
 * id and continues whatever turn is active, so it carries neither — misplacing them is
 * unrepresentable rather than rejected. `session_id` is the one universal address and
 * rides the envelope. A submit's body rebuilds a [`ClientPayload`] at the seam.
 *
 * The body of an interrupt resume: which interrupt, and the payload delivered
 * to the worker. Shared by the [`ClientInput::InterruptResume`] input and the
 * [`DecisionTrigger::InterruptResumed`] trigger.
 */
export interface ClientInput {
    agent_id?: string;
    message?: DraftMessage;
    stream?: boolean;
    turn_id?: null | string;
    type: ClientInputType;
    client?: ClientContext;
    messages?: DraftMessage[];
    args?: any;
    name?: string;
    interrupt_id?: string;
    payload?: any;
    attempt?: number | null;
    id?: string;
    result?: any;
    error?: string;
    retryable?: boolean;
}

/**
 * Inputs a client declares on its run (the AG-UI `tools`/`context`/`state`/
 * `forwardedProps`), forwarded to the worker on the `client.messages` decision.
 * `tools` are the browser's frontend tools, normalized to client-handled
 * [`AgentTool`]s; the engine layers them onto the proposed config by default, and
 * the worker may override (e.g. whitelist) by returning its own `agent`.
 *
 * Inputs the client declared on its run; the engine layers `client.tools`
 * onto the proposed config by default.
 */
export interface ClientContext {
    context?: any[];
    forwarded_props?: any;
    state?: any;
    tools?: AgentTool[];
}

/**
 * A function tool the agent offers. The model-facing contract is
 * `name`/`description`/`input`/`output`; `handler` selects where a call runs —
 * `Some(Client)` ⇒ client-executed, absent ⇒ worker-executed (the default).
 * `server` is invalid for tools.
 */
export interface AgentTool {
    description?: string;
    handler?: Handler | null;
    input?: any;
    name: string;
    output?: any;
}

/**
 * Server-side executor resolves the provider and makes the call (LLM only).
 *
 * Dispatched to the work queue for the worker to execute.
 *
 * Executed by the client. Session goes Idle while waiting (tools only).
 *
 * Where a call runs — one wire enum so `handler` has a single type on every
 * surface. Tool calls accept `worker` (default) or `client`; LLM calls accept
 * `server` (default) or `worker`. The invalid pairing (a `server` tool, a
 * `client` LLM call) is rejected at the decision seam.
 *
 * `server` or `worker`; omitted ⇒ `server`.
 *
 * `worker` or `client`; omitted ⇒ `worker`.
 */
export type Handler = "server" | "worker" | "client";

/**
 * The wire form of a [`Message`]: `id` is optional because a client-submitted or
 * worker-authored message is not yet recorded. `record`/`rerecord`
 * (`runtime::session::wire`) are the seams that lower it to the internal
 * [`Message`] (id always present) at recording time.
 */
export interface DraftMessage {
    content?: ContentPart[] | null | string;
    id?: null | string;
    name?: null | string;
    role: Role;
    tool_call_id?: null | string;
    tool_calls?: ToolCall[] | null;
}

export interface ContentPart {
    text?: string;
    type: ContentPartType;
    image_url?: ImageURL;
    file?: FileData;
    input_audio?: AudioData;
    video_url?: VideoURL;
}

export interface FileData {
    file_data: string;
    filename: string;
}

export interface ImageURL {
    url: string;
}

export interface AudioData {
    data: string;
    format: string;
}

export type ContentPartType = "text" | "image_url" | "file" | "input_audio" | "video_url";

export interface VideoURL {
    url: string;
}

export type Role = "system" | "user" | "assistant" | "tool";

export interface ToolCall {
    function: ToolCallFunction;
    id: string;
    type: string;
}

export interface ToolCallFunction {
    arguments: string;
    name: string;
}

export type ClientInputType =
    | "client.message"
    | "client.messages"
    | "client.action"
    | "interrupt.resume"
    | "tool.result"
    | "tool.error";

/**
 * The client→engine inbound *submit* wire form: an untrusted client submits a message,
 * its full conversation view, or a named action. Lowered to domain events at the
 * `SubmitClientPayload` command seam (`runtime::session::command`); never persisted
 * as-is. Carried verbatim inside [`ClientInput`], which is the full client input
 * surface.
 *
 * The body of a `client.message`: one message, optionally streamed.
 *
 * The body of a `client.messages`: the client's full conversation view, optionally
 * streamed.
 *
 * The payload of a `client.action`: a named action with optional JSON args.
 */
export interface ClientPayload {
    message?: DraftMessage;
    stream?: boolean;
    type: ClientPayloadType;
    client?: ClientContext;
    messages?: DraftMessage[];
    args?: any;
    name?: string;
}

export type ClientPayloadType = "client.message" | "client.messages" | "client.action";

export interface DecisionRequest {
    /**
     * The agent config resolved for the active path (`null` when none is set).
     */
    agent?: AgentConfig | null;
    agent_id: string;
    ancestry: string[];
    attempts: number;
    calls: Effect[];
    deadline?: Date | null;
    decision_id: string;
    identity: WorkerIdentity;
    message_tree: MessageTree;
    messages: Message[];
    /**
     * Count of in-flight `tool_call`/`sub_agent` calls.
     */
    pending_calls: number;
    /**
     * The engine's default continuation for `trigger` (`null` when it needs
     * worker knowledge). Advisory: accept by echoing it as the decision.
     */
    proposed?: DecisionResponse | null;
    session_id: string;
    state: any;
    trigger: DecisionTrigger;
    turn_id?: null | string;
}

/**
 * A declared agent identity. `model` is the only required field; everything else
 * refines the proposed LLM request the engine derives for `client.messages`.
 */
export interface AgentConfig {
    /**
     * Provider wire format for worker-handled calls; requires `handler:
     * worker`. Absent ⇒ the neutral format.
     */
    format?: LlmFormat | null;
    /**
     * Where the proposed LLM call runs: `Some(Worker)` ⇒ the worker executes it
     * (answering `llm.execute`); absent or `Some(Server)` ⇒ the engine's
     * server-side provider. `client` is invalid and rejected at the decision seam.
     */
    handler?: Handler | null;
    model: string;
    retry?: RetryPolicy | null;
    stream?: boolean;
    /**
     * Sub-agents the model can delegate to. Presented to the model as tools (by
     * id) alongside `tools`, but each call spawns a child session rather than
     * executing a function.
     */
    sub_agents?: SubAgent[];
    system?: null | string;
    /**
     * Worker- or client-executed tools the model can call.
     */
    tools?: AgentTool[];
}

/**
 * OpenAI Chat Completions.
 *
 * Anthropic Messages API.
 */
export type LlmFormat = "openai" | "anthropic";

/**
 * Fully-resolved retry policy — no optional fields. Stored on call state and
 * read directly by retry logic.
 */
export interface RetryPolicy {
    backoff_base_secs: number;
    backoff_max_secs: number;
    max_retries: number;
    timeout_secs?: number | null;
}

/**
 * A sub-agent the model can delegate to. Named by `id` (the child agent to spawn,
 * and the tool name the model calls); its model-facing input is the conventional
 * single-`message` delegation schema.
 */
export interface SubAgent {
    description?: string;
    id: string;
}

/**
 * An in-flight effect (Pending or RetryScheduled) surfaced on each worker decision.
 * A flat envelope plus kind-specific fields: a tool call's
 * `name`/`arguments`/`handler`, an LLM call's `handler`/`stream`, a
 * sub-agent's `agent_id`/`session_id`.
 */
export interface Effect {
    agent_id?: null | string;
    /**
     * The tree node the effect was requested at.
     */
    anchor?: null | string;
    arguments?: null | string;
    attempt: number;
    deadline?: Date | null;
    handler?: Handler | null;
    id: string;
    kind: EffectKind;
    name?: null | string;
    session_id?: null | string;
    status: EffectStatus;
    stream?: boolean | null;
}

export type EffectKind = "tool_call" | "sub_agent" | "llm_call";

export type EffectStatus = "pending" | "completed" | "failed" | "retry_scheduled" | "queued";

/**
 * The owner as delivered to the worker on `DecisionRequest.identity`: the
 * subject and its metadata, without the tenant. The tenant scopes the session
 * internally but is not sent to the worker.
 */
export interface WorkerIdentity {
    id?: null | string;
    metadata?: { [key: string]: string };
}

export interface MessageTree {
    head_id?: null | string;
    nodes?: Node[];
}

export interface Node {
    message: Message;
    parent_id?: null | string;
}

export interface Message {
    content?: ContentPart[] | null | string;
    id: string;
    name?: null | string;
    role: Role;
    tool_call_id?: null | string;
    tool_calls?: ToolCall[] | null;
}

/**
 * A decision: the messages/actions to author, plus optional state/agent writes.
 * The worker returns one; the engine also proposes one as the default
 * continuation (`DecisionRequest::proposed`), which the worker echoes or amends.
 */
export interface DecisionResponse {
    actions?: DecisionAction[];
    /**
     * A new agent config write; omitted keeps the current config.
     */
    agent?: AgentConfig | null;
    messages?: DraftMessage[];
    /**
     * Omitted or `null` keeps the current state; clear with a non-null empty value.
     */
    state?: any;
}

/**
 * The action a worker authors on the wire. Mirrors the internal `Action`, but a
 * settle's effect id may be omitted: on the sync/pull paths the answered
 * `*.execute` trigger names it, so echoing it is redundant. `resolve_response`
 * (`runtime::session::wire`) turns this into the internal `Action` (id always
 * present) at the transport boundary.
 *
 * A flat, all-optional LLM request. `id` omitted ⇒ the engine mints one; it
 * becomes the assistant node's id. Omitted fields are filled from the agent
 * config (merge source), then engine defaults; `messages` omitted ⇒
 * `[config.system?] + the decision's declared view`. Explicit `messages`
 * suppress system injection. A bare `{"type":"llm.call"}` prompts per the
 * agent's identity over the current view.
 *
 * `id` omitted ⇒ the engine mints one (LLM-driven tools carry the model's id).
 *
 * `id`/`attempt` omitted ⇒ taken from the answering `tool.execute` trigger,
 * fencing the result to the attempt that ran.
 *
 * `id`/`attempt` omitted ⇒ taken from the answering `llm.execute` trigger,
 * fencing the result to the attempt that ran.
 *
 * `interrupt_id` omitted ⇒ the engine mints one to correlate the later resume.
 */
export interface DecisionAction {
    /**
     * `server` or `worker`; omitted ⇒ `server`.
     *
     * `worker` or `client`; omitted ⇒ `worker`.
     */
    handler?: Handler;
    id?: null | string;
    max_completion_tokens?: number | null;
    messages?: DraftMessage[] | null;
    model?: null | string;
    reasoning?: ReasoningConfig | null;
    retry?: RetryPolicy | null;
    stream?: boolean | null;
    temperature?: number | null;
    tools?: LlmTool[] | null;
    type: DecisionActionType;
    arguments?: any;
    name?: string;
    attempt?: number | null;
    result?: any;
    /**
     * A neutral `LlmResponse`, or the provider's native response when the
     * answered `llm.execute` carried a `format`.
     */
    response?: any;
    code?: ErrorCode | null;
    detail?: any;
    error?: string;
    /**
     * Omitted ⇒ terminal.
     */
    retryable?: boolean;
    agent_id?: string;
    session_id?: string;
    /**
     * The model tool-call this delegation answers — always required.
     */
    tool_call_id?: string;
    message?: DraftMessage;
    interrupt_id?: null | string;
    payload?: any;
    reason?: string;
    data?: any;
}

export type ErrorCode = "provider_error" | "rate_limited" | "refused" | "budget_exceeded" | "deadline_exceeded";

export interface ReasoningConfig {
    effort?: ReasoningEffort | null;
    enabled?: boolean | null;
    exclude?: boolean | null;
    max_tokens?: number | null;
}

export type ReasoningEffort = "xhigh" | "high" | "medium" | "low" | "minimal" | "none";

/**
 * A tool's declared contract: flat on the wire. Providers that need
 * OpenAI-style `{"type": "function", "function": {…}}` nesting re-wrap at
 * their own boundary.
 */
export interface LlmTool {
    description: string;
    /**
     * JSON Schema for the tool's arguments; omitted declares a no-argument
     * tool. The engine validates each call's arguments against it and hands
     * providers their native form.
     */
    input?: any;
    name: string;
    /**
     * JSON Schema the settled result must satisfy; never sent to the model.
     * A violating result settles as a terminal tool error.
     */
    output?: any;
}

export type DecisionActionType =
    | "llm.call"
    | "tool.call"
    | "tool.result"
    | "llm.result"
    | "tool.error"
    | "llm.error"
    | "sub_agent.spawn"
    | "message.send"
    | "interrupt"
    | "done";

/**
 * The trigger a worker sees on the wire — the materialized projection of the
 * engine's internal decision trigger. It has no `ClientMessage`: a bare client
 * message is always materialized to `ClientTranscript` by `to_wire_trigger`
 * (`runtime::session::wire`) before delivery, so an unmaterialized message can
 * never reach a worker.
 *
 * The first decision of every session; carries no proposal.
 *
 * Answer with `tool.result`/`tool.error`.
 *
 * Answer with `llm.result`/`llm.error`.
 *
 * The body of an interrupt resume: which interrupt, and the payload delivered
 * to the worker. Shared by the [`ClientInput::InterruptResume`] input and the
 * [`DecisionTrigger::InterruptResumed`] trigger.
 */
export interface DecisionTrigger {
    type: DecisionTriggerType;
    /**
     * Inputs the client declared on its run; the engine layers `client.tools`
     * onto the proposed config by default.
     */
    client?: ClientContext;
    messages?: DraftMessage[];
    new_from?: number;
    args?: any;
    name?: string;
    arguments?: string;
    attempt?: number;
    deadline?: Date | null;
    id?: string;
    /**
     * The engine's classification of `arguments` against the tool's
     * declared `input` schema: `valid` (with the parsed `value`),
     * `invalid` (value plus the violation), or `malformed` (not a JSON
     * object). Always on the wire.
     */
    input?: ToolInput;
    error?: null | string;
    ok?: boolean;
    result?: null | string;
    format?: LlmFormat | null;
    /**
     * The neutral `LlmRequest` JSON, or the provider's native request body
     * when `format` is set.
     */
    request?: any;
    stream?: boolean;
    code?: ErrorCode | null;
    cost?: null | string;
    detail?: any;
    message?: DraftMessage | null;
    truncated?: boolean;
    usage?: any;
    agent_id?: string;
    session_id?: string;
    interrupt_id?: string;
    payload?: any;
}

/**
 * The engine's classification of `arguments` against the tool's
 * declared `input` schema: `valid` (with the parsed `value`),
 * `invalid` (value plus the violation), or `malformed` (not a JSON
 * object). Always on the wire.
 *
 * The engine's classification of a tool call's arguments, delivered on the
 * `tool.execute` trigger alongside the raw `arguments` string. Always on the
 * wire — absence never carries meaning.
 *
 * Parsed and, when the tool declares an `input` schema, conforming to it.
 * `value` is exactly the parsed `arguments` — the engine never mutates it.
 *
 * Parsed to an object that violates the declared `input` schema.
 *
 * Not a JSON object: malformed JSON or a non-object value.
 */
export interface ToolInput {
    status: Status;
    value?: any;
    error?: string;
}

export type Status = "valid" | "invalid" | "malformed";

export type DecisionTriggerType =
    | "session.start"
    | "client.messages"
    | "client.action"
    | "tool.execute"
    | "tool.finished"
    | "llm.execute"
    | "llm.finished"
    | "sub_agent.finished"
    | "interrupt.resumed";

export interface StreamDelta {
    finish_reason?: null | string;
    reasoning?: null | string;
    text?: null | string;
    tool_calls?: ToolCallChunk[];
}

export interface ToolCallChunk {
    arguments?: null | string;
    id: string;
    name?: null | string;
}

export interface TokenDelta {
    agent_id: string;
    attempt: number;
    call_id: string;
    finish_reason?: null | string;
    reasoning?: null | string;
    /**
     * Transport routing key.
     */
    root_session_id: string;
    /**
     * Per-call counter, distinct from event-store sequence.
     */
    seq: number;
    /**
     * May be a sub-agent of root.
     */
    session_id: string;
    /**
     * Tenant isolation key — subscribers must match.
     */
    tenant_id: string;
    text?: null | string;
    tool_calls?: ToolCallChunk[];
    turn_id?: null | string;
}
