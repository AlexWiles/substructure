/**
 * One entry point per wire surface; every protocol type is named under $defs.
 */
export interface Protocol {
    client_input?: ClientInput;
    client_payload?: ClientPayload;
    decision_request?: DecisionRequest;
    decision_response?: DecisionResponseClass;
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
 */
export interface ClientInput {
    agent_id?: string;
    message?: ClientInputMessage;
    stream?: boolean;
    turn_id?: null | string;
    type: ClientInputType;
    client?: Client;
    messages?: ClientInputMessage[];
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
export interface Client {
    context?: any[];
    forwarded_props?: any;
    state?: any;
    tools?: ClientTool[];
}

/**
 * A function tool the agent offers. The model-facing contract is
 * `name`/`description`/`input`/`output`; `handler` selects where a call runs —
 * `Some(Client)` ⇒ client-executed, absent ⇒ worker-executed (the default).
 * `server` is invalid for tools.
 */
export interface ClientTool {
    description?: string;
    handler?: HandlerEnum | null;
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
export type HandlerEnum = "server" | "worker" | "client";

/**
 * The wire form of a [`Message`]: `id` is optional because a client-submitted or
 * worker-authored message is not yet recorded. `record`/`rerecord`
 * (`runtime::session::wire`) are the seams that lower it to the internal
 * [`Message`] (id always present) at recording time.
 */
export interface ClientInputMessage {
    content?: ContentElement[] | null | string;
    id?: null | string;
    name?: null | string;
    role: Role;
    tool_call_id?: null | string;
    tool_calls?: MessageToolCall[] | null;
}

export interface ContentElement {
    text?: string;
    type: ContentType;
    image_url?: ImageURL;
    file?: File;
    input_audio?: InputAudio;
    video_url?: VideoURL;
}

export interface File {
    file_data: string;
    filename: string;
}

export interface ImageURL {
    url: string;
}

export interface InputAudio {
    data: string;
    format: string;
}

export type ContentType = "text" | "image_url" | "file" | "input_audio" | "video_url";

export interface VideoURL {
    url: string;
}

export type Role = "system" | "user" | "assistant" | "tool";

export interface MessageToolCall {
    function: Function;
    id: string;
    type: string;
}

export interface Function {
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
    message?: ClientInputMessage;
    stream?: boolean;
    type: ClientPayloadType;
    client?: Client;
    messages?: ClientInputMessage[];
    args?: any;
    name?: string;
}

export type ClientPayloadType = "client.message" | "client.messages" | "client.action";

export interface DecisionRequest {
    /**
     * The agent config resolved for the active path (`null` when none is set).
     */
    agent?: AgentClass | null;
    agent_id: string;
    ancestry: string[];
    attempts: number;
    calls: CallElement[];
    deadline?: Date | null;
    decision_id: string;
    identity: Identity;
    message_tree: MessageTree;
    messages: NodeMessage[];
    /**
     * Count of in-flight `tool_call`/`sub_agent` calls.
     */
    pending_calls: number;
    /**
     * The engine's default continuation for `trigger` (`null` when it needs
     * worker knowledge). Advisory: accept by echoing it as the decision.
     */
    proposed?: DecisionResponseClass | null;
    session_id: string;
    state: any;
    trigger: Trigger;
    turn_id?: null | string;
}

/**
 * A declared agent identity. `model` is the only required field; everything else
 * refines the proposed LLM request the engine derives for `client.messages`.
 */
export interface AgentClass {
    /**
     * Provider wire format for worker-handled calls; requires `handler:
     * worker`. Absent ⇒ the neutral format.
     */
    format?: FormatEnum | null;
    /**
     * Where the proposed LLM call runs: `Some(Worker)` ⇒ the worker executes it
     * (answering `llm.execute`); absent or `Some(Server)` ⇒ the engine's
     * server-side provider. `client` is invalid and rejected at the decision seam.
     */
    handler?: HandlerEnum | null;
    model: string;
    retry?: RetryClass | null;
    stream?: boolean;
    /**
     * Sub-agents the model can delegate to. Presented to the model as tools (by
     * id) alongside `tools`, but each call spawns a child session rather than
     * executing a function.
     */
    sub_agents?: SubAgentElement[];
    system?: null | string;
    /**
     * Worker- or client-executed tools the model can call.
     */
    tools?: ClientTool[];
}

/**
 * OpenAI Chat Completions.
 *
 * Anthropic Messages API.
 */
export type FormatEnum = "openai" | "anthropic";

/**
 * Fully-resolved retry policy — no optional fields. Stored on call state and
 * read directly by retry logic.
 */
export interface RetryClass {
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
export interface SubAgentElement {
    description?: string;
    id: string;
}

/**
 * An in-flight effect (Pending or RetryScheduled) surfaced on each worker decision.
 * A flat envelope plus kind-specific fields: a tool call's
 * `name`/`arguments`/`handler`, an LLM call's `handler`/`stream`, a
 * sub-agent's `agent_id`/`session_id`.
 */
export interface CallElement {
    agent_id?: null | string;
    /**
     * The tree node the effect was requested at.
     */
    anchor?: null | string;
    arguments?: null | string;
    attempt: number;
    deadline?: Date | null;
    handler?: HandlerEnum | null;
    id: string;
    kind: CallKind;
    name?: null | string;
    session_id?: null | string;
    status: CallStatus;
    stream?: boolean | null;
}

export type CallKind = "tool_call" | "sub_agent" | "llm_call";

export type CallStatus = "pending" | "completed" | "failed" | "retry_scheduled" | "queued";

export interface Identity {
    id?: null | string;
    metadata?: { [key: string]: string };
    tenant_id: string;
}

export interface MessageTree {
    head_id?: null | string;
    nodes?: NodeElement[];
}

export interface NodeElement {
    kind: NodeKind;
    message?: NodeMessage;
    parent_id?: null | string;
    control?: Control;
}

/**
 * A non-conversational tree marker (interrupt/resume); filtered out of LLM prompts.
 */
export interface Control {
    id: string;
    interrupt_id: string;
    kind: ControlKind;
    origin: Origin;
    payload?: any;
    reason?: string;
}

export type ControlKind = "interrupt" | "resume";

/**
 * Privilege level of the caller that issued an interrupt. Derived from the
 * authenticated `Caller`, never from request data; resuming requires a
 * caller at or above the origin's privilege.
 */
export type Origin = "system" | "machine" | "frontend";

export type NodeKind = "message" | "control";

export interface NodeMessage {
    content?: ContentElement[] | null | string;
    id: string;
    name?: null | string;
    role: Role;
    tool_call_id?: null | string;
    tool_calls?: MessageToolCall[] | null;
}

/**
 * A decision: the messages/actions to author, plus optional state/agent writes.
 * The worker returns one; the engine also proposes one as the default
 * continuation (`DecisionRequest::proposed`), which the worker echoes or amends.
 */
export interface DecisionResponseClass {
    actions?: ActionElement[];
    /**
     * A new agent config write; omitted keeps the current config.
     */
    agent?: AgentClass | null;
    messages?: ClientInputMessage[];
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
export interface ActionElement {
    /**
     * `server` or `worker`; omitted ⇒ `server`.
     *
     * `worker` or `client`; omitted ⇒ `worker`.
     */
    handler?: HandlerEnum;
    id?: null | string;
    max_completion_tokens?: number | null;
    messages?: ClientInputMessage[] | null;
    model?: null | string;
    reasoning?: ReasoningClass | null;
    retry?: RetryClass | null;
    stream?: boolean | null;
    temperature?: number | null;
    tools?: ActionTool[] | null;
    type: ActionType;
    arguments?: any;
    name?: string;
    attempt?: number | null;
    result?: any;
    /**
     * A neutral `LlmResponse`, or the provider's native response when the
     * answered `llm.execute` carried a `format`.
     */
    response?: any;
    code?: CodeEnum | null;
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
    message?: ClientInputMessage;
    interrupt_id?: null | string;
    payload?: any;
    reason?: string;
    data?: any;
}

export type CodeEnum = "provider_error" | "rate_limited" | "refused" | "budget_exceeded" | "deadline_exceeded";

export interface ReasoningClass {
    effort?: EffortEnum | null;
    enabled?: boolean | null;
    exclude?: boolean | null;
    max_tokens?: number | null;
}

export type EffortEnum = "xhigh" | "high" | "medium" | "low" | "minimal" | "none";

/**
 * A tool's declared contract: flat on the wire. Providers that need
 * OpenAI-style `{"type": "function", "function": {…}}` nesting re-wrap at
 * their own boundary.
 */
export interface ActionTool {
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

export type ActionType =
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
 */
export interface Trigger {
    type: TriggerType;
    /**
     * Inputs the client declared on its run; the engine layers `client.tools`
     * onto the proposed config by default.
     */
    client?: Client;
    messages?: ClientInputMessage[];
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
    input?: Input;
    error?: null | string;
    ok?: boolean;
    result?: null | string;
    format?: FormatEnum | null;
    /**
     * The neutral `LlmRequest` JSON, or the provider's native request body
     * when `format` is set.
     */
    request?: any;
    stream?: boolean;
    code?: CodeEnum | null;
    cost?: null | string;
    detail?: any;
    message?: ClientInputMessage | null;
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
export interface Input {
    status: InputStatus;
    value?: any;
    error?: string;
}

export type InputStatus = "valid" | "invalid" | "malformed";

export type TriggerType =
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
    tool_calls?: StreamDeltaToolCall[];
}

export interface StreamDeltaToolCall {
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
    tool_calls?: StreamDeltaToolCall[];
    turn_id?: null | string;
}
