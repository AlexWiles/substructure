/**
 * One entry point per wire surface; every protocol type is named under $defs.
 */
export interface Protocol {
    client_input?: ClientInput;
    client_payload?: ClientPayload;
    decision_request?: DecisionRequest;
    decision_response?: DecisionResponse;
    interrupt_payload?: InterruptPayload;
    interrupt_resolution?: InterruptResolution;
    stream_delta?: StreamDelta;
    token_delta?: TokenDelta;
    [property: string]: unknown;
}

/**
 * Everything a client can send on the input surface: submit a message / a full view / an
 * append batch / a named action, resume an interrupt, or settle a client tool. A flat,
 * internally-tagged union — its seven tags produce serde's "unknown variant, expected one
 * of …" error for free. `Runtime::handle_client_input` is the single seam that dispatches
 * it (mirroring `resolve_response` on the worker side).
 *
 * Addressing lives where it is meaningful, not in a shared envelope: `agent_id` (routes
 * the turn, creating the session if new) and the optional idempotency `turn_id` are
 * fields of the four submit variants only. A resume/settle addresses an interrupt/effect
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
    /**
     * Hold this message for the next turn instead of refusing it when one
     * is already running. Off by default: rejection stays the contract for
     * a plain submitter, and queuing is declared intent.
     *
     * Hold this batch for the next turn instead of refusing it when one is
     * already running. Off by default: rejection stays the contract for a
     * plain submitter, and queuing is declared intent.
     */
    queue?: boolean;
    stream?: boolean;
    turn_id?: null | string;
    type: ClientInputType;
    client?: ClientContext;
    messages?: DraftMessage[];
    args?: unknown;
    name?: string;
    interrupt_id?: string;
    payload?: unknown;
    attempt?: number | null;
    id?: string;
    result?: unknown;
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
    context?: unknown[];
    forwarded_props?: unknown;
    state?: unknown;
    tools?: AgentTool[];
}

/**
 * A function tool the agent offers. The model-facing contract is
 * `name`/`description`/`input`/`output`; `handler` selects where a call runs —
 * `Some(Client)` ⇒ client-executed, absent ⇒ worker-executed (the default).
 * `server` is invalid for tools: engine-executed tools come from a connector,
 * which a worker declares by id rather than by tool.
 */
export interface AgentTool {
    description?: string;
    handler?: Handler | null;
    input?: unknown;
    name: string;
    output?: unknown;
}

/**
 * Server-side executor resolves the provider or connection and makes the call.
 *
 * Dispatched to the work queue for the worker to execute.
 *
 * Executed by the client. Session goes Idle while waiting (tools only).
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
    | "client.append"
    | "client.action"
    | "interrupt.resume"
    | "tool.result"
    | "tool.error";

/**
 * The client→engine inbound *submit* wire form: an untrusted client submits a message,
 * its full conversation view, an append batch, or a named action. Lowered to domain events
 * at the
 * `SubmitClientPayload` command seam (`runtime::session::command`); never persisted
 * as-is. Carried verbatim inside [`ClientInput`], which is the full client input
 * surface.
 *
 * The body of a `client.message`: one message, optionally streamed.
 *
 * The body of a `client.messages`: the client's full conversation view, optionally
 * streamed.
 *
 * The body of a `client.append`: messages appended at the session head. The
 * view is composed against the active path at delivery, so a queued append
 * lands after whatever turn beat it — it can never fork the tree. Messages
 * whose ids are already recorded are dropped.
 *
 * The payload of a `client.action`: a named action with optional JSON args.
 */
export interface ClientPayload {
    message?: DraftMessage;
    stream?: boolean;
    type: ClientPayloadType;
    client?: ClientContext;
    messages?: DraftMessage[];
    args?: unknown;
    name?: string;
}

export type ClientPayloadType = "client.message" | "client.messages" | "client.append" | "client.action";

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
     * The engine's default continuation for `trigger` (empty when it needs
     * worker knowledge). Advisory: accept by echoing it as the decision.
     */
    proposed: DecisionResponse;
    session_id: string;
    state: unknown;
    trigger: DecisionTrigger;
    turn_id?: null | string;
}

/**
 * A declared agent identity — the same shape whether it is written in an
 * `[agent.<id>]` section or returned by a worker.
 *
 * `llm` names the `[llm.*]` block every proposed call runs on, and so decides
 * both the venue (the engine with a vendor key, or the agent's own worker) and
 * the wire shape of a worker-run call. It is effectively required: a config
 * that names none fails when the engine resolves a call against it.
 */
export interface AgentConfig {
    /**
     * The `[llm.*]` block this agent's calls run on.
     */
    llm?: null | string;
    /**
     * MCP servers the agent draws tools from. The engine resolves each against
     * its connection registry into [`ConnectorTool`]s the model sees alongside
     * `tools`. Like `sub_agents`, these are never merged into `tools` — the
     * worker declares the server, not its tools.
     *
     * A second protocol gets its own field rather than a `type` tag here: its
     * filter would not be this one (MCP annotations mean nothing to an A2A
     * agent), and a union of conditionally-valid fields generates badly in the
     * Go and Python bindings.
     */
    mcp?: MCPServer[];
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
 * An MCP server the agent draws tools from. `id` resolves against the engine's
 * connection registry — locally from `[mcp]` in `substructure.toml`, in the
 * cloud from the connections an admin granted this app. The worker never names
 * a URL or a credential.
 */
export interface MCPServer {
    id: string;
    /**
     * Narrows what the model sees. Absent ⇒ every tool the connection grants.
     */
    tools?: MCPTools | null;
}

/**
 * An MCP server's tool filter. Applied in order — capability predicates, then
 * `include`, then `exclude` — and only ever narrowing, so a filter can never
 * widen what the connection grants.
 *
 * `include`/`exclude` are globs matched against the tool's name on the
 * connection, the name its own documentation uses, not the prefixed name the
 * model sees. Capability predicates read the MCP annotations; a tool that
 * carries none fails the predicate, so an unannotated server yields nothing
 * under `read_only` rather than silently passing everything through.
 */
export interface MCPTools {
    exclude?: string[];
    idempotent?: boolean | null;
    include?: string[];
    non_destructive?: boolean | null;
    read_only?: boolean | null;
}

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
 * sub-agent's `agent_id`/`session_id`. A connector sync carries none — its
 * `id` is the connection being fetched.
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
    status: EffectStatus;
    stream?: boolean | null;
    /**
     * The model tool call a delegation answers; its own `id` is the child session.
     */
    tool_call_id?: null | string;
}

/**
 * What kind of work an effect is. One enum for the wire and for the engine's
 * own scheduling: a decision and a turn's end queue beside the calls and are
 * swept the same way, so they are kinds too. Neither ever appears on an
 * [`Effect`] — a decision rides the decision list, a turn end has no record.
 *
 * Fetching one connection's tool list. Its `id` is the connection id.
 *
 * A worker decision.
 *
 * The turn's completion, dependent on its `turn.finished` finalizer
 * decision settling. Carries the turn id; the frozen output lives in the
 * session's `finalizing`. Never swept: it has no deadline of its own.
 */
export type EffectKind = "tool_call" | "sub_agent" | "llm_call" | "connector_sync" | "decision" | "turn_end";

export type EffectStatus = "pending" | "completed" | "failed" | "retry_scheduled" | "queued";

/**
 * The owner as delivered to the worker on `DecisionRequest.identity`: the
 * subject and its metadata, without the tenant. The tenant scopes the session
 * internally but is not sent to the worker.
 */
export interface WorkerIdentity {
    id?: null | string;
    metadata: { [key: string]: string };
}

export interface MessageTree {
    head_id?: null | string;
    nodes: Node[];
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
    tool_calls?: ToolCall[];
}

/**
 * The engine's default continuation for `trigger` (empty when it needs
 * worker knowledge). Advisory: accept by echoing it as the decision.
 *
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
    state?: unknown;
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
 * There is no `handler`: where a call runs follows from its name. A tool
 * resolved from a connector runs on the engine, a tool declared
 * `handler: client` runs on the client, and anything else runs on the
 * worker. The engine already knows all three, so asking the worker to
 * restate it only creates a way for the two to disagree.
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
    id?: null | string;
    /**
     * The `[llm.*]` block this call runs on; omitted ⇒ the merge source
     * config's `llm`. Naming a different block moves one call to another
     * venue or vendor.
     */
    llm?: null | string;
    max_completion_tokens?: number | null;
    messages?: DraftMessage[] | null;
    model?: null | string;
    reasoning?: ReasoningConfig | null;
    retry?: RetryPolicy | null;
    stream?: boolean | null;
    temperature?: number | null;
    tools?: LlmTool[] | null;
    type: DecisionActionType;
    arguments?: unknown;
    name?: string;
    attempt?: number | null;
    result?: unknown;
    /**
     * A neutral `LlmResponse`, or the provider's native response when the
     * answered `llm.execute` carried a `format`.
     */
    response?: unknown;
    code?: ErrorCode | null;
    detail?: unknown;
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
    payload?: unknown;
    reason?: string;
    data?: unknown;
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
    input?: unknown;
    name: string;
    /**
     * JSON Schema the settled result must satisfy; never sent to the model.
     * A violating result settles as a terminal tool error.
     */
    output?: unknown;
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
 *
 * Fired after a turn completes, carrying its final output; blocks the session
 * going idle until answered. Echo the proposed `done` to finalize.
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
    args?: unknown;
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
    request?: unknown;
    stream?: boolean;
    code?: ErrorCode | null;
    cost?: null | string;
    detail?: unknown;
    message?: DraftMessage | null;
    truncated?: boolean;
    usage?: unknown;
    agent_id?: string;
    session_id?: string;
    interrupt_id?: string;
    payload?: unknown;
    data?: unknown;
    turn_id?: string;
}

/**
 * OpenAI Chat Completions.
 *
 * Anthropic Messages API.
 */
export type LlmFormat = "openai" | "anthropic";

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
    value?: unknown;
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
    | "interrupt.resumed"
    | "turn.finished";

/**
 * An interrupt payload following the AG-UI Interrupt shape (spec spelling;
 * `id` and `reason` live on the interrupt itself).
 */
export interface InterruptPayload {
    /**
     * RFC 3339; display only until engine TTLs land.
     */
    expiresAt?: null | string;
    /**
     * Markdown; channels down-convert. Without it, channels fall back to
     * the interrupt's `reason`.
     */
    message?: null | string;
    /**
     * Free-form, delivered to clients verbatim. `metadata.options`
     * ([`InterruptOption`] list) renders as Slack buttons.
     */
    metadata?: unknown;
    /**
     * JSON Schema for the expected resolution payload.
     */
    responseSchema?: unknown;
    /**
     * Binds the interrupt to a prior tool call.
     */
    toolCallId?: null | string;
}

/**
 * A channel-authored resume payload: the AG-UI resume shape
 * (`{status, payload}`) plus a provenance stamp.
 */
export interface InterruptResolution {
    payload?: unknown;
    responder?: InterruptResponder | null;
    status: ResumeStatus;
}

/**
 * Who resolved it, stamped by the channel — never by the requester.
 */
export interface InterruptResponder {
    /**
     * The channel kind, e.g. `slack`, `ag-ui`.
     */
    channel: string;
    /**
     * The chosen option's label, when the resolution was a pick.
     */
    label?: null | string;
    /**
     * Channel-native user id.
     */
    user?: null | string;
}

export type ResumeStatus = "resolved" | "cancelled";

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
    tool_calls: ToolCallChunk[];
    turn_id?: null | string;
}
