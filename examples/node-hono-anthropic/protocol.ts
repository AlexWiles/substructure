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
    content?: ToolContent[] | null;
    id?: string;
    is_error?: boolean;
    result?: unknown;
    structured_content?: unknown;
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
    /**
     * Keep this tool out of the request. See [`LlmTool::defer`]. Absent ⇒
     * the agent's `defer_tools`.
     */
    defer?: boolean | null;
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

export interface ToolContent {
    text?: string;
    type: ToolContentType;
    data?: string;
    mimeType?: null | string;
    resource?: ResourceContents;
    name?: null | string;
    uri?: string;
}

export interface ResourceContents {
    blob?: null | string;
    mimeType?: null | string;
    text?: null | string;
    uri: string;
}

export type ToolContentType = "text" | "image" | "audio" | "resource" | "resource_link";

/**
 * The wire form of a [`Message`]: `id` is optional because a client-submitted or
 * worker-authored message is not yet recorded. `record`/`rerecord`
 * (`runtime::session::wire`) are the seams that lower it to the internal
 * [`Message`] (id always present) at recording time.
 */
export interface DraftMessage {
    content?: StoredContent[] | null | string;
    id?: null | string;
    name?: null | string;
    reasoning?: Reasoning | null;
    role: Role;
    tool_call_id?: null | string;
    tool_calls?: ToolCall[] | null;
}

export interface StoredContent {
    text?: string;
    type: StoredContentType;
    uri?: string;
    mimeType?: null | string;
    name?: null | string;
}

export type StoredContentType = "text" | "blob" | "link";

/**
 * What the model thought before it answered. `text` is for a reader; `blocks`
 * are the provider's own, held verbatim because Anthropic requires the
 * thinking that precedes a tool call back unmodified, signature included.
 */
export interface Reasoning {
    blocks?: unknown[];
    provider: ReasoningProvider;
    text?: null | string;
}

/**
 * Which provider wrote a [`Reasoning`]'s blocks. They ride back only to it:
 * another provider reads them as noise, and Anthropic rejects blocks it did
 * not sign.
 */
export type ReasoningProvider = "anthropic" | "openai" | "openrouter";

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
 * `llm` names the `[llm.*]` block every proposed call runs on, and so decides
 * both the venue (the engine with a vendor key, or the agent's own worker) and
 * the wire shape of a worker-run call. It is effectively required: a config
 * that names none fails when the engine resolves a call against it.
 */
export interface AgentConfig {
    /**
     * Where the engine tells the model that an MCP server is available, and
     * what that server says it is for.
     * Separate from `defer_tools`: a server exists whether or not its tools
     * are deferred, and where a notice lands is a fact about this agent's
     * prompt rather than about any server.
     */
    announce_mcp?: Announce;
    /**
     * Defer every tool this agent offers, from any source, unless the tool or
     * the connection says otherwise. Absent ⇒ the agent defers nothing of its
     * own; a connection may still defer on its own account.
     * Presence is the switch, so an agent cannot carry settings that do
     * nothing. Declared on the agent because an agent can hold this opinion
     * before it names a connection: one that sets it gets the search tools
     * from its first turn, so a connection added later costs no cache.
     */
    defer_tools?: boolean | null | DeferTools;
    /**
     * How hard the model thinks, carried on the agent because it pairs with
     * the model. Unset sends no reasoning config and leaves the provider its
     * own default.
     */
    effort?: ReasoningEffort | null;
    /**
     * The `[llm.*]` block this agent's calls run on.
     */
    llm?: null | string;
    /**
     * MCP servers
     */
    mcp?: MCPServer[];
    model: string;
    /**
     * Boxed: five per-kind overrides is a lot of bytes to carry inline
     * through every command that holds a config.
     */
    retry?: RetryConfig | null;
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
 * Where the engine tells the model that an MCP server is available, and
 * what that server says it is for.
 * Separate from `defer_tools`: a server exists whether or not its tools
 * are deferred, and where a notice lands is a fact about this agent's
 * prompt rather than about any server.
 *
 * Where an MCP announcement lands.
 *
 * The system prompt while no call has dispatched; then a block on the
 * last user message; then a message of its own. The engine takes the
 * first place it can use, so the order is not a setting.
 *
 * Nowhere. For a server whose own words help nobody.
 */
export type Announce = "auto" | "never";

/**
 * How an agent's deferred tools reach the model.
 */
export interface DeferTools {
    /**
     * The most matches one search answers with. Never zero: a search that can
     * answer with nothing is a search the model cannot use.
     */
    max_matches?: number;
    /**
     * Which tools the agent gets to reach the ones it defers.
     */
    strategy?: DeferToolsStrategy;
    [property: string]: unknown;
}

/**
 * Which tools the agent gets to reach the ones it defers.
 *
 * How the tools an agent defers reach the model.
 * The engine holds every deferred definition whatever this says, and answers
 * its own tools whatever this says. This chooses two things: which of those
 * tools the request advertises, and whether the request carries the deferred
 * definitions.
 * Declared on the agent, beside `defer_tools`: which tools an agent gets is
 * the agent's business, the same way as whether it defers at all.
 *
 * `tool_search` and `call_tool`. A search answers with the schema, so one
 * search is the whole distance to a call, and nothing hands the model a
 * name it cannot then reach.
 */
export type DeferToolsStrategy = "search";

export type ReasoningEffort = "xhigh" | "high" | "medium" | "low" | "minimal" | "none";

/**
 * An MCP server the agent draws tools from. `id` resolves against the engine's
 * connection registry — locally from `[mcp]` in `substructure.toml`, in the
 * cloud from the connections an admin granted this app. The worker never names
 * a URL or a credential.
 */
export interface MCPServer {
    auth_failure?: AuthFailure;
    id: string;
    /**
     * Narrows what the model sees. Absent ⇒ every tool the connection grants.
     */
    tools?: MCPTools | null;
}

/**
 * What a session does when a connection needs a person to authorize it. It
 * belongs to the pair: one credential serves an agent that stops and asks, and
 * an agent that has nobody to ask.
 *
 * Stop and ask. A channel that cannot show the question degrades instead.
 *
 * Go on without this connection's tools.
 */
export type AuthFailure = "interrupt" | "degrade";

/**
 * What the model sees of one connection, for one agent: which tools, and how
 * they reach the model.
 * The filter is applied in order — capability predicates, then `include`, then
 * `exclude` — and only ever narrowing, so a filter can never widen what the
 * connection grants. `defer` runs after it and removes nothing.
 * `include`/`exclude` are globs matched against the tool's name on the
 * connection, the name its own documentation uses, not the prefixed name the
 * model sees. Capability predicates read the MCP annotations; a tool that
 * carries none fails the predicate, so an unannotated server yields nothing
 * under `read_only` rather than silently passing everything through.
 */
export interface MCPTools {
    /**
     * Keep every surviving tool out of the request. See [`LlmTool::defer`].
     * Absent ⇒ the agent's `defer_tools`.
     */
    defer?: boolean | null;
    exclude?: string[];
    idempotent?: boolean | null;
    include?: string[];
    non_destructive?: boolean | null;
    read_only?: boolean | null;
}

/**
 * An agent's retry overrides, one per effect kind. `default` covers the kinds
 * that name nothing; a kind is layered on top of it, so the two compose.
 * Per kind because the kinds are not alike: an LLM call is idempotent and worth
 * retrying, a tool call may not be, and a connector fetch holds up every
 * decision behind it.
 */
export interface RetryConfig {
    connector?: RetryOverride | null;
    default?: RetryOverride | null;
    llm?: RetryOverride | null;
    sub_agent?: RetryOverride | null;
    tool?: RetryOverride | null;
}

/**
 * A partial retry policy: only the fields it names change, and the rest are
 * inherited. Every override is a layer over the engine's default for the effect
 * kind, so tuning one knob does not mean restating the other four — and leaving
 * a timeout out keeps the default bound rather than removing it.
 * An override cannot set a timeout back to unbounded. Waiting effectively
 * forever is a large number, which is also the honest way to say it.
 */
export interface RetryOverride {
    attempt_timeout_secs?: number | null;
    backoff_base_secs?: number | null;
    backoff_max_secs?: number | null;
    max_attempts?: number | null;
    total_timeout_secs?: number | null;
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

/**
 * Dispatched and alive, awaiting its result. Off the deadline clock: the
 * work succeeded in starting, and how long it then runs is its own
 * business. A delegation sits here for as long as its child turn takes.
 */
export type EffectStatus = "pending" | "completed" | "failed" | "retry_scheduled" | "queued" | "running";

/**
 * The owner as delivered to the worker on `DecisionRequest.identity`, without
 * the tenant. Read `kind` with `id`: only `frontend` is an end user.
 */
export interface WorkerIdentity {
    id?: null | string;
    kind?: OwnerKind;
    metadata: { [key: string]: string };
}

/**
 * What kind of caller owns a session. Part of the identity: only `frontend` is
 * an end user, and an ownership check grants access to no other kind.
 */
export type OwnerKind = "frontend" | "operator" | "api_key" | "system";

export interface MessageTree {
    head_id?: null | string;
    nodes: Node[];
}

export interface Node {
    message: Message;
    parent_id?: null | string;
}

export interface Message {
    content?: StoredContent[] | null | string;
    id: string;
    name?: null | string;
    reasoning?: Reasoning | null;
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
    /**
     * How each channel shows this decision, keyed by channel kind (e.g.
     * `slack`). Opaque to the engine.
     */
    channels?: { [key: string]: unknown };
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
 *
 * Resolve an open interrupt and resume the session.
 *
 * Fetch a connection's tools again, after a person replaced its
 * credential.
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
    /**
     * Layered over the agent config's `llm` policy, else over the engine's
     * default.
     *
     * Layered over the agent config's policy for this kind, else over the
     * engine's default for where the tool runs.
     *
     * Layered over the agent config's `sub_agent` policy, else over the
     * engine's default.
     */
    retry?: RetryOverride | null;
    stream?: boolean | null;
    temperature?: number | null;
    tools?: LlmTool[] | null;
    type: DecisionActionType;
    arguments?: unknown;
    name?: string;
    attempt?: number | null;
    content?: ToolContent[] | null;
    is_error?: boolean;
    result?: unknown;
    structured_content?: unknown;
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
    /**
     * The child's opening message. It travels with the spawn, so it
     * cannot race the creation of the session it opens.
     */
    message?: DraftMessage | null;
    session_id?: string;
    /**
     * The model tool-call this delegation answers — always required.
     */
    tool_call_id?: string;
    interrupt_id?: null | string;
    payload?: unknown;
    reason?: string;
    data?: unknown;
}

/**
 * What kind of failure, for a consumer that branches instead of reading the
 * sentence. A closed set, and required on every [`ErrorInfo`]: an optional
 * code is one nobody fills in, which leaves every consumer handling a `None`
 * that should not exist.
 * `provider_error`, `rate_limited`, `refused`, `budget_exceeded` and
 * `deadline_exceeded` describe a call that ran and went wrong.
 * `invalid_response` — a document did not parse, or parsed into something
 * unusable. `handler_error` — whoever was asked to do the work (a worker, a
 * client) reported a failure of its own. `worker_unreachable` — it was never
 * reached. `unroutable` — nothing could decide. `internal` — the engine's own
 * fault, and the honest answer when nothing else fits.
 */
export type ErrorCode =
    | "provider_error"
    | "rate_limited"
    | "refused"
    | "budget_exceeded"
    | "deadline_exceeded"
    | "invalid_response"
    | "handler_error"
    | "worker_unreachable"
    | "unroutable"
    | "internal";

export interface ReasoningConfig {
    effort?: ReasoningEffort | null;
    enabled?: boolean | null;
    exclude?: boolean | null;
    max_tokens?: number | null;
}

/**
 * A tool's declared contract: flat on the wire. Providers that need
 * OpenAI-style `{"type": "function", "function": {…}}` nesting re-wrap at
 * their own boundary.
 */
export interface LlmTool {
    /**
     * Keep this definition out of the request.
     * The engine still records it, still routes a call to it, and still finds
     * it in a search. Only the request omits it, which is what keeps a large
     * tool set out of the model's context and out of the cached prefix.
     * Any source can set it: a tool the config declares, a connection, or
     * whatever comes next. Deferral is a property of a tool, not of where it
     * came from.
     */
    defer?: boolean;
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
    | "interrupt.resolve"
    | "connector.sync"
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
    error?: ErrorInfo | null;
    ok?: boolean;
    result?: StoredResult | null | string;
    format?: LlmFormat | null;
    /**
     * The neutral `LlmRequest` JSON, or the provider's native request body
     * when `format` is set.
     */
    request?: unknown;
    stream?: boolean;
    cost?: null | string;
    message?: DraftMessage | null;
    /**
     * True when the model declined the request rather than answering it.
     * A refusal reads as a turn that stopped well and said nothing, so
     * without this the run continues from a blank answer.
     */
    refused?: boolean;
    truncated?: boolean;
    usage?: Usage | null;
    agent_id?: string;
    session_id?: string;
    interrupt_id?: string;
    payload?: unknown;
    data?: unknown;
    turn_id?: string;
}

/**
 * Why something failed. One shape on every event, on the wire, and in the
 * internal carriers that produce them — shaped after a Stripe API error.
 * `retryable` is deliberately absent: whether to try again is a decision the
 * engine makes about one attempt, not a fact about the failure, and it is
 * meaningless on a terminal like `turn.completed`. It rides on the events
 * that settle an attempt instead.
 */
export interface ErrorInfo {
    code: ErrorCode;
    /**
     * Small structured particulars: a status, the llm blocks that exist.
     */
    detail?: unknown;
    /**
     * One engine-authored sentence, safe to show a human. Never a raw
     * document — an unbounded body belongs in the log.
     */
    message: string;
    /**
     * The one input to go and fix, when the failure names one: `agent.llm`,
     * `actions[0].type`. Stripe's `param`.
     */
    param?: null | string;
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

export interface StoredResult {
    content?: StoredContent[];
    isError?: boolean;
    structuredContent?: unknown;
}

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
 * What one call read and wrote, in counts every provider means the same way.
 * Each vendor names and scopes these differently: Anthropic reports the part
 * of the prompt it did not read from the cache, OpenAI reports the whole
 * prompt including that part. A session that changes model, and a tree whose
 * agents name different blocks, add these counts together, so the adapter
 * normalizes them rather than the reader.
 */
export interface Usage {
    /**
     * The part of `input` the provider read from the cache.
     */
    cache_read: number;
    /**
     * The part of `input` the provider wrote to the cache.
     */
    cache_write: number;
    /**
     * Every input token of the call, cached or not.
     */
    input: number;
    output: number;
    /**
     * The counts as the provider reported them, for a reader that wants a
     * number this type does not name.
     */
    provider?: unknown;
    /**
     * `input` and `output` together.
     */
    total: number;
    /**
     * The part of `input` the provider read fresh.
     */
    uncached_input: number;
}

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
