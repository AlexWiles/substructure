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
 * Everything a client can send: submit a message, a full view, an append
 * batch, or a named action; resume an interrupt; or settle a client tool.
 *
 * Each variant carries only the addressing it needs. The four submit variants
 * carry `agent_id`, which routes the turn and starts the session if it is new,
 * and an optional `turn_id`. A resume or settle names an interrupt or effect
 * and continues whatever turn is running, so it carries neither.
 * `session_id` is on the envelope.
 *
 * The body of an interrupt resume: which interrupt, and the payload delivered
 * to the worker. Shared by the [`ClientInput::InterruptResume`] input and the
 * [`DecisionTrigger::InterruptResumed`] trigger.
 */
export interface ClientInput {
    agent_id?: string;
    message?: DraftMessage;
    /**
     * Hold this message for the next turn instead of refusing it while a
     * turn is running. Off by default.
     *
     * Hold this batch for the next turn instead of refusing it while a
     * turn is running. Off by default.
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
 * Inputs a client declares on its run, passed to the worker on the
 * `client.messages` decision. `tools` are the browser's own tools, read as
 * client-handled [`AgentTool`]s. The engine adds them to the proposed config.
 * A worker can override that by returning its own `agent`.
 *
 * Inputs the client declared on its run. The engine adds
 * `client.tools` to the proposed config.
 */
export interface ClientContext {
    context?: unknown[];
    forwarded_props?: unknown;
    state?: unknown;
    tools?: AgentTool[];
}

/**
 * A function tool the agent offers. `handler` says where a call runs:
 * `client` for the client, absent for the worker. `server` is invalid here.
 * The engine runs only connector tools, and a worker declares those by
 * connection id.
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
 * The engine makes the call.
 *
 * The worker executes it.
 *
 * The client executes it. The session goes idle until it answers.
 * Tools only.
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
 * The wire form of a [`Message`]. `id` is absent until the message is
 * recorded.
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
 * What the model thought before it answered. `text` is for a reader.
 * `blocks` are the provider's own and stay unchanged. Anthropic requires the
 * thinking before a tool call back with its signature.
 */
export interface Reasoning {
    blocks?: unknown[];
    provider: ReasoningProvider;
    text?: null | string;
}

/**
 * Which provider wrote the blocks. They go back only to that provider.
 * Anthropic rejects blocks it did not sign.
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
 * What a client submits: a message, its full conversation view, an append
 * batch, or a named action. The engine turns it into events and never stores
 * it as it arrived.
 *
 * The body of a `client.message`: one message, optionally streamed.
 *
 * The body of a `client.messages`: the client's full conversation view, optionally
 * streamed.
 *
 * The body of a `client.append`. Messages are added at the session head,
 * against the active path at delivery. A queued append lands after whatever
 * turn beat it, so it cannot fork the tree. A message whose id is already
 * recorded is dropped.
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
     * The agent config for the active path. `null` when none is set.
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
     * The engine's default continuation for `trigger`. Empty when only the
     * worker can decide. Accept it by echoing it back.
     */
    proposed: DecisionResponse;
    session_id: string;
    state: unknown;
    trigger: DecisionTrigger;
    turn_id?: null | string;
}

/**
 * A declared agent. The same shape whether a file writes it or a worker
 * returns it.
 *
 * `llm` names the `[llm.*]` block every call runs on. That block decides where
 * the call runs and what shape it takes. A config that names no block fails
 * when the engine resolves a call.
 */
export interface AgentConfig {
    /**
     * Defer every tool this agent offers, whatever its source. A tool or a
     * connection overrides this with its own `defer`. Absent, the agent defers
     * nothing; a connection can still defer on its own.
     */
    defer_tools?: boolean | null | DeferTools;
    /**
     * How hard the model thinks. Unset leaves the provider's own default.
     */
    effort?: ReasoningEffort | null;
    /**
     * The `[llm.*]` block this agent's calls run on.
     */
    llm?: null | string;
    /**
     * MCP servers this agent draws tools from.
     */
    mcp?: MCPServer[];
    /**
     * Whether the engine tells the model that an MCP server is available, and
     * what that server says it is for.
     */
    mcp_announce?: MCPAnnounce;
    model: string;
    /**
     * Plugins this agent uses.
     */
    plugins?: AgentPlugin[];
    /**
     * Boxed. Five per-kind overrides are too many bytes to carry inline.
     */
    retry?: RetryConfig | null;
    /**
     * Sub-agents the model can delegate to. The model sees them as tools.
     * Each call starts a child session.
     */
    sub_agents?: SubAgent[];
    system?: null | string;
    /**
     * Worker- or client-executed tools the model can call.
     */
    tools?: AgentTool[];
}

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
 *
 * The engine holds every deferred definition whatever this says. This chooses
 * which tools the request advertises, and whether the request carries the
 * deferred definitions.
 *
 * `tool_search` and `call_tool`. A search answers with the schema, so one
 * search is enough to make a call.
 */
export type DeferToolsStrategy = "search";

export type ReasoningEffort = "xhigh" | "high" | "medium" | "low" | "minimal" | "none";

/**
 * An MCP server the agent draws tools from. `path` names a connection the
 * engine holds. A worker never writes a URL or a credential.
 */
export interface MCPServer {
    approve?: Approve;
    auth_failure?: MCPAuthFailure;
    path: string;
    tool_sync_failure?: MCPToolSyncFailure;
    /**
     * Narrows what the model sees. Absent ⇒ every tool the connection grants.
     */
    tools?: MCPTools | null;
}

/**
 * Which of a connection's calls stop for a person.
 *
 * A tool that the connection marks `destructiveHint`.
 */
export type Approve = "never" | "always" | "destructive";

/**
 * What a session does when a connection needs a person to authorize it.
 *
 * Stop and ask. A channel that cannot show the question degrades.
 *
 * Go on without this connection's tools.
 */
export type MCPAuthFailure = "interrupt" | "degrade";

/**
 * Whether the model is told that a connection's tool fetch failed. The turn
 * goes ahead without those tools either way.
 *
 * Name the connection wherever its tools would have been.
 *
 * Say nothing. For a connection the agent does not need.
 */
export type MCPToolSyncFailure = "warn" | "silent";

/**
 * Which of a connection's tools the model sees, and how they reach it.
 *
 * The filter runs in order: capability predicates, then `include`, then
 * `exclude`. Each step only removes, so a filter cannot widen what the
 * connection grants. `defer` runs last and removes nothing.
 *
 * `include` and `exclude` are globs over the tool's name on the connection,
 * not the prefixed name the model sees.
 *
 * Capability predicates read the MCP annotations. A tool with no annotation
 * fails the predicate, so a server that annotates nothing yields nothing under
 * `read_only`.
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
 * Whether the engine tells the model that an MCP server is available, and
 * what that server says it is for.
 *
 * Where an MCP announcement lands.
 *
 * The system prompt while no call has run. Then a block on the last user
 * message. Then a message of its own. The engine takes the first place it
 * can use.
 *
 * Nowhere. For a server whose own description does not help.
 */
export type MCPAnnounce = "auto" | "never";

/**
 * A plugin an agent uses. The skills and servers come from the bundle when the
 * config loads.
 */
export interface AgentPlugin {
    approve?: Approve;
    auth_failure?: MCPAuthFailure;
    description?: string;
    id: string;
    /**
     * Where each of this plugin's servers is declared.
     */
    servers?: string[];
    skills?: SkillMeta[];
    tool_sync_failure?: MCPToolSyncFailure;
    /**
     * Applied to each of the plugin's servers.
     */
    tools?: MCPTools | null;
}

/**
 * What the model sees of a skill before it loads it.
 */
export interface SkillMeta {
    description?: string;
    name: string;
}

/**
 * Retry overrides, one for each effect kind. `default` covers the kinds that
 * name nothing. A kind layers on top of `default`.
 */
export interface RetryConfig {
    connector?: RetryOverride | null;
    default?: RetryOverride | null;
    llm?: RetryOverride | null;
    sub_agent?: RetryOverride | null;
    tool?: RetryOverride | null;
}

/**
 * Only the fields it names change. An override cannot make a timeout
 * unbounded.
 */
export interface RetryOverride {
    backoff_base_secs?: number | null;
    backoff_max_secs?: number | null;
    max_attempts?: number | null;
    queue_timeout_secs?: number | null;
    run_timeout_secs?: number | null;
    total_timeout_secs?: number | null;
}

/**
 * A sub-agent the model can delegate to. `id` is both the child agent and the
 * tool name the model calls. Its input is one `message`.
 */
export interface SubAgent {
    description?: string;
    id: string;
}

/**
 * An effect still running, shown on each worker decision. A flat envelope
 * plus the fields of its kind. A connector sync carries none. Its `id` is the
 * connection being fetched.
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
     * The model tool call a delegation answers. Its `id` is the child
     * session.
     */
    tool_call_id?: null | string;
}

/**
 * What kind of work an effect is. One enum for the wire and for scheduling. A
 * decision and a turn's end queue beside the calls and are swept the same way,
 * so they are kinds too. Neither appears on an [`Effect`].
 *
 * Fetching one connection's tool list. Its `id` is the connection id.
 *
 * A worker decision.
 *
 * The turn's completion, which waits for the `turn.finished` decision to
 * settle. Carries the turn id. Never swept, because it has no deadline.
 */
export type EffectKind = "tool_call" | "sub_agent" | "llm_call" | "connector_sync" | "decision" | "turn_end";

/**
 * Running, waiting for its result. Off the deadline clock. A delegation
 * stays here for as long as its child turn takes.
 */
export type EffectStatus = "pending" | "completed" | "failed" | "retry_scheduled" | "queued" | "running";

/**
 * The owner as the worker receives it, without the tenant. Read `kind` with
 * `id`. Only `frontend` is an end user.
 */
export interface WorkerIdentity {
    metadata: { [key: string]: string };
    subject?: Subject | null;
    visibility?: Visibility;
}

/**
 * One identity, as the source that authenticated it named it. An id is
 * unique only within its issuer.
 */
export interface Subject {
    id: string;
    issuer: string;
}

/**
 * Who can read what a session says. The transport sets it once, when the
 * session starts. Absent or unknown reads as `shared`. `shared` never selects
 * a personal credential.
 *
 * More than one person can read the answer.
 *
 * One person only.
 */
export type Visibility = "shared" | "private";

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
 * The engine's default continuation for `trigger`. Empty when only the
 * worker can decide. Accept it by echoing it back.
 *
 * The messages and actions to author, plus optional state and agent writes.
 * A worker returns one. The engine proposes one too, which the worker echoes
 * or changes.
 */
export interface DecisionResponse {
    actions?: DecisionAction[];
    /**
     * A new agent config write; omitted keeps the current config.
     */
    agent?: AgentConfig | null;
    /**
     * How each channel shows this decision, keyed by channel kind. The engine
     * does not read it.
     */
    channels?: { [key: string]: unknown };
    messages?: DraftMessage[];
    /**
     * Absent or `null` keeps the current state. Send an empty value to
     * clear it.
     */
    state?: unknown;
}

/**
 * The action a worker writes on the wire. A settle can leave out the effect
 * id, because the `*.execute` trigger it answers already names it.
 *
 * A flat LLM request. Every field is optional.
 *
 * Without `id`, the engine mints one, and it becomes the assistant node's
 * id. A field left out comes from the agent config, then from the engine's
 * default. Without `messages`, the request carries the config's system
 * message and the decision's view. Given `messages`, no system message is
 * added.
 *
 * Without `id`, the engine mints one. A tool the model called carries the
 * model's id.
 *
 * There is no `handler`. The name says where the call runs: a connector
 * tool on the engine, a `handler: client` tool on the client, anything
 * else on the worker. The engine knows all three already.
 *
 * Without `id` and `attempt`, both come from the `tool.execute` trigger
 * this answers. That ties the result to the attempt that ran.
 *
 * Without `id` and `attempt`, both come from the `llm.execute` trigger
 * this answers. That ties the result to the attempt that ran.
 *
 * Without `interrupt_id`, the engine mints one to match the later
 * resume.
 *
 * Resolve an open interrupt and resume the session.
 *
 * Fetch a connection's tools again, after a person replaced its
 * credential.
 */
export interface DecisionAction {
    id?: null | string;
    /**
     * The `[llm.*]` block this call runs on. Absent uses the config's
     * `llm`. Naming another block moves this one call elsewhere.
     */
    llm?: null | string;
    max_completion_tokens?: number | null;
    messages?: DraftMessage[] | null;
    model?: null | string;
    reasoning?: ReasoningConfig | null;
    /**
     * Layered over the agent config's `llm` policy, or over the engine's
     * default.
     *
     * Layered over the agent config's policy for this kind, or over the
     * engine's default for where the tool runs.
     *
     * Layered over the agent config's `sub_agent` policy, or over the
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
     * An `LlmResponse`, or the provider's own response when the
     * `llm.execute` this answers carried a `format`.
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
     * cannot arrive before the session exists.
     */
    message?: DraftMessage | null;
    session_id?: string;
    /**
     * The model tool call this delegation answers. Required.
     */
    tool_call_id?: string;
    interrupt_id?: null | string;
    payload?: unknown;
    reason?: string;
    path?: string;
    data?: unknown;
}

/**
 * What kind of failure, so a consumer can branch on it instead of reading the
 * sentence. A closed set, required on every [`ErrorInfo`].
 *
 * `provider_error`, `rate_limited`, `refused`, `budget_exceeded`, and
 * `deadline_exceeded` mean a call ran and went wrong.
 * `invalid_response` means a document did not parse, or parsed into something
 * unusable. `handler_error` means the worker or client reported its own
 * failure. `worker_unreachable` means it was never reached. `unroutable`
 * means nothing could decide. `internal` means the engine's own fault.
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
 * A tool's declared contract. Flat on the wire. A provider that needs it
 * nested re-wraps it at its own boundary.
 */
export interface LlmTool {
    /**
     * Keep this definition out of the request.
     *
     * The engine still records it, still routes a call to it, and still finds
     * it in a search. Only the request leaves it out. That keeps a large tool
     * set out of the model's context and out of the cached prefix.
     *
     * Any source can set it. Deferral belongs to a tool, not to where the tool
     * came from.
     */
    defer?: boolean;
    description: string;
    /**
     * JSON Schema for the arguments. Absent declares a tool with no
     * arguments. The engine checks every call against it.
     */
    input?: unknown;
    name: string;
    /**
     * JSON Schema the result must satisfy. The model never sees it. A result
     * that breaks it becomes a terminal tool error.
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
 * The trigger a worker sees on the wire. There is no `ClientMessage`: the
 * engine turns a bare client message into `ClientTranscript` before it sends
 * it.
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
 * Sent after a turn completes, with its final output. The session stays
 * busy until it is answered. Echo the proposed `done` to finish.
 */
export interface DecisionTrigger {
    type: DecisionTriggerType;
    /**
     * Inputs the client declared on its run. The engine adds
     * `client.tools` to the proposed config.
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
     * What the engine made of `arguments` against the tool's `input`
     * schema: `valid`, `invalid`, or `malformed`. Always on the wire.
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
     * True when the model declined the request. Without it, a refusal
     * looks like a turn that ended well and said nothing.
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
 * Why something failed. One shape on every event and on the wire.
 *
 * There is no `retryable` field. Whether to try again is a decision about one
 * attempt, not a fact about the failure. The events that settle an attempt
 * carry it instead.
 */
export interface ErrorInfo {
    code: ErrorCode;
    /**
     * Small structured details, such as a status or the llm blocks that
     * exist.
     */
    detail?: unknown;
    /**
     * One sentence the engine wrote, safe to show a human. Never a raw
     * document. An unbounded body belongs in the log.
     */
    message: string;
    /**
     * The one input to fix, when the failure names one. For example
     * `agent.llm` or `actions[0].type`.
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
 * What the engine made of `arguments` against the tool's `input`
 * schema: `valid`, `invalid`, or `malformed`. Always on the wire.
 *
 * What the engine made of a tool call's arguments, sent with the raw
 * `arguments` string. Always on the wire.
 *
 * Parsed, and valid against the `input` schema if the tool declares one.
 * `value` is the parsed `arguments`. The engine never changes it.
 *
 * Parsed to an object that violates the declared `input` schema.
 *
 * Not a JSON object. Either malformed JSON or another type.
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
 * What one call read and wrote. Every provider means these counts the same
 * way.
 *
 * Vendors report different things. Anthropic gives the part of the prompt it
 * did not read from the cache. OpenAI gives the whole prompt. Each adapter
 * converts to this shape, because these counts get added together.
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
     * The counts as the provider reported them, for a number this type does
     * not name.
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
 * An interrupt payload in the AG-UI shape. `id` and `reason` live on the
 * interrupt itself.
 */
export interface InterruptPayload {
    /**
     * RFC 3339. Display only.
     */
    expiresAt?: null | string;
    /**
     * Markdown. A channel converts it as it needs. Without it, a channel
     * shows the interrupt's `reason`.
     */
    message?: null | string;
    /**
     * Free-form, delivered to clients unchanged. `metadata.options` is a
     * list of [`InterruptOption`], which Slack shows as buttons.
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
 * A resume payload a channel wrote: the AG-UI shape plus who resolved it.
 */
export interface InterruptResolution {
    payload?: unknown;
    responder?: InterruptResponder | null;
    status: ResumeStatus;
}

/**
 * Who resolved it. The channel sets this, never the requester.
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
     * The chosen option's `style`, when the resolution was a pick.
     */
    style?: null | string;
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
