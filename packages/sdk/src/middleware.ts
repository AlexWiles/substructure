import type {
    DecisionTrigger,
    LlmRequest,
    LlmTool,
    Message,
    RetryPolicy,
    ToolHandler,
    ToolResult,
    WorkerAction,
} from "./types";
import type { Handler, AgentRequest, AgentResponse, MiddlewareFn, Next, StateContributor } from "./worker";

export const DEFAULT_RETRY: RetryPolicy = {
    timeout_secs: 120,
    max_retries: 0,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

// ── Core primitives ────────────────────────────────────────────────────────

function decodeWorkerState(raw: string): unknown {
    if (!raw || raw === "") return {};
    return JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(raw), (c) => c.charCodeAt(0))));
}

function encodeWorkerState(value: unknown): string {
    return btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(value))));
}

/** Initialize a state slice's keys on a raw state object. */
function initSlice<A extends object>(rawState: unknown, init: A): A {
    const state = (rawState && typeof rawState === "object" ? rawState : {}) as Record<string, unknown>;
    for (const key of Object.keys(init)) {
        state[key] ??= structuredClone((init as Record<string, unknown>)[key]);
    }
    return state as A;
}

/**
 * The single primitive for defining middleware.
 *
 * With `state`: initializes a state slice and returns a `StateContributor`.
 * Without `state`: wraps a handler function.
 */
export function middleware<A extends object>(config: {
    state: A;
    handler?: (ctx: AgentRequest<A>, next: Next<A>) => Promise<AgentResponse> | AgentResponse;
}): StateContributor<A> & MiddlewareFn<unknown, A> & { _init: A };
export function middleware<S = unknown>(config: {
    handler: (ctx: AgentRequest<S>, next: Next<S>) => Promise<AgentResponse> | AgentResponse;
}): MiddlewareFn<S>;
export function middleware<A extends object>(config: {
    state?: A;
    handler?: (ctx: AgentRequest<A>, next: Next<A>) => Promise<AgentResponse> | AgentResponse;
}): MiddlewareFn<unknown> {
    const handler = config.handler ?? ((req: AgentRequest<A>, next: Next<A>) => next(req));

    if (!config.state) {
        return handler as MiddlewareFn<unknown>;
    }

    const init = config.state;
    const mw = (req: AgentRequest<unknown>, next: Next<A>) => {
        const state = initSlice(req.state, init);
        const typedReq: AgentRequest<A> = { ...req, state: state as A };
        return handler(typedReq, next);
    };

    return Object.assign(mw, { _contributes: init, _init: init }) as StateContributor<A> &
        MiddlewareFn<unknown, A> & { _init: A };
}

export type StateSliceMw<A extends object> = StateContributor<A> & MiddlewareFn<unknown, A> & { _init: A };

/**
 * Declare a state slice. Returns an object that works as:
 * - Middleware via `.use(slice)` — initializes the slice keys
 * - State reference via `tool({ state: slice })` — gives tools typed access
 */
export function stateSlice<A extends object>(init: A): StateSliceMw<A> {
    return middleware({ state: init });
}

// ── JSON state codec ───────────────────────────────────────────────────────

/**
 * Base64 JSON state serialization middleware.
 * Decodes `wire.worker_state` into `req.state` (falling back to `{}`),
 * runs the chain, then encodes `res.state` back into `workerState`.
 */
export function jsonState(): MiddlewareFn<unknown> {
    return middleware({
        handler: async (req: AgentRequest<unknown>, next: Next<unknown>) => {
            const enriched: AgentRequest<unknown> = {
                ...req,
                state: decodeWorkerState(req.wire.worker_state),
            };
            const res = await next(enriched);
            return {
                ...res,
                workerState: encodeWorkerState(res.state),
            };
        },
    });
}

// ── Tool ───────────────────────────────────────────────────────────────────

/**
 * Sentinel returned by `ctx.defer()` to tell the `tools` middleware to
 * skip emitting `return.tool.result` / `return.tool.error` for this call.
 * The worker submits zero actions for the `tool.execute` trigger and the
 * engine leaves the tool call pending until `submitToolCallResult` is
 * called with the eventual outcome.
 *
 * Tools should call `ctx.defer()` rather than importing this value.
 */
export const DEFERRED: unique symbol = Symbol.for("substructure.tool.deferred");
export type Deferred = typeof DEFERRED;

/** Per-call context passed to a tool's `execute`. Carries the
 *  identifiers needed to complete the call out-of-band, plus the
 *  `defer()` helper used to signal that completion will arrive later. */
export interface ToolExecutionContext {
    sessionId: string;
    toolCallId: string;
    attempt: number;
    /** Signal that this tool will deliver its result later via
     *  `submitToolCallResult`. Return the value: `return ctx.defer();`. */
    defer: () => Deferred;
}

export type ToolFn = (args: string, state?: unknown, ctx?: ToolExecutionContext) => Promise<unknown>;

export interface ToolDef {
    name: string;
    description: string;
    parameters: unknown;
    execute: ToolFn;
    retry?: RetryPolicy;
    stateSlice?: StateSliceMw<any>;
    /** Who completes this tool. Defaults to "worker". Set to "client" to
     *  declare a browser/frontend tool: the worker's `execute` should
     *  return `ctx.defer()`, and the result is submitted via
     *  `submitToolCallResult` from a frontend client. */
    handler?: ToolHandler;
}

/** Define a tool from plain data. */
export function tool<S extends object = never>(config: {
    name: string;
    description: string;
    parameters: unknown;
    state?: StateSliceMw<S>;
    execute: [S] extends [never]
        ? (args: string, ctx: ToolExecutionContext) => unknown | Promise<unknown>
        : (args: string, state: S, ctx: ToolExecutionContext) => unknown | Promise<unknown>;
    retry?: RetryPolicy;
    handler?: ToolHandler;
}): ToolDef {
    return {
        name: config.name,
        description: config.description,
        parameters: config.parameters,
        execute: async (args: string, state?: unknown, ctx?: ToolExecutionContext) => {
            if (config.state) {
                return (config.execute as (a: string, s: unknown, c: ToolExecutionContext) => unknown)(
                    args,
                    state,
                    ctx as ToolExecutionContext,
                );
            }
            return (config.execute as (a: string, c: ToolExecutionContext) => unknown)(
                args,
                ctx as ToolExecutionContext,
            );
        },
        retry: config.retry,
        stateSlice: config.state,
        handler: config.handler,
    };
}

// ── Client actions ─────────────────────────────────────────────────────────

export type ActionHandlerResult = void | WorkerAction[];

export interface ActionDef<Args = unknown, S = unknown> {
    name: string;
    /** Optional JSON schema for `args`. Currently documentation-only; no
     *  runtime validation is performed. */
    parameters?: unknown;
    stateSlice?: StateSliceMw<any>;
    handler: (args: Args, state: S) => ActionHandlerResult | Promise<ActionHandlerResult>;
}

/**
 * Define a client.action handler with typed args.
 *
 * Pair with `actions()` to dispatch by `client.action.name`. The handler
 * receives `args` cast to `Args` (no runtime validation) and the typed
 * state slice if `state` was supplied. A `void` return lets the chain
 * proceed (e.g. the LLM call fires on the trigger); returning a
 * `WorkerAction[]` short-circuits with those actions.
 */
export function action<Args = unknown, S extends object = never>(config: {
    name: string;
    parameters?: unknown;
    state?: StateSliceMw<S>;
    handler: [S] extends [never]
        ? (args: Args) => ActionHandlerResult | Promise<ActionHandlerResult>
        : (args: Args, state: S) => ActionHandlerResult | Promise<ActionHandlerResult>;
}): ActionDef<Args, S> {
    return {
        name: config.name,
        parameters: config.parameters,
        stateSlice: config.state,
        handler: config.handler as ActionDef<Args, S>["handler"],
    };
}

/**
 * Dispatch `client.action` triggers to matching `ActionDef`s by name.
 * Non-matching triggers and non-client-action triggers pass through.
 */
export function actions(defs: ActionDef<any, any>[]): MiddlewareFn<unknown> {
    const byName: Record<string, ActionDef<any, any>> = {};
    for (const def of defs) byName[def.name] = def;

    return middleware({
        handler: async (req, next) => {
            const { trigger } = req;
            if (trigger.type !== "client.action") return next(req);

            const def = byName[trigger.name];
            if (!def) return next(req);

            const state = def.stateSlice ? initSlice(req.state, def.stateSlice._init) : req.state;
            const result = await def.handler(trigger.args ?? {}, state);

            if (Array.isArray(result)) {
                return { state: req.state, actions: result };
            }
            return next(req);
        },
    });
}

// ── Logging ────────────────────────────────────────────────────────────────

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface Logger {
    debug(msg: string, data?: Record<string, unknown>): void;
    info(msg: string, data?: Record<string, unknown>): void;
    warn(msg: string, data?: Record<string, unknown>): void;
    error(msg: string, data?: Record<string, unknown>): void;
}

export interface LoggingOptions {
    label?: string;
    logger?: Logger;
    level?: LogLevel;
}

const LOG_LEVELS: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

function defaultLogger(minLevel: LogLevel): Logger {
    const min = LOG_LEVELS[minLevel];
    const noop = () => {};
    const emit = (level: LogLevel) => {
        if (LOG_LEVELS[level] < min) return noop;
        return (msg: string, data?: Record<string, unknown>) => {
            console.log(JSON.stringify({ level, msg, ...data, ts: new Date().toISOString() }));
        };
    };
    return { debug: emit("debug"), info: emit("info"), warn: emit("warn"), error: emit("error") };
}

export function logging(options?: string | LoggingOptions): MiddlewareFn<unknown, unknown> {
    const opts: LoggingOptions = typeof options === "string" ? { label: options } : (options ?? {});
    const level = opts.level ?? "info";
    const log = opts.logger ?? defaultLogger(level);
    const label = opts.label;

    return middleware({
        handler: async (req, next) => {
            const t = req.trigger;
            const tag = t.type === "tool.execute" ? `${t.type}:${t.name}` : t.type;
            const ctx: Record<string, unknown> = {
                agent: req.agentId,
                session: req.wire.session_id,
                decision: req.wire.decision_id,
                trigger: tag,
            };
            if (label) ctx.label = label;

            log.info("decision.start", ctx);
            log.debug("decision.trigger", { ...ctx, payload: t });

            const start = performance.now();
            try {
                const result = await next(req);
                const durationMs = Number((performance.now() - start).toFixed(1));
                const actionTypes = result.actions.map((a) => a.type);

                log.info("decision.end", { ...ctx, actions: actionTypes, durationMs });
                log.debug("decision.actions", { ...ctx, actions: result.actions, durationMs });

                return result;
            } catch (err) {
                const durationMs = Number((performance.now() - start).toFixed(1));
                log.error("decision.error", {
                    ...ctx,
                    error: err instanceof Error ? err.message : String(err),
                    durationMs,
                });
                throw err;
            }
        },
    });
}

// ── Message history ────────────────────────────────────────────────────────

export type MessageSelector<S> = (state: S, req: AgentRequest<S>) => Message[];

/**
 * Translate a decision trigger into the message (if any) that belongs in the
 * conversation transcript. Returns `null` for triggers that don't correspond
 * to a turn entry (`client.action`, `tool.execute`, `llm.error`, ...).
 *
 * Building block for custom history middleware. Pair with `withHistory`
 * on the way out.
 */
export function triggerToMessage(trigger: DecisionTrigger): Message | null {
    switch (trigger.type) {
        case "user.message":
        case "llm.response":
            return trigger.message;
        case "tool.result":
            return {
                role: "tool",
                content: trigger.result.content,
                tool_call_id: trigger.result.tool_call_id,
                name: trigger.result.name,
            };
        default:
            return null;
    }
}

/**
 * Prepend a transcript to every `call.llm` action's `messages`. Non-LLM
 * actions pass through unchanged.
 *
 * Building block for custom history middleware: pair with `triggerToMessage`
 * to record on the way in, then call this on the way out to attach the
 * transcript to whatever LLM call the chain below produced.
 */
export function prependHistoryToLlmCalls(history: Message[], actions: WorkerAction[]): WorkerAction[] {
    return actions.map((action) =>
        action.type === "call.llm"
            ? {
                  ...action,
                  request: {
                      ...action.request,
                      messages: [...history, ...action.request.messages],
                  },
              }
            : action,
    );
}

/**
 * Conversation history middleware. Contributes `{ messages: Message[] }` to state.
 * Records incoming messages and augments `call_llm` actions with the full history.
 *
 * Implementation is intentionally tiny: `triggerToMessage` +
 * `prependHistoryToLlmCalls` composed against a state slice. For anything
 * non-default (clear on a signal, cap at N, persist out-of-band) write
 * your own middleware using the same helpers.
 */
export function messageHistory(): StateContributor<{ messages: Message[] }> &
    MiddlewareFn<unknown, { messages: Message[] }> {
    return middleware({
        state: { messages: [] as Message[] },
        handler: async (req, next) => {
            const msg = triggerToMessage(req.trigger);
            if (msg) req.state.messages.push(msg);

            const result = await next(req);
            return {
                ...result,
                actions: prependHistoryToLlmCalls(req.state.messages, result.actions),
            };
        },
    });
}

/**
 * Like `messageHistory`, but only retains messages from the current turn.
 * Resets the buffer whenever `req.wire.turn_id` changes.
 *
 * Requires callers to supply a distinct `turn_id` per submit. If `turn_id` is
 * omitted by the caller, this degrades to the same behavior as `messageHistory`.
 */
export function messageHistoryCurrentTurn(): StateContributor<{
    messages: Message[];
    lastTurnId?: string;
}> &
    MiddlewareFn<unknown, { messages: Message[]; lastTurnId?: string }> {
    return middleware({
        state: { messages: [] as Message[], lastTurnId: undefined as string | undefined },
        handler: async (req, next) => {
            if (req.wire.turn_id !== req.state.lastTurnId) {
                req.state.messages = [];
                req.state.lastTurnId = req.wire.turn_id;
            }

            const msg = triggerToMessage(req.trigger);
            if (msg) req.state.messages.push(msg);

            const result = await next(req);
            return {
                ...result,
                actions: prependHistoryToLlmCalls(req.state.messages, result.actions),
            };
        },
    });
}

// ── System message ─────────────────────────────────────────────────────────

export type SystemMessageSelector<S> = (state: S, req: AgentRequest<S>) => string;

export function systemMessage<S>(selectorOrValue: SystemMessageSelector<S> | string): MiddlewareFn<S> {
    const selector: SystemMessageSelector<S> =
        typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;

    return middleware({
        handler: async (req: AgentRequest<S>, next: Next<S>) => {
            const sysMsg: Message = { role: "system", content: selector(req.state, req) };

            const result = await next(req);
            const actions = result.actions.map((action) => {
                if (action.type !== "call.llm") return action;
                return {
                    ...action,
                    request: {
                        ...action.request,
                        messages: [sysMsg, ...action.request.messages],
                    },
                };
            });

            return { ...result, actions };
        },
    });
}

// ── Tools ──────────────────────────────────────────────────────────────────

export type ToolInput = Record<string, ToolDef> | ToolDef[];
export type ToolSelector<S> = (state: S, req: AgentRequest<S>) => ToolInput;

function resolveTools(input: ToolInput): Record<string, ToolDef> {
    if (Array.isArray(input)) {
        const record: Record<string, ToolDef> = {};
        for (const def of input) {
            record[def.name] = def;
        }
        return record;
    }
    return input;
}

/**
 * Tool execution and pending-tracking middleware.
 * Contributes `{ pendingToolCalls: string[] }` to state.
 *
 * On `llm_response`: records tool_call_ids as pending.
 * On `tool_result`: removes the ID. Suppresses `call_llm` until all results are in.
 * On `tool_execute`: executes the tool and prepends the result to downstream actions.
 */
export function tools<S>(
    selectorOrValue: ToolSelector<S> | ToolInput,
): StateContributor<{ pendingToolCalls: string[] }> & MiddlewareFn<unknown, { pendingToolCalls: string[] }> {
    const selector: ToolSelector<S> = typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;

    return middleware({
        state: { pendingToolCalls: [] as string[] },
        handler: async (req, next) => {
            const toolMap = resolveTools(selector(req.state as S, req as unknown as AgentRequest<S>));

            const downstream = await next(req);

            // Handle tool execution
            if (req.trigger.type === "tool.execute") {
                const t = toolMap[req.trigger.name];
                if (!t) {
                    return {
                        ...downstream,
                        actions: [
                            {
                                type: "return.tool.error" as const,
                                tool_call_id: req.trigger.tool_call_id,
                                error: `Unknown tool: ${req.trigger.name}`,
                                retryable: false,
                                attempt: req.trigger.attempt,
                            },
                            ...downstream.actions,
                        ],
                    };
                }

                try {
                    let toolState: unknown;
                    if (t.stateSlice) {
                        toolState = initSlice(req.state, t.stateSlice._init);
                    }
                    const ctx: ToolExecutionContext = {
                        sessionId: req.wire.session_id,
                        toolCallId: req.trigger.tool_call_id,
                        attempt: req.trigger.attempt,
                        defer: () => DEFERRED,
                    };
                    const output = await t.execute(req.trigger.arguments, toolState, ctx);
                    if (output === DEFERRED) {
                        return downstream;
                    }
                    return {
                        ...downstream,
                        actions: [
                            {
                                type: "return.tool.result" as const,
                                tool_call_id: req.trigger.tool_call_id,
                                result: typeof output === "string" ? output : JSON.stringify(output),
                                attempt: req.trigger.attempt,
                            },
                            ...downstream.actions,
                        ],
                    };
                } catch (error: unknown) {
                    return {
                        ...downstream,
                        actions: [
                            {
                                type: "return.tool.error" as const,
                                tool_call_id: req.trigger.tool_call_id,
                                error: error instanceof Error ? error.message : String(error),
                                retryable: false,
                                attempt: req.trigger.attempt,
                            },
                            ...downstream.actions,
                        ],
                    };
                }
            }

            // Track pending tool calls from LLM response and emit call_tool actions
            // Only emit call.tool for tools this middleware knows about.
            if (req.trigger.type === "llm.response") {
                const toolCalls = req.trigger.message.tool_calls;
                if (toolCalls && toolCalls.length > 0) {
                    const known = toolCalls.filter((tc) => tc.function.name in toolMap);
                    req.state.pendingToolCalls = known.map((tc) => tc.id);

                    const callToolActions: WorkerAction[] = known.map((tc) => {
                        const def = toolMap[tc.function.name];
                        return {
                            type: "call.tool" as const,
                            tool_call_id: tc.id,
                            name: tc.function.name,
                            arguments: tc.function.arguments,
                            handler: def?.handler ?? ("worker" as const),
                            retry: def?.retry ?? DEFAULT_RETRY,
                        };
                    });

                    return {
                        ...downstream,
                        actions: [...callToolActions, ...downstream.actions],
                    };
                }
            }

            // Augment actions with tool definitions and retry policies
            const actions = downstream.actions.map((action) => {
                if (action.type === "call.llm") {
                    return {
                        ...action,
                        request: {
                            ...action.request,
                            tools: mergeTools(
                                action.request.tools,
                                Object.entries(toolMap).map(([name, def]) => ({
                                    function: { name, description: def.description, parameters: def.parameters },
                                })),
                            ),
                        },
                    };
                }
                if (action.type === "call.tool") {
                    const def = toolMap[action.name];
                    if (def?.retry) {
                        return { ...action, retry: def.retry };
                    }
                }
                return action;
            });

            // Suppress call_llm until all tool results are in
            if (req.trigger.type === "tool.result") {
                const resultId = req.trigger.result.tool_call_id;
                req.state.pendingToolCalls = req.state.pendingToolCalls.filter((id) => id !== resultId);
                if (req.state.pendingToolCalls.length > 0) {
                    return {
                        ...downstream,
                        actions: actions.filter((a) => a.type !== "call.llm"),
                    };
                }
            }

            return { ...downstream, actions };
        },
    });
}

// ── LLM loop ───────────────────────────────────────────────────────────────

export interface LlmLoopSelection {
    request: Omit<LlmRequest, "messages"> & { messages?: Message[] };
    retry?: RetryPolicy;
    stream?: boolean;
    toolRetries?: Record<string, RetryPolicy>;
}

export function llmLoop<S>(
    selectorOrValue: ((state: S, req: AgentRequest<S>) => LlmLoopSelection) | LlmLoopSelection,
): MiddlewareFn<S> {
    const selector: (state: S, req: AgentRequest<S>) => LlmLoopSelection =
        typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;

    return middleware({
        handler: async (req: AgentRequest<S>, next: Next<S>) => {
            const selection = selector(req.state, req);
            const downstream = await next(req);

            const { trigger } = req;
            switch (trigger.type) {
                case "user.message":
                case "client.action":
                case "tool.result": {
                    return {
                        ...downstream,
                        actions: [
                            {
                                type: "call.llm",
                                request: {
                                    ...selection.request,
                                    messages: [],
                                },
                                retry: selection.retry ?? DEFAULT_RETRY,
                                stream: selection.stream ?? false,
                            },
                            ...downstream.actions,
                        ],
                    };
                }
                case "llm.response": {
                    if (!trigger.message.tool_calls || trigger.message.tool_calls.length === 0) {
                        return {
                            ...downstream,
                            actions: [{ type: "done", data: trigger.message.content }, ...downstream.actions],
                        };
                    }
                }

                default:
                    return downstream;
            }
        },
    });
}

// ── Sub-agents ─────────────────────────────────────────────────────────────

export interface SubAgentTrack {
    toolCallId: string;
    name: string;
}

/**
 * Sub-agent delegation middleware.
 * Contributes `{ subAgentTracker: Record<string, SubAgentTrack> }` to state.
 */
export function subAgents<S>(config: {
    agents: Handler[];
    retry?: RetryPolicy;
}): StateContributor<{ subAgentTracker: Record<string, SubAgentTrack> }> &
    MiddlewareFn<unknown, { subAgentTracker: Record<string, SubAgentTrack> }> {
    const subAgentMap: Record<string, { agentId: string }> = {};
    for (const handler of config.agents) {
        subAgentMap[handler.agentId] = { agentId: handler.agentId };
    }

    return middleware({
        state: { subAgentTracker: {} as Record<string, SubAgentTrack> },
        handler: async (req, next) => {
            const tracker = req.state.subAgentTracker;

            switch (req.trigger.type) {
                case "llm.response": {
                    // Intercept tool calls destined for sub-agents directly from
                    // the trigger, so this middleware works with or without
                    // agent.tools() in the chain.
                    const spawnActions: WorkerAction[] = [];
                    const toolCalls = req.trigger.message.tool_calls ?? [];
                    for (const tc of toolCalls) {
                        const sub = subAgentMap[tc.function.name];
                        if (!sub) continue;

                        const childSessionId = crypto.randomUUID();
                        let message = tc.function.arguments;
                        try {
                            const args = JSON.parse(tc.function.arguments);
                            if (typeof args?.message === "string") {
                                message = args.message;
                            }
                        } catch {
                            // no-op
                        }
                        tracker[childSessionId] = {
                            toolCallId: tc.id,
                            name: tc.function.name,
                        };
                        spawnActions.push(
                            {
                                type: "spawn.sub_agent",
                                session_id: childSessionId,
                                agent_id: sub.agentId,
                                retry: config.retry ?? DEFAULT_RETRY,
                            },
                            {
                                type: "send.message",
                                session_id: childSessionId,
                                message: {
                                    role: "user",
                                    content: message,
                                },
                            },
                        );
                    }

                    const downstream = await next(req);

                    // Merge sub-agent tool defs into any call.llm actions, and
                    // filter out call.tool actions that we already spawned above.
                    const actions: WorkerAction[] = [];
                    for (const action of downstream.actions) {
                        if (action.type === "call.tool" && subAgentMap[action.name]) {
                            continue; // already handled via spawn above
                        }
                        if (action.type === "call.llm") {
                            actions.push({
                                ...action,
                                request: {
                                    ...action.request,
                                    tools: mergeTools(action.request.tools, handlersToLlmTools(config.agents)),
                                },
                            });
                            continue;
                        }
                        actions.push(action);
                    }

                    return { ...downstream, actions: [...spawnActions, ...actions] };
                }

                case "sub_agent.turn.complete": {
                    const tracked = tracker[req.trigger.session_id];
                    if (!tracked) {
                        return next(req);
                    }
                    delete tracker[req.trigger.session_id];

                    const content =
                        typeof req.trigger.data === "string" ? req.trigger.data : JSON.stringify(req.trigger.data);
                    const result: ToolResult = {
                        tool_call_id: tracked.toolCallId,
                        name: tracked.name,
                        content,
                        is_error: false,
                    };

                    return appendToolResultToLlmCalls(
                        await next({ ...req, trigger: { type: "tool.result", result } }),
                        result,
                    );
                }

                case "sub_agent.error": {
                    const tracked = tracker[req.trigger.session_id];
                    if (!tracked) {
                        return next(req);
                    }
                    delete tracker[req.trigger.session_id];

                    const result: ToolResult = {
                        tool_call_id: tracked.toolCallId,
                        name: tracked.name,
                        content: `Sub-agent ${req.trigger.agent_id} failed: ${req.trigger.error}`,
                        is_error: true,
                    };

                    return appendToolResultToLlmCalls(
                        await next({ ...req, trigger: { type: "tool.result", result } }),
                        result,
                    );
                }

                default: {
                    const downstream = await next(req);
                    const actions = downstream.actions.map((action) => {
                        if (action.type !== "call.llm") return action;
                        return {
                            ...action,
                            request: {
                                ...action.request,
                                tools: mergeTools(action.request.tools, handlersToLlmTools(config.agents)),
                            },
                        };
                    });
                    return { ...downstream, actions };
                }
            }
        },
    });
}

// ── Helpers ────────────────────────────────────────────────────────────────

function handlersToLlmTools(handlers: Handler[]): LlmTool[] {
    return handlers.map((handler) => ({
        function: {
            name: handler.agentId,
            description: `Delegate to ${handler.agentId}`,
            parameters: {
                type: "object",
                properties: {
                    message: {
                        type: "string",
                        description: "The message to send to the agent",
                    },
                },
                required: ["message"],
            },
        },
    }));
}

/** Append a tool result message to any `call.llm` actions in the response. */
function appendToolResultToLlmCalls(response: AgentResponse, result: ToolResult): AgentResponse {
    const toolMsg: Message = {
        role: "tool",
        content: result.content,
        tool_call_id: result.tool_call_id,
        name: result.name,
    };
    const actions = response.actions.map((action) => {
        if (action.type !== "call.llm") return action;
        return {
            ...action,
            request: {
                ...action.request,
                messages: [...action.request.messages, toolMsg],
            },
        };
    });
    return { ...response, actions };
}

function mergeTools(existing: LlmTool[] | undefined, added: LlmTool[]): LlmTool[] {
    const byName = new Map<string, LlmTool>();
    for (const t of existing ?? []) {
        byName.set(t.function.name, t);
    }
    for (const t of added) {
        byName.set(t.function.name, t);
    }
    return Array.from(byName.values());
}
