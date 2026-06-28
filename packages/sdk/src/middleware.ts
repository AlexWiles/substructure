import type {
    DecisionTrigger,
    LlmHandler,
    LlmRequest,
    LlmResponse,
    LlmTokenDeltaInput,
    LlmTool,
    Message,
    MessageTree,
    ReasoningConfig,
    RetryPolicy,
    StreamPart,
    ToolHandler,
    WorkerAction,
    WorkerDecisionRequestWire,
} from "./types";
import type { Handler, AgentContext, AgentResponse, MiddlewareFn, Next, StateContributor } from "./worker";

export const DEFAULT_RETRY: RetryPolicy = {
    timeout_secs: 120,
    max_retries: 0,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

function decodeWorkerState(raw: string): unknown {
    if (!raw || raw === "") return {};
    return JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(raw), (c) => c.charCodeAt(0))));
}

function encodeWorkerState(value: unknown): string {
    return btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(value))));
}

function initSlice<A extends object>(rawState: unknown, init: A): A {
    const state = (rawState && typeof rawState === "object" ? rawState : {}) as Record<string, unknown>;
    for (const key of Object.keys(init)) {
        state[key] ??= structuredClone((init as Record<string, unknown>)[key]);
    }
    return state as A;
}

/**
 * With `state`: initializes a state slice and returns a `StateContributor`.
 * Without `state`: wraps a handler function.
 */
export function middleware<A extends object, S = unknown>(config: {
    state: A;
    handler?: (ctx: AgentContext<S & A>, next: Next<S & A>) => Promise<AgentResponse> | AgentResponse;
}): StateContributor<A> & MiddlewareFn<S, S & A> & { _init: A };
export function middleware<S = unknown>(config: {
    handler: (ctx: AgentContext<S>, next: Next<S>) => Promise<AgentResponse> | AgentResponse;
}): MiddlewareFn<S>;
export function middleware<A extends object>(config: {
    state?: A;
    handler?: (ctx: AgentContext<A>, next: Next<A>) => Promise<AgentResponse> | AgentResponse;
}): MiddlewareFn<unknown> {
    const handler = config.handler ?? ((ctx: AgentContext<A>, next: Next<A>) => next(ctx));

    if (!config.state) {
        return handler as MiddlewareFn<unknown>;
    }

    const init = config.state;
    const mw = (ctx: AgentContext<unknown>, next: Next<A>) => {
        const state = initSlice(ctx.state, init);
        const typedCtx: AgentContext<A> = { ...ctx, state: state as A };
        return handler(typedCtx, next);
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

/** Round-trips base64 JSON state between `request.worker_state` and `workerState`. */
export function jsonState(): MiddlewareFn<unknown> {
    return middleware({
        handler: async (ctx: AgentContext<unknown>, next: Next<unknown>) => {
            const enriched: AgentContext<unknown> = {
                ...ctx,
                state: decodeWorkerState(ctx.request.worker_state),
            };
            const res = await next(enriched);
            return {
                ...res,
                workerState: encodeWorkerState(res.state),
            };
        },
    });
}

/**
 * Sentinel from `ctx.defer()`: the worker emits no result for this `tool.execute`,
 * and the engine leaves the call pending until `submitToolCallResult` is called.
 * Tools call `ctx.defer()` rather than importing this.
 */
export const DEFERRED: unique symbol = Symbol.for("substructure.tool.deferred");
export type Deferred = typeof DEFERRED;

export interface ToolExecutionContext {
    sessionId: string;
    toolCallId: string;
    attempt: number;
    request: WorkerDecisionRequestWire;
    /** Signal out-of-band completion: `return ctx.defer();`. */
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
    /** "worker" (default) runs `execute` on the worker; "client" routes the call
     *  to the frontend, which completes it via `submitToolCallResult` — the
     *  worker never runs `execute` for a client tool. */
    handler?: ToolHandler;
}

/**
 * Recover a slice's contributed state shape from the slice value itself.
 *
 * We infer the whole `StateSliceMw` type as one blob and read its `_init`
 * field, rather than inferring the slice's `S` directly: `StateContributor`
 * is branded `MiddlewareFn<any, any>`, so inferring `S` structurally picks up
 * that `any` (via `_out` / the call signature) and collapses `state` to `any`.
 * `_init` is the one field the `any` never reaches.
 */
type SliceState<Slice> = Slice extends { _init: infer A } ? A : never;

type ToolResult = string | Deferred | Promise<string | Deferred>;

// Tools return the result string directly (no implicit serialization — call
// `JSON.stringify` yourself for structured data), or `ctx.defer()` to complete
// the call out-of-band. A `state` slice gives the executor typed access to it.
//
// `handler` discriminates how the tool runs:
//   - "worker" (default): the engine runs `execute`, so it's required.
//   - "client": the call is completed in the browser, so `execute` is optional
//     and never runs on the worker.
export function tool<Slice extends StateSliceMw<object> = never>(
    config: {
        name: string;
        description: string;
        parameters: unknown;
        state?: Slice;
        retry?: RetryPolicy;
    } & (
        | {
              handler?: "worker";
              execute: [Slice] extends [never]
                  ? (args: string, ctx: ToolExecutionContext) => ToolResult
                  : (args: string, state: SliceState<Slice>, ctx: ToolExecutionContext) => ToolResult;
          }
        | { handler: "client"; execute?: (args: string, ctx: ToolExecutionContext) => ToolResult }
    ),
): ToolDef {
    return {
        name: config.name,
        description: config.description,
        parameters: config.parameters,
        execute: async (args: string, state?: unknown, ctx?: ToolExecutionContext) => {
            if (!config.execute) {
                // Client tools are completed by the frontend; reaching here means
                // a worker tool was declared without an executor.
                throw new Error(`Tool "${config.name}" has no server-side execute.`);
            }
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
 * Pair with `actions()` to dispatch by name. `args` is cast to `Args` (no runtime
 * validation). A `void` return continues the chain; a `WorkerAction[]` short-circuits.
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

/** Dispatch `client.action` triggers to a matching `ActionDef` by name; others pass through. */
export function actions(defs: ActionDef<any, any>[]): MiddlewareFn<unknown> {
    const byName: Record<string, ActionDef<any, any>> = {};
    for (const def of defs) byName[def.name] = def;

    return middleware({
        handler: async (ctx, next) => {
            const { trigger } = ctx.request;
            if (trigger.type !== "client.action") return next(ctx);

            const def = byName[trigger.name];
            if (!def) return next(ctx);

            const state = def.stateSlice ? initSlice(ctx.state, def.stateSlice._init) : ctx.state;
            const result = await def.handler(trigger.args ?? {}, state);

            if (Array.isArray(result)) {
                return { state: ctx.state, actions: result };
            }
            return next(ctx);
        },
    });
}

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
        handler: async (ctx, next) => {
            const t = ctx.request.trigger;
            const tag = t.type === "tool.execute" ? `${t.type}:${t.name}` : t.type;
            const fields: Record<string, unknown> = {
                agent: ctx.request.agent_id,
                session: ctx.request.session_id,
                decision: ctx.request.decision_id,
                trigger: tag,
            };
            if (label) fields.label = label;

            log.info("decision.start", fields);
            log.debug("decision.trigger", { ...fields, payload: t });

            const start = performance.now();
            try {
                const result = await next(ctx);
                const durationMs = Number((performance.now() - start).toFixed(1));
                const actionTypes = result.actions.map((a) => a.type);

                log.info("decision.end", { ...fields, actions: actionTypes, durationMs });
                log.debug("decision.actions", { ...fields, actions: result.actions, durationMs });

                return result;
            } catch (err) {
                const durationMs = Number((performance.now() - start).toFixed(1));
                log.error("decision.error", {
                    ...fields,
                    error: err instanceof Error ? err.message : String(err),
                    durationMs,
                });
                throw err;
            }
        },
    });
}

export type MessageSelector<S> = (state: S, ctx: AgentContext<S>) => Message[] | Promise<Message[]>;

export type SystemMessageSelector<S> = (state: S, ctx: AgentContext<S>) => string | Promise<string>;

/** The transcript message for a trigger, or `null` for triggers with no turn entry
 *  (`client.action`, `tool.execute`, `llm.error`, …). */
export function triggerToMessage(trigger: DecisionTrigger): Message | null {
    switch (trigger.type) {
        case "user.message":
        case "llm.response":
            return trigger.message;
        default:
            return null;
    }
}

/** Like `triggerToMessage`, but expands a `tool.results` batch into its messages. */
export function triggerToMessages(trigger: DecisionTrigger): Message[] {
    if (trigger.type === "tool.results") {
        return trigger.messages.map((n) => n.message);
    }
    const msg = triggerToMessage(trigger);
    return msg ? [msg] : [];
}

/** Prepend a transcript to every `call.llm` action's `messages`; other actions pass through. */
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

/** The active transcript: the `head_id`-to-root path of the engine-owned tree. */
export function activePath(tree?: MessageTree): Message[] {
    if (!tree) return [];
    const out: Message[] = [];
    let target = tree.head_id;
    // Parents are appended before their children, so one reverse pass collects
    // the head-to-root chain (and stops at a missing parent).
    for (let i = tree.nodes.length - 1; i >= 0 && target != null; i--) {
        if (tree.nodes[i].id === target) {
            out.push(tree.nodes[i].message);
            target = tree.nodes[i].parent_id;
        }
    }
    return out.reverse();
}

// ── Tools ──────────────────────────────────────────────────────────────────

export type ToolInput = Record<string, ToolDef> | ToolDef[];
export type ToolSelector<S> = (state: S, ctx: AgentContext<S>) => ToolInput | Promise<ToolInput>;

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
 * Tool middleware: executes `tool.execute` triggers, turns `llm.response` tool
 * calls into `call.tool` actions, and merges tool definitions into every `call.llm`.
 */
export function tools<S>(selectorOrValue: ToolSelector<S> | ToolInput): MiddlewareFn<S, S> {
    const selector: ToolSelector<S> = typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;

    return middleware({
        handler: async (ctx: AgentContext<S>, next: Next<S>) => {
            const toolMap = resolveTools(await selector(ctx.state, ctx));

            const downstream = await next(ctx);

            if (ctx.request.trigger.type === "tool.execute") {
                const t = toolMap[ctx.request.trigger.name];
                if (!t) {
                    return {
                        ...downstream,
                        actions: [
                            {
                                type: "return.tool.error" as const,
                                tool_call_id: ctx.request.trigger.tool_call_id,
                                error: `Unknown tool: ${ctx.request.trigger.name}`,
                                retryable: false,
                                attempt: ctx.request.trigger.attempt,
                            },
                            ...downstream.actions,
                        ],
                    };
                }

                try {
                    let toolState: unknown;
                    if (t.stateSlice) {
                        toolState = initSlice(ctx.state, t.stateSlice._init);
                    }
                    const toolCtx: ToolExecutionContext = {
                        sessionId: ctx.request.session_id,
                        toolCallId: ctx.request.trigger.tool_call_id,
                        attempt: ctx.request.trigger.attempt,
                        request: ctx.request,
                        defer: () => DEFERRED,
                    };
                    const output = await t.execute(ctx.request.trigger.arguments, toolState, toolCtx);
                    if (output === DEFERRED) {
                        return downstream;
                    }
                    return {
                        ...downstream,
                        actions: [
                            {
                                type: "return.tool.result" as const,
                                tool_call_id: ctx.request.trigger.tool_call_id,
                                // Tools are typed to return a string; guard at runtime so an
                                // untyped caller can't drop the field and make the engine reject
                                // the action ("missing field `result`").
                                result: typeof output === "string" ? output : "",
                                attempt: ctx.request.trigger.attempt,
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
                                tool_call_id: ctx.request.trigger.tool_call_id,
                                error: error instanceof Error ? error.message : String(error),
                                retryable: false,
                                attempt: ctx.request.trigger.attempt,
                            },
                            ...downstream.actions,
                        ],
                    };
                }
            }

            // Only emit `call.tool` for tools this middleware knows about.
            if (ctx.request.trigger.type === "llm.response") {
                const toolCalls = ctx.request.trigger.message.tool_calls;
                if (toolCalls && toolCalls.length > 0) {
                    const known = toolCalls.filter((tc) => tc.function.name in toolMap);

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

            return { ...downstream, actions };
        },
    });
}

// ── LLM loop ───────────────────────────────────────────────────────────────

export interface LlmSelection {
    /** A worker-side provider generator (e.g. `anthropicGenerate`) or `serverGenerate`. */
    generator: LlmGenerator;
    /** System prompt prepended to the transcript; a function is resolved per call. */
    instructions?: string | (() => string | Promise<string>);
    /** Override the generator's streaming setting. */
    stream?: boolean;
    retry?: RetryPolicy;
}

/**
 * One worker-side LLM call, used as a worker generator's `run`. `request.tools` is
 * already merged in by `tools()`; stream deltas via `ctx.emitDelta`.
 */
export type LlmGenerate = (
    request: LlmRequest,
    ctx: { emitDelta?: (part: StreamPart) => Promise<void> },
) => Promise<LlmResponse>;

/**
 * A bound LLM backend for `llm`. A worker generator supplies `run` (the call
 * runs on your worker); a server generator omits it (the Substructure server calls
 * its configured provider).
 */
export interface LlmGenerator {
    request: Omit<LlmRequest, "messages" | "tools">;
    handler: LlmHandler;
    stream?: boolean;
    run?: LlmGenerate;
}

/** `model` is the server's provider model id, e.g. "anthropic/claude-sonnet-4-6". */
export function serverGenerate(settings: {
    model: string;
    temperature?: number;
    maxTokens?: number;
    reasoning?: ReasoningConfig;
    stream?: boolean;
}): LlmGenerator {
    const request: Omit<LlmRequest, "messages" | "tools"> = { model: settings.model };
    if (settings.temperature !== undefined) request.temperature = settings.temperature;
    if (settings.maxTokens !== undefined) request.max_completion_tokens = settings.maxTokens;
    if (settings.reasoning !== undefined) request.reasoning = settings.reasoning;
    return { request, handler: "server", stream: settings.stream ?? false };
}

function flattenStreamPart(part: StreamPart): LlmTokenDeltaInput | null {
    switch (part.type) {
        case "text-delta":
            return { text: part.delta };
        case "reasoning-delta":
            return { reasoning: part.delta };
        case "tool-input-start":
            return { tool_calls: [{ id: part.toolCallId, name: part.toolName }] };
        case "tool-input-delta":
            return { tool_calls: [{ id: part.toolCallId, arguments: part.inputTextDelta }] };
        case "finish":
            return part.finishReason ? { finish_reason: part.finishReason } : null;
        default:
            return null;
    }
}

async function runWorkerLlmCall(
    trigger: Extract<DecisionTrigger, { type: "llm.request" }>,
    generate: LlmGenerate | undefined,
    emitDelta?: (delta: LlmTokenDeltaInput) => Promise<void>,
): Promise<WorkerAction> {
    if (!generate) {
        return {
            type: "return.llm.error",
            call_id: trigger.call_id,
            error: 'llm received an "llm.request" trigger but the generator has no `run` for worker-handled LLM calls',
            retryable: false,
            attempt: trigger.attempt,
        };
    }
    const emitPart = emitDelta
        ? async (part: StreamPart) => {
              const flat = flattenStreamPart(part);
              if (flat) await emitDelta(flat);
          }
        : undefined;
    try {
        const response = await generate(trigger.request, { emitDelta: emitPart });
        return {
            type: "return.llm.result",
            call_id: trigger.call_id,
            response,
            attempt: trigger.attempt,
        };
    } catch (error) {
        return {
            type: "return.llm.error",
            call_id: trigger.call_id,
            error: error instanceof Error ? error.message : String(error),
            retryable: false,
            attempt: trigger.attempt,
        };
    }
}

/**
 * The loop body: on a turn-continuing trigger emit a `call.llm` built from the
 * tree's active path (plus `instructions`); on `llm.request` run the worker-side
 * call; on an `llm.response` with no tool calls emit `done`. Compose with
 * `stopWhen` to cap or otherwise halt the loop.
 */
export function llm<S>(
    selectorOrValue: ((state: S, ctx: AgentContext<S>) => LlmSelection | Promise<LlmSelection>) | LlmSelection,
): MiddlewareFn<S, S> {
    const selector: (state: S, ctx: AgentContext<S>) => LlmSelection | Promise<LlmSelection> =
        typeof selectorOrValue === "function" ? selectorOrValue : () => selectorOrValue;

    return middleware({
        handler: async (ctx: AgentContext<S>, next: Next<S>) => {
            const selection = await selector(ctx.state, ctx);
            const generator = selection.generator;
            const stream = selection.stream ?? generator.stream ?? false;
            const downstream = await next(ctx);

            const { trigger } = ctx.request;
            switch (trigger.type) {
                case "user.message":
                case "client.action":
                case "tool.results": {
                    const active = activePath(ctx.request.message_tree);
                    const instructions =
                        typeof selection.instructions === "function"
                            ? await selection.instructions()
                            : selection.instructions;
                    const messages = instructions
                        ? [{ role: "system", content: instructions } as Message, ...active]
                        : active;
                    return {
                        ...downstream,
                        actions: [
                            {
                                type: "call.llm",
                                request: {
                                    ...generator.request,
                                    messages,
                                },
                                retry: selection.retry ?? DEFAULT_RETRY,
                                stream,
                                handler: generator.handler,
                            },
                            ...downstream.actions,
                        ],
                    };
                }
                case "llm.request": {
                    const action = await runWorkerLlmCall(trigger, generator.run, ctx.emitDelta);
                    return { ...downstream, actions: [action, ...downstream.actions] };
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

// ── Loop control ─────────────────────────────────────────────────────────────

export interface StopInfo<S = unknown> {
    /** Assistant rounds taken since the last user message. */
    steps: number;
    /** The assistant message whose tool calls just ran. */
    lastResponse: Message | undefined;
    tree: MessageTree;
    ctx: AgentContext<S>;
}

export type StopCondition<S = unknown> = (info: StopInfo<S>) => boolean | Promise<boolean>;

function stopInfo<S>(ctx: AgentContext<S>): StopInfo<S> {
    const tree = ctx.request.message_tree ?? { nodes: [] };
    const path = activePath(tree);
    let steps = 0;
    let lastResponse: Message | undefined;
    for (let i = path.length - 1; i >= 0; i--) {
        if (path[i].role === "user") break;
        if (path[i].role === "assistant") {
            steps++;
            lastResponse ??= path[i];
        }
    }
    return { steps, lastResponse, tree, ctx };
}

/**
 * Guards the loop's back-edge: on `tool.results`, if a continuation `call.llm` is
 * still pending and `cond` holds, replace it with `done`. No-ops when nothing is
 * continuing, so chaining several `stopWhen`s gives OR with no duplicate `done`.
 */
export function stopWhen<S>(cond: StopCondition<S>): MiddlewareFn<S, S> {
    return middleware({
        handler: async (ctx: AgentContext<S>, next: Next<S>) => {
            const down = await next(ctx);
            if (ctx.request.trigger.type !== "tool.results") return down;
            if (!down.actions.some((a) => a.type === "call.llm")) return down;

            const info = stopInfo(ctx);
            if (!(await cond(info))) return down;

            return {
                ...down,
                actions: [
                    { type: "done", data: info.lastResponse?.content ?? "" },
                    ...down.actions.filter((a) => a.type !== "call.llm"),
                ],
            };
        },
    });
}

/** Stop once the turn has taken `n` assistant rounds. */
export const stepCountIs =
    (n: number): StopCondition =>
    (info) =>
        info.steps >= n;

/**
 * Presents each sub-agent to the model as a tool and turns a call into a
 * `spawn.sub_agent` child session. Results return via `tool.results`, so
 * this middleware keeps no state.
 */
export function subAgents<S>(config: { agents: Handler[]; retry?: RetryPolicy }): MiddlewareFn<S, S> {
    const subAgentMap: Record<string, { agentId: string }> = {};
    for (const handler of config.agents) {
        subAgentMap[handler.agentId] = { agentId: handler.agentId };
    }

    const mergeSubAgentTools = (actions: WorkerAction[]): WorkerAction[] =>
        actions.map((action) =>
            action.type === "call.llm"
                ? {
                      ...action,
                      request: {
                          ...action.request,
                          tools: mergeTools(action.request.tools, handlersToLlmTools(config.agents)),
                      },
                  }
                : action,
        );

    return middleware({
        handler: async (ctx: AgentContext<S>, next: Next<S>) => {
            if (ctx.request.trigger.type !== "llm.response") {
                const downstream = await next(ctx);
                return { ...downstream, actions: mergeSubAgentTools(downstream.actions) };
            }

            // Read from the trigger so this works with or without tools() in the chain.
            const spawnActions: WorkerAction[] = [];
            for (const tc of ctx.request.trigger.message.tool_calls ?? []) {
                const sub = subAgentMap[tc.function.name];
                if (!sub) continue;

                const childSessionId = crypto.randomUUID();
                let message = tc.function.arguments;
                try {
                    const args = JSON.parse(tc.function.arguments);
                    if (typeof args?.message === "string") message = args.message;
                } catch {
                    // leave raw arguments as the message
                }
                spawnActions.push(
                    {
                        type: "spawn.sub_agent",
                        session_id: childSessionId,
                        agent_id: sub.agentId,
                        tool_call_id: tc.id,
                        retry: config.retry ?? DEFAULT_RETRY,
                    },
                    {
                        type: "send.message",
                        session_id: childSessionId,
                        message: { role: "user", content: message },
                    },
                );
            }

            const downstream = await next(ctx);
            // Sub-agent names are spawned, not run as tools.
            const actions = mergeSubAgentTools(
                downstream.actions.filter((a) => !(a.type === "call.tool" && subAgentMap[a.name])),
            );
            return { ...downstream, actions: [...spawnActions, ...actions] };
        },
    });
}

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
