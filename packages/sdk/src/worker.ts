import type {
    WorkerDecisionRequestWire,
    WorkerAction,
    SubmitRequest,
    SpanContext,
    LlmTool,
    Message,
    RetryPolicy,
    Uuid,
} from "./types";
import { contentText } from "./types";
import type { NativeRuntime } from "./runtime";
import type { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// ── Handler types ────────────────────────────────────────────────────────────

export interface DecisionResult {
    actions: WorkerAction[];
    state: string;
}

export type DecisionHandler = (
    request: WorkerDecisionRequestWire
) => Promise<DecisionResult>;

export interface HandlerContext<S = unknown> {
    agentId: string;
    trigger: WorkerDecisionRequestWire["trigger"];
    state: S;
    request: WorkerDecisionRequestWire;
    /** @internal — set by state serialization middleware */
    _encodedState?: string;
}

export interface HandlerResult {
    actions: WorkerAction[];
}

/** Terminal handler: receives context with state S, returns actions. */
export type Next<S = unknown> = (
    ctx: HandlerContext<S>,
) => Promise<HandlerResult> | HandlerResult;

/**
 * Middleware function. Receives the current context and a `next` callback.
 * The Out type parameter determines what state type downstream sees.
 *
 * Most middleware is passthrough (In=Out). State providers transform the type.
 */
export interface MiddlewareFn<In = unknown, Out = In> {
    (ctx: HandlerContext<In>, next: Next<Out>): Promise<HandlerResult> | HandlerResult;
    /** Type brand carrying the output state type — never set at runtime. */
    readonly _out?: Out;
}

// ── Handler ─────────────────────────────────────────────────────────────────

export interface Handler {
    /** Agent ids collected from .use(agent) calls. Empty for raw handlers. */
    readonly agentIds: string[];
    toDecisionHandler(): DecisionHandler;
}

// ── State ───────────────────────────────────────────────────────────────────

/**
 * Middleware that contributes state keys. The `_contributes` brand
 * tells the builder to intersect `A` onto the current state type.
 */
export type StateContributor<A> = MiddlewareFn<any, any> & { readonly _contributes: A };

/**
 * JSON + base64 state serialization middleware.
 * Decodes `request.worker_state` into `ctx.state`, runs the chain,
 * then encodes `ctx.state` back.
 */
export function withJsonState(): MiddlewareFn<any, any> {
    return async (ctx, next) => {
        const raw = ctx.request.worker_state;
        ctx.state = (raw && raw !== "")
            ? JSON.parse(Buffer.from(raw, "base64").toString("utf-8"))
            : {};
        const result = await next(ctx);
        ctx._encodedState = Buffer.from(JSON.stringify(ctx.state), "utf-8").toString("base64");
        return result;
    };
}

/**
 * Explicit state initializer. Merges default keys onto the state object.
 * Use this for custom state that isn't provided by a middleware.
 */
export function withState<S extends object>(init: () => S): StateContributor<S> {
    const mw: MiddlewareFn<any, any> = (ctx, next) => {
        if (ctx.state == null) ctx.state = {};
        Object.assign(ctx.state, init());
        return next(ctx);
    };
    return Object.assign(mw, { _contributes: undefined as unknown as S });
}

// ── withMessages ────────────────────────────────────────────────────────────

export interface HasMessages {
    messages: Message[];
}

export function withMessages(): StateContributor<HasMessages> {
    const mw: MiddlewareFn<any, any> = (ctx, next) => {
        if (!ctx.state.messages) ctx.state.messages = [];
        const { trigger, state } = ctx as HandlerContext<HasMessages>;
        if (trigger.type === "user_message") {
            state.messages.push(trigger.message);
        } else if (trigger.type === "llm_response") {
            state.messages.push(trigger.message);
        } else if (trigger.type === "tool_result") {
            state.messages.push({
                role: "tool",
                content: trigger.result.content,
                tool_call_id: trigger.result.tool_call_id,
                name: trigger.result.name,
            });
        }
        return next(ctx);
    };
    return Object.assign(mw, { _contributes: undefined as unknown as HasMessages });
}

// ── Tool types ──────────────────────────────────────────────────────────────

export type ToolFn = (args: Record<string, unknown>) => Promise<unknown> | unknown;

const DEFAULT_TOOL_RETRY: RetryPolicy = {
    timeout_secs: 30,
    max_retries: 0,
    backoff_base_secs: 0,
    backoff_max_secs: 0,
};

export interface ToolDef {
    description: string;
    parameters: unknown;
    execute: ToolFn;
    retry?: RetryPolicy;
    /** @internal — set by subAgent() */
    _agentId?: string;
}

function isSubAgentTool(t: ToolDef): boolean {
    return t._agentId !== undefined;
}

/** Define a tool with a zod schema for type-safe parameters. */
export function tool<T extends z.ZodType<any, any>>(config: {
    description: string;
    parameters: T;
    execute: (args: z.infer<T>) => unknown | Promise<unknown>;
    retry?: RetryPolicy;
}): ToolDef {
    return {
        description: config.description,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        parameters: zodToJsonSchema(config.parameters as any, { target: "openAi" }),
        execute: config.execute as ToolFn,
        retry: config.retry,
    };
}

/** Create a tool definition backed by a sub-agent. */
export function subAgent(binding: { id: string }, description: string): ToolDef {
    return {
        description,
        parameters: {
            type: "object",
            properties: {
                message: { type: "string", description: "The message to send to the agent" },
            },
            required: ["message"],
        },
        execute: () => { throw new Error("sub-agent tools are not directly executable"); },
        _agentId: binding.id,
    };
}

// ── withAgentLoop ───────────────────────────────────────────────────────────

export interface AgentLoopUsage {
    /** Prompt tokens from the most recent LLM call — the current context size. */
    promptTokens: number;
    /** Total completion tokens across all calls. */
    completionTokens: number;
    /** Number of LLM calls. */
    calls: number;
    /** Cumulative cost. */
    totalCost: number;
    /** Number of times compaction has been triggered. */
    compactionCount: number;
    /** True when waiting for a summarization response. */
    isCompacting: boolean;
}

export interface PendingSubAgent {
    toolCallId: string;
    name: string;
}

export interface HasAgentLoop {
    messages: Message[];
    agentLoop: AgentLoopUsage;
    pendingSubAgents: Record<string, PendingSubAgent>;
}

export interface CompactionConfig {
    /** Compact when prompt tokens exceed this threshold. */
    maxPromptTokens: number;
    /** Model to use for summarization (can be a cheaper model). */
    model?: string;
    /** LLM client for summarization. Falls back to main client. */
    client?: string;
}

export interface AgentLoopOptions {
    model: string;
    client: string;
    system?: string;
    retry: RetryPolicy;
    tools?: Record<string, ToolDef>;
    stream?: boolean;
    compaction?: CompactionConfig;
}

export function withAgentLoop(options: AgentLoopOptions): StateContributor<HasAgentLoop> {
    const { client, retry } = options;
    const stream = options.stream ?? false;

    const toolDefs: LlmTool[] = options.tools
        ? Object.entries(options.tools).map(([name, t]) => ({
            function: { name, description: t.description, parameters: t.parameters },
        }))
        : [];

    function buildCallLlm(state: HasAgentLoop): WorkerAction {
        const messages: Message[] = [];
        if (options.system) {
            messages.push({ role: "system", content: options.system });
        }
        messages.push(...state.messages);
        return {
            type: "call_llm",
            request: {
                model: options.model,
                messages,
                ...(toolDefs.length ? { tools: toolDefs } : {}),
            },
            stream,
            llm_client: client,
            retry,
        };
    }

    function buildSummarizeCall(state: HasAgentLoop, compaction: CompactionConfig): WorkerAction {
        return {
            type: "call_llm",
            request: {
                model: compaction.model ?? options.model,
                messages: [
                    {
                        role: "system",
                        content: "Summarize the following conversation concisely. Preserve key facts, decisions, and tool results. Output only the summary.",
                    },
                    ...state.messages,
                ],
            },
            stream: false,
            llm_client: compaction.client ?? client,
            retry,
        };
    }

    function shouldCompact(state: HasAgentLoop): boolean {
        return !!(options.compaction && state.agentLoop.promptTokens > options.compaction.maxPromptTokens);
    }

    function callLlmOrCompact(state: HasAgentLoop): HandlerResult {
        if (shouldCompact(state)) {
            state.agentLoop.isCompacting = true;
            return { actions: [buildSummarizeCall(state, options.compaction!)] };
        }
        return { actions: [buildCallLlm(state)] };
    }

    const mw: MiddlewareFn<any, any> = async (ctx, next) => {
        const { trigger } = ctx;
        const state = ctx.state as HasAgentLoop;

        if (!state.messages) state.messages = [];
        if (!state.pendingSubAgents) state.pendingSubAgents = {};
        if (!state.agentLoop) {
            state.agentLoop = { promptTokens: 0, completionTokens: 0, calls: 0, totalCost: 0, compactionCount: 0, isCompacting: false };
        }

        // ── user_message: track + call LLM (or compact) ──
        if (trigger.type === "user_message") {
            state.messages.push(trigger.message);
            return callLlmOrCompact(state);
        }

        // ── tool_result: track + call LLM (or compact) ──
        if (trigger.type === "tool_result") {
            state.messages.push({
                role: "tool",
                content: trigger.result.content,
                tool_call_id: trigger.result.tool_call_id,
                name: trigger.result.name,
            });
            return callLlmOrCompact(state);
        }

        // ── tool_execute: run the handler ──
        if (trigger.type === "tool_execute" && options.tools) {
            const tool = options.tools[trigger.name];
            if (tool && !isSubAgentTool(tool)) {
                try {
                    const args = JSON.parse(trigger.arguments);
                    const result = await tool.execute(args);
                    return {
                        actions: [{
                            type: "return_tool_result" as const,
                            tool_call_id: trigger.tool_call_id,
                            result: typeof result === "string" ? result : JSON.stringify(result),
                            attempt: trigger.attempt,
                        }],
                    };
                } catch (e: any) {
                    return {
                        actions: [{
                            type: "return_tool_error" as const,
                            tool_call_id: trigger.tool_call_id,
                            error: e.message,
                            retryable: false,
                            attempt: trigger.attempt,
                        }],
                    };
                }
            }
        }

        // ── sub_agent_turn_complete: inject result as tool message ──
        if (trigger.type === "sub_agent_turn_complete") {
            const pending = state.pendingSubAgents[trigger.session_id];
            if (pending) {
                delete state.pendingSubAgents[trigger.session_id];
                const parts = trigger.artifacts.flatMap((a: any) => a.parts);
                const text = parts
                    .map((p: any) => (p.kind === "text" ? p.text : JSON.stringify(p.data)))
                    .join("\n");
                state.messages.push({
                    role: "tool",
                    content: text || `Sub-agent ${trigger.agent_id} completed with no output.`,
                    tool_call_id: pending.toolCallId,
                    name: pending.name,
                });
                return callLlmOrCompact(state);
            }
        }

        // ── sub_agent_error: inject error as tool message ──
        if (trigger.type === "sub_agent_error") {
            const pending = state.pendingSubAgents[trigger.session_id];
            if (pending) {
                delete state.pendingSubAgents[trigger.session_id];
                state.messages.push({
                    role: "tool",
                    content: `Sub-agent ${trigger.agent_id} failed: ${trigger.error}`,
                    tool_call_id: pending.toolCallId,
                    name: pending.name,
                });
                return callLlmOrCompact(state);
            }
        }

        // ── llm_error: done with error ──
        if (trigger.type === "llm_error") {
            return {
                actions: [{
                    type: "done" as const,
                    artifacts: [{ parts: [{ kind: "text" as const, text: `LLM error: ${trigger.error}` }] }],
                }],
            };
        }

        // ── llm_response: track usage, dispatch tools or compact or done ──
        if (trigger.type === "llm_response") {
            state.agentLoop.calls++;
            if (trigger.usage) {
                const u = trigger.usage as Record<string, number>;
                state.agentLoop.promptTokens = Number(u.prompt_tokens ?? 0);
                state.agentLoop.completionTokens += Number(u.completion_tokens ?? 0);
            }
            if (trigger.cost) {
                state.agentLoop.totalCost += Number(trigger.cost);
            }

            // Summary response — replace messages and emit the real LLM call
            if (state.agentLoop.isCompacting) {
                const summaryText = contentText(trigger.message.content);
                const lastUserMsg = [...state.messages].reverse().find(m => m.role === "user");
                state.messages = [
                    { role: "system", content: `Previous conversation summary:\n${summaryText}` },
                    ...(lastUserMsg ? [lastUserMsg] : []),
                ];
                state.agentLoop.isCompacting = false;
                state.agentLoop.compactionCount++;
                return { actions: [buildCallLlm(state)] };
            }

            // Tool calls — track message + dispatch (tools or sub-agents)
            if (trigger.message.tool_calls?.length) {
                state.messages.push(trigger.message);
                const actions: WorkerAction[] = [];

                for (const tc of trigger.message.tool_calls) {
                    const toolDef = options.tools?.[tc.function.name];
                    if (toolDef && isSubAgentTool(toolDef)) {
                        const childSessionId = crypto.randomUUID() as Uuid;
                        const args = JSON.parse(tc.function.arguments);
                        state.pendingSubAgents[childSessionId] = {
                            toolCallId: tc.id,
                            name: tc.function.name,
                        };
                        actions.push(
                            {
                                type: "spawn_sub_agent",
                                session_id: childSessionId,
                                agent_id: toolDef._agentId!,
                                retry: toolDef.retry ?? retry,
                            },
                            {
                                type: "send_message",
                                session_id: childSessionId,
                                message: { role: "user", content: args.message },
                            },
                        );
                    } else {
                        actions.push({
                            type: "call_tool" as const,
                            tool_call_id: tc.id,
                            name: tc.function.name,
                            arguments: tc.function.arguments,
                            handler: "worker" as const,
                            retry: toolDef?.retry ?? DEFAULT_TOOL_RETRY,
                        });
                    }
                }

                return { actions };
            }

            // Final response — track message + done
            state.messages.push(trigger.message);
            const last = state.messages.at(-1);
            const lastText = contentText(last?.content);
            return {
                actions: [{
                    type: "done" as const,
                    artifacts: lastText
                        ? [{ parts: [{ kind: "text" as const, text: lastText }] }]
                        : [],
                }],
            };
        }

        return next(ctx);
    };

    return Object.assign(mw, { _contributes: undefined as unknown as HasAgentLoop });
}

// ── withLogging ─────────────────────────────────────────────────────────────

export function withLogging(label?: string): MiddlewareFn<any, any> {
    const prefix = label ? `[${label}]` : "[handler]";
    return async (ctx, next) => {
        const t = ctx.trigger;
        const tag = t.type === "tool_execute" ? `${t.type}:${t.name}` : t.type;
        console.log(`${prefix} ${tag}`);
        const start = performance.now();
        const result = await next(ctx);
        const ms = (performance.now() - start).toFixed(1);
        console.log(`${prefix} ${tag} -> ${result.actions.map((a) => a.type).join(", ")} (${ms}ms)`);
        return result;
    };
}

// ── Composable ──────────────────────────────────────────────────────────────

/** Object that can be used as middleware via .use() (e.g. Agent). */
export interface Composable<A = any> {
    readonly id: string;
    toMiddleware(): StateContributor<A>;
}

// ── HandlerBuilder ──────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyMiddleware = MiddlewareFn<any, any>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNext = Next<any>;

/** Extract the output state type from a middleware. */
type MiddlewareOut<M> = M extends MiddlewareFn<any, infer Out> ? Out : never;

/** Detect the `any` type. */
type IsAny<T> = 0 extends (1 & T) ? true : false;

const DEFAULT_FALLBACK: AnyNext = () => ({ actions: [] });

class HandlerBuilder<S> implements Handler {
    private middlewares: AnyMiddleware[] = [];
    private _agentIds: string[] = [];

    get agentIds(): string[] {
        return this._agentIds;
    }

    /** Composable object (e.g. Agent): extracts middleware and intersects its state. */
    use<A>(mw: Composable<A>): HandlerBuilder<S & A>;
    /** State contributor: intersects new keys onto current state. */
    use<A>(mw: StateContributor<A>): HandlerBuilder<S & A>;
    /** State transformer: replaces state type (e.g. withState). */
    use<M extends MiddlewareFn<S, any>>(mw: M): HandlerBuilder<IsAny<MiddlewareOut<M>> extends true ? S : MiddlewareOut<M>>;
    use(mw: AnyMiddleware | Composable): HandlerBuilder<any> {
        if (typeof mw !== "function" && "toMiddleware" in mw) {
            this._agentIds.push(mw.id);
            this.middlewares.push(mw.toMiddleware());
        } else {
            this.middlewares.push(mw as AnyMiddleware);
        }
        return this as any;
    }

    /** Set a custom terminal handler. Returns a plain Handler. */
    handle(fn: Next<S>): Handler;
    handle(): Handler;
    handle(fn?: Next<S>): Handler {
        return this.build((fn as AnyNext) ?? DEFAULT_FALLBACK);
    }

    toDecisionHandler(): DecisionHandler {
        return this.build(DEFAULT_FALLBACK).toDecisionHandler();
    }

    private build(fn: AnyNext): Handler {
        const agentIds = this._agentIds;
        const middlewares = this.middlewares;
        const chain = composeChain(middlewares, fn);

        return {
            agentIds,
            toDecisionHandler(): DecisionHandler {
                return async (request: WorkerDecisionRequestWire) => {
                    const ctx: HandlerContext<unknown> = {
                        agentId: request.agent_id,
                        trigger: request.trigger,
                        state: undefined,
                        request,
                    };
                    const result = await chain(ctx);
                    return {
                        actions: result.actions,
                        state: ctx._encodedState ?? "",
                    };
                };
            },
        };
    }
}

function composeChain(middlewares: AnyMiddleware[], handle: AnyNext): AnyNext {
    let fn: AnyNext = handle;
    for (let i = middlewares.length - 1; i >= 0; i--) {
        const mw = middlewares[i];
        const next = fn;
        fn = (ctx) => mw(ctx, next);
    }
    return fn;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export function defineHandler(): HandlerBuilder<{}> {
    return new HandlerBuilder();
}

// ── Worker (internal) ───────────────────────────────────────────────────────

export class Worker {
    readonly agentIds: string[];
    private handler: DecisionHandler;

    constructor(handler: Handler) {
        this.agentIds = handler.agentIds;
        this.handler = handler.toDecisionHandler();
    }

    async register(runtime: NativeRuntime, tenantId: string): Promise<void> {
        const self = this;
        await runtime.registerWorker(
            tenantId,
            this.agentIds,
            async (decisionJson: string) => {
                const request: WorkerDecisionRequestWire = JSON.parse(decisionJson);
                const submit = await self.handleDecision(request);
                return JSON.stringify(submit);
            },
        );
    }

    /**
     * Returns a fetch-compatible handler: (Request) => Promise<Response>.
     * Works with Bun.serve, Deno.serve, Cloudflare Workers, or any Node adapter.
     */
    fetchHandler(): (req: Request) => Promise<Response> {
        return async (req: Request) => {
            const decision = (await req.json()) as WorkerDecisionRequestWire;
            const submit = await this.handleDecision(decision);
            return Response.json(submit);
        };
    }

    async handleDecision(request: WorkerDecisionRequestWire): Promise<SubmitRequest> {
        const result = await this.handler(request);
        return {
            session_id: request.session_id,
            decision_id: request.decision_id,
            actions: result.actions,
            state: result.state,
            span: childSpan(request.span, "worker_submit"),
        };
    }
}

function randomHex(bytes: number): string {
    const buf = new Uint8Array(bytes);
    crypto.getRandomValues(buf);
    return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

function childSpan(parent: SpanContext, name: string): SpanContext {
    return {
        trace_id: parent.trace_id,
        span_id: randomHex(8),
        parent_span_id: parent.span_id,
        trace_flags: parent.trace_flags,
        trace_state: parent.trace_state,
        name,
    };
}
