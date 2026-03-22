import type {
    WorkerDecisionRequestWire,
    WorkerAction,
    SubmitRequest,
    SpanContext,
    RetryPolicy,
} from "./types";
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
 * Decodes `request.worker_state` into `ctx.state` (falling back to `init`),
 * runs the chain, then encodes `ctx.state` back.
 */
export function withState<S extends object>(init: S): StateContributor<S> {
    const mw: MiddlewareFn<any, any> = async (ctx, next) => {
        const raw = ctx.request.worker_state;
        ctx.state = (raw && raw !== "")
            ? JSON.parse(Buffer.from(raw, "base64").toString("utf-8"))
            : { ...init };
        const result = await next(ctx);
        ctx._encodedState = Buffer.from(JSON.stringify(ctx.state), "utf-8").toString("base64");
        return result;
    };
    return Object.assign(mw, { _contributes: init });
}

// ── Tool types ──────────────────────────────────────────────────────────────

export type ToolFn = (args: Record<string, unknown>) => Promise<unknown> | unknown;

export interface ToolDef {
    description: string;
    parameters: unknown;
    execute: ToolFn;
    retry?: RetryPolicy;
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

    /** Composable object (e.g. Agent): extracts middleware and intersects its state. */
    use<A>(mw: Composable<A>): HandlerBuilder<S & A>;
    /** State contributor: intersects new keys onto current state. */
    use<A>(mw: StateContributor<A>): HandlerBuilder<S & A>;
    /** State transformer: replaces state type (e.g. withState). */
    use<M extends MiddlewareFn<S, any>>(mw: M): HandlerBuilder<IsAny<MiddlewareOut<M>> extends true ? S : MiddlewareOut<M>>;
    use(mw: AnyMiddleware | Composable): HandlerBuilder<any> {
        if (typeof mw !== "function" && "toMiddleware" in mw) {
            this.middlewares.push(mw.toMiddleware());
        } else {
            this.middlewares.push(mw as AnyMiddleware);
        }
        return this as any;
    }

    toDecisionHandler(): DecisionHandler {
        const middlewares = this.middlewares;
        const chain = composeChain(middlewares, DEFAULT_FALLBACK);

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
export function defineAgent(): HandlerBuilder<{}> {
    return new HandlerBuilder();
}

// ── Worker (internal) ───────────────────────────────────────────────────────

export class Worker {
    readonly agentIds: string[];
    private handlers: Map<string, DecisionHandler>;

    constructor(agents: Record<string, Handler>) {
        this.handlers = new Map();
        for (const [id, handler] of Object.entries(agents)) {
            this.handlers.set(id, handler.toDecisionHandler());
        }
        this.agentIds = [...this.handlers.keys()];
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
        const handler = this.handlers.get(request.agent_id);
        if (!handler) {
            throw new Error(`No handler registered for agent: ${request.agent_id}`);
        }
        const result = await handler(request);
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
