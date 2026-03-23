import type {
    WorkerDecisionRequestWire,
    WorkerAction,
    SubmitRequest,
    SpanContext,
} from "./types";
import type { NativeRuntime } from "./runtime";

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
    readonly agentId: string;
    toDecisionHandler(): DecisionHandler;
}

// ── State ───────────────────────────────────────────────────────────────────

/**
 * Middleware that contributes state keys. The `_contributes` brand
 * tells the builder to intersect `A` onto the current state type.
 */
export type StateContributor<A> = MiddlewareFn<unknown, unknown> & { readonly _contributes: A };

export {
    withState,
    tool,
    withLogging,
    withMessageHistory,
    withMessages,
    withConversation,
    withSystemMessage,
    withTools,
    withCallLLM,
    withSubAgents,
} from "./middleware";
export type {
    ToolFn,
    ToolDef,
    SubAgentTrack,
    MessagesAdapter,
    SubAgentTrackerAdapter,
    CallLlmSelection,
    ToolSelector,
    MessageSelector,
    SystemMessageSelector,
} from "./middleware";

// ── HandlerBuilder ──────────────────────────────────────────────────────────

type UnknownMiddleware = MiddlewareFn<unknown, unknown>;
type UnknownNext = Next<unknown>;

const DEFAULT_FALLBACK: UnknownNext = () => ({ actions: [] });

class HandlerBuilder<S> implements Handler {
    readonly agentId: string;
    private middlewares: UnknownMiddleware[] = [];

    constructor(agentId: string) {
        this.agentId = agentId;
    }

    /** State contributor: intersects new keys onto current state. */
    use<A>(mw: StateContributor<A>): HandlerBuilder<S & A>;
    /** State transformer: replaces state type (e.g. withState). */
    use<Out>(mw: MiddlewareFn<S, Out>): HandlerBuilder<Out>;
    use<Out>(mw: MiddlewareFn<S, Out>): HandlerBuilder<Out> {
        this.middlewares.push(mw as UnknownMiddleware);
        return this as unknown as HandlerBuilder<Out>;
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
                state: ctx.request.worker_state,
            };
        };
    }
}

function composeChain(middlewares: UnknownMiddleware[], handle: UnknownNext): UnknownNext {
    let fn: UnknownNext = handle;
    for (let i = middlewares.length - 1; i >= 0; i--) {
        const mw = middlewares[i];
        const next = fn;
        fn = (ctx) => mw(ctx, next);
    }
    return fn;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export function defineAgent(agentId: string): HandlerBuilder<{}> {
    return new HandlerBuilder(agentId);
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
