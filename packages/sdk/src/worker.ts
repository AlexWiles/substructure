import type { WorkerDecisionRequestWire, WorkerAction, SubmitRequest, SpanContext, LlmTokenDeltaInput } from "./types";
import { jsonState } from "./middleware";
import { createSseStream, type SseStream } from "./sse";
export interface NativeRuntime {
    registerWorker(
        tenantId: string,
        agentIds: string[],
        callback: (decision: string) => Promise<string>,
    ): Promise<void>;
    submitPayload(
        sessionId: string,
        agentId: string,
        payloadJson: string,
        identityJson: string,
        turnId?: string,
    ): Promise<{ sessionId: string; turnId: string }>;
    submitToolCallResult(
        sessionId: string,
        tenantId: string,
        toolCallId: string,
        attempt: number,
        resultJson: string | undefined,
        errorMessage: string | undefined,
        retryable: boolean | undefined,
    ): Promise<void>;
    streamSession(
        tenantId: string,
        sessionId: string,
        turnId?: string,
        sequenceAfter?: number,
    ): AsyncGenerator<string, void, unknown>;
    emitTokenDelta(deltaJson: string): Promise<void>;
    shutdown(): Promise<void>;
}
import { verifyWebhookSignature } from "./webhook";

export interface DecisionResult {
    actions: WorkerAction[];
    state: string;
}

export interface DecisionRuntime {
    emitDelta?: (delta: LlmTokenDeltaInput) => Promise<void>;
}

export type DecisionHandler = (
    request: WorkerDecisionRequestWire,
    runtime?: DecisionRuntime,
) => Promise<DecisionResult>;

export interface AgentContext<S = unknown> {
    state: S;
    /** Raw decision envelope off the wire (snake_case). */
    request: WorkerDecisionRequestWire;
    emitDelta?: (delta: LlmTokenDeltaInput) => Promise<void>;
}

export interface AgentResponse {
    actions: WorkerAction[];
    state: unknown;
    workerState?: string;
}

export type Next<S = unknown> = (ctx: AgentContext<S>) => Promise<AgentResponse> | AgentResponse;

/** The `Out` type parameter is the state type downstream middleware sees:
 *  passthrough middleware leaves it unchanged, state providers transform it. */
export interface MiddlewareFn<In = unknown, Out = In> {
    (ctx: AgentContext<In>, next: Next<Out>): Promise<AgentResponse> | AgentResponse;
    /** Type brand carrying the output state type — never set at runtime. */
    readonly _out?: Out;
}

export interface Handler {
    readonly agentId: string;
    toDecisionHandler(): DecisionHandler;
}

/** The `_contributes` brand tells the builder to intersect `A` onto the
 *  current state type. */
export type StateContributor<A> = MiddlewareFn<any, any> & { readonly _contributes: A };

/** Lets a constructed agent be passed straight to `.use()` without the builder
 *  knowing how its middleware is built. */
export interface MiddlewareSource {
    toMiddleware(): MiddlewareFn<any, any>;
}

type UnknownMiddleware = MiddlewareFn<unknown, unknown>;
type UnknownNext = Next<unknown>;

const DEFAULT_FALLBACK: UnknownNext = (ctx) => ({ actions: [], state: ctx.state });

export class HandlerBuilder<S> implements Handler {
    readonly agentId: string;
    private middlewares: UnknownMiddleware[] = [];

    constructor(agentId: string) {
        this.agentId = agentId;
    }

    /** State contributor: intersects new keys onto current state. */
    use<A>(mw: StateContributor<A>): HandlerBuilder<S & A>;
    /** State transformer: replaces state type (e.g. withState). */
    use<Out>(mw: MiddlewareFn<S, Out>): HandlerBuilder<Out>;
    /** Constructed agent or other middleware source (e.g. `new ToolLoopAgent(...)`). */
    use(mw: MiddlewareSource): HandlerBuilder<unknown>;
    use<Out>(mw: MiddlewareFn<S, Out> | MiddlewareSource): HandlerBuilder<Out> {
        const fn = typeof mw === "function" ? mw : mw.toMiddleware();
        this.middlewares.push(fn as UnknownMiddleware);
        return this as unknown as HandlerBuilder<Out>;
    }

    toDecisionHandler(): DecisionHandler {
        // `jsonState` is applied by default as the outermost middleware: it
        // decodes `worker_state` into `ctx.state` on the way in and re-encodes
        // it on the way out. Applying it again explicitly is idempotent, so
        // chains that still `.use(agent.jsonState())` keep working.
        const middlewares = [jsonState() as UnknownMiddleware, ...this.middlewares];
        const chain = composeChain(middlewares, DEFAULT_FALLBACK);

        return async (request: WorkerDecisionRequestWire, runtime?: DecisionRuntime) => {
            const ctx: AgentContext<unknown> = {
                state: undefined,
                request,
                emitDelta: runtime?.emitDelta,
            };
            const result = await chain(ctx);
            return {
                actions: result.actions,
                state: result.workerState ?? request.worker_state,
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

export interface FetchHandlerOptions {
    signingSecret?: string;
    /** Timestamp validation tolerance in seconds (default: 300) */
    tolerance?: number;
}

export class Worker {
    readonly agentIds: string[];
    private handlers: Map<string, DecisionHandler>;

    constructor(agents: Handler[]) {
        this.handlers = new Map();
        for (const handler of agents) {
            this.handlers.set(handler.agentId, handler.toDecisionHandler());
        }
        this.agentIds = [...this.handlers.keys()];
    }

    async register(runtime: NativeRuntime, tenantId: string): Promise<void> {
        const self = this;
        await runtime.registerWorker(tenantId, this.agentIds, async (decisionJson: string) => {
            const request: WorkerDecisionRequestWire = JSON.parse(decisionJson);
            const submit = await self.handleDecision(request, embeddedDecisionRuntime(runtime, request));
            return JSON.stringify(submit);
        });
    }

    /** When `options.signingSecret` is provided, incoming requests are verified
     *  against the HMAC-SHA256 signature in the `X-Substructure-Signature` header. */
    fetchHandler(options?: FetchHandlerOptions): (req: Request) => Promise<Response> {
        return async (req: Request) => {
            let decision: WorkerDecisionRequestWire;

            if (options?.signingSecret) {
                decision = await verifyWebhookSignature<WorkerDecisionRequestWire>(req, options.signingSecret, {
                    tolerance: options.tolerance,
                });
            } else {
                decision = (await req.json()) as WorkerDecisionRequestWire;
            }

            const wantsStream = req.headers.get("accept")?.includes("text/event-stream") ?? false;
            if (wantsStream) {
                return this.handleDecisionStream(decision);
            }

            const submit = await this.handleDecision(decision);
            return Response.json(submit);
        };
    }

    async handleDecision(request: WorkerDecisionRequestWire, runtime?: DecisionRuntime): Promise<SubmitRequest> {
        const handler = this.handlers.get(request.agent_id);
        if (!handler) {
            throw new Error(`No handler registered for agent: ${request.agent_id}`);
        }
        const result = await handler(request, runtime);
        return {
            session_id: request.session_id,
            decision_id: request.decision_id,
            actions: result.actions,
            state: result.state,
            span: childSpan(request.span, "worker_submit"),
        };
    }

    private handleDecisionStream(request: WorkerDecisionRequestWire): Response {
        const sse = createSseStream();

        void (async () => {
            try {
                const submit = await this.handleDecision(request, sseDecisionRuntime(sse));
                await sse.writeSSE({ event: "decision.result", data: submit });
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                await sse.writeSSE({ event: "decision.error", data: { message, retryable: true } });
            } finally {
                await sse.close();
            }
        })();

        return new Response(sse.readable, {
            headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        });
    }
}

function sseDecisionRuntime(sse: SseStream): DecisionRuntime {
    return {
        emitDelta: (delta) => sse.writeSSE({ event: "llm.token.delta", data: delta }),
    };
}

function embeddedDecisionRuntime(
    runtime: NativeRuntime,
    request: WorkerDecisionRequestWire,
): DecisionRuntime | undefined {
    const { trigger } = request;
    if (trigger.type !== "llm.request" || !trigger.stream) return undefined;

    const rootSessionId = request.ancestry?.[0] ?? request.session_id;
    let seq = 0;

    return {
        emitDelta: async (delta: LlmTokenDeltaInput) => {
            await runtime.emitTokenDelta(
                JSON.stringify({
                    tenant_id: request.tenant_id,
                    root_session_id: rootSessionId,
                    session_id: request.session_id,
                    agent_id: request.agent_id,
                    turn_id: request.turn_id,
                    call_id: trigger.call_id,
                    attempt: trigger.attempt,
                    seq: seq++,
                    text: delta.text,
                    reasoning: delta.reasoning,
                    tool_calls: delta.tool_calls,
                    finish_reason: delta.finish_reason,
                }),
            );
        },
    };
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
