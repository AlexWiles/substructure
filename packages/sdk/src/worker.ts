import type { Agent, DecisionRequest } from "./core";
import { createSseStream, type SseStream } from "./sse";
import type { LlmTokenDeltaInput, SpanContext, SubmitRequest, WorkerDecisionRequestWire } from "./types";
import { verifyWebhookSignature } from "./webhook";

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

export interface DecisionRuntime {
    emitDelta?: (delta: LlmTokenDeltaInput) => Promise<void>;
}

export interface FetchHandlerOptions {
    signingSecret?: string;
    /** Timestamp validation tolerance in seconds (default: 300) */
    tolerance?: number;
}

/** The agents to serve — each must be named (`agent({ name, ... })`). They carry
 *  independent state types, so the collection is `Agent<any>`. */
// biome-ignore lint/suspicious/noExplicitAny: heterogeneous state types across agents
export type Agents = Agent<any>[];

// ── State codec (base64 JSON, the worker boundary) ───────────────────────────

function decodeState(raw: string): unknown {
    if (!raw) return {};
    return JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(raw), (c) => c.charCodeAt(0))));
}

function encodeState(value: unknown): string {
    return btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(value))));
}

/** The one transform: decode `worker_state` into the decision, run the agent,
 *  encode the returned state back out. Everything else is the wire envelope. */
async function runDecision(
    fn: Agent,
    request: WorkerDecisionRequestWire,
    runtime?: DecisionRuntime,
): Promise<SubmitRequest> {
    const req: DecisionRequest = {
        ...request,
        state: decodeState(request.worker_state),
        emitDelta: runtime?.emitDelta,
    };
    const out = await fn(req);
    return {
        session_id: request.session_id,
        decision_id: request.decision_id,
        actions: out.actions ?? [],
        transcript: out.transcript ?? request.transcript ?? [],
        state: encodeState(out.state !== undefined ? out.state : req.state),
        span: childSpan(request.span, "worker_submit"),
    };
}

export class Worker {
    readonly agentIds: string[];
    private agents: Record<string, Agent>;

    constructor(agents: Agents) {
        this.agents = {};
        for (const a of agents) {
            if (!a.agentName) {
                throw new Error("agent passed to worker() has no name — give it agent({ name, ... })");
            }
            this.agents[a.agentName] = a;
        }
        this.agentIds = Object.keys(this.agents);
    }

    async register(runtime: NativeRuntime, tenantId: string): Promise<void> {
        await runtime.registerWorker(tenantId, this.agentIds, async (decisionJson: string) => {
            const request: WorkerDecisionRequestWire = JSON.parse(decisionJson);
            const submit = await this.handleDecision(request, embeddedDecisionRuntime(runtime, request));
            return JSON.stringify(submit);
        });
    }

    /** `worker({...}).fetch(opts)` — alias for `fetchHandler`. */
    fetch(options?: FetchHandlerOptions): (req: Request) => Promise<Response> {
        return this.fetchHandler(options);
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
        const fn = this.agents[request.agent_id];
        if (!fn) {
            throw new Error(`No agent registered for: ${request.agent_id}`);
        }
        return runDecision(fn, request, runtime);
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

/** Build the HTTP worker for these (named) agents. `worker([agent]).fetch(opts)`
 *  is a `(Request) => Response` handler; each agent is routed by its name. */
export function worker(agents: Agents): Worker {
    return new Worker(agents);
}

/** One-liner: the fetch handler for these agents. */
export function serve(agents: Agents, opts?: FetchHandlerOptions): (req: Request) => Promise<Response> {
    return new Worker(agents).fetch(opts);
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
