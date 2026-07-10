import type { Agent, DecisionRequest, EmitDelta, NamedAgent } from "./core";
import { createSseStream, type SseStream } from "./sse";
import type { LlmTokenDeltaInput, WireDecisionRequest, WireDecisionResponse } from "./types";
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
    settleEffect(
        sessionId: string,
        tenantId: string,
        kind: string,
        id: string,
        attempt: number | undefined,
        resultJson: string | undefined,
        responseJson: string | undefined,
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

export interface FetchHandlerOptions {
    signingSecret?: string;
}

/** The agents to serve, each a `NamedAgent` from `agent({ name, ... })`. */
// biome-ignore lint/suspicious/noExplicitAny: heterogeneous state types across agents
export type Agents = NamedAgent<any>[];

async function runDecision(
    fn: Agent,
    request: WireDecisionRequest,
    emitDelta?: EmitDelta,
): Promise<WireDecisionResponse> {
    const req: DecisionRequest = { ...request, state: request.state ?? null, emitDelta };
    const out = await fn(req);
    return {
        actions: out.actions ?? [],
        messages: out.messages ?? request.messages ?? [],
        // undefined keeps the current state; a returned value is deduped engine-side.
        state: out.state,
    };
}

export class Worker {
    readonly agentIds: string[];
    private agents: Record<string, Agent>;

    constructor(agents: Agents) {
        this.agents = {};
        for (const a of agents) {
            this.agents[a.agentName] = a;
        }
        this.agentIds = Object.keys(this.agents);
    }

    async register(runtime: NativeRuntime, tenantId: string): Promise<void> {
        await runtime.registerWorker(tenantId, this.agentIds, async (decisionJson: string) => {
            const request: WireDecisionRequest = JSON.parse(decisionJson);
            const submit = await this.handleDecision(request, embeddedEmitDelta(runtime, request));
            return JSON.stringify(submit);
        });
    }

    /** `worker({...}).fetch(opts)`, alias for `fetchHandler`. */
    fetch(options?: FetchHandlerOptions): (req: Request) => Promise<Response> {
        return this.fetchHandler(options);
    }

    /** With `options.signingSecret`, verifies the `X-Substructure-Signature` HMAC-SHA256 header. */
    fetchHandler(options?: FetchHandlerOptions): (req: Request) => Promise<Response> {
        return async (req: Request) => {
            let decision: WireDecisionRequest;

            if (options?.signingSecret) {
                decision = await verifyWebhookSignature<WireDecisionRequest>(req, options.signingSecret);
            } else {
                decision = (await req.json()) as WireDecisionRequest;
            }

            const wantsStream = req.headers.get("accept")?.includes("text/event-stream") ?? false;
            if (wantsStream) {
                return this.handleDecisionStream(decision);
            }

            const submit = await this.handleDecision(decision);
            return Response.json(submit);
        };
    }

    async handleDecision(request: WireDecisionRequest, emitDelta?: EmitDelta): Promise<WireDecisionResponse> {
        const fn = this.agents[request.agent_id];
        if (!fn) {
            throw new Error(`No agent registered for: ${request.agent_id}`);
        }
        return runDecision(fn, request, emitDelta);
    }

    private handleDecisionStream(request: WireDecisionRequest): Response {
        const sse = createSseStream();

        void (async () => {
            try {
                const submit = await this.handleDecision(request, sseEmitDelta(sse));
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

/** Build the HTTP worker for these agents; `worker([...]).fetch(opts)` is a
 *  `(Request) => Response` handler routed by agent name. */
export function worker(agents: Agents): Worker {
    return new Worker(agents);
}

/** One-liner: the fetch handler for these agents. */
export function serve(agents: Agents, opts?: FetchHandlerOptions): (req: Request) => Promise<Response> {
    return new Worker(agents).fetch(opts);
}

function sseEmitDelta(sse: SseStream): EmitDelta {
    return (delta) => sse.writeSSE({ event: "llm.token.delta", data: delta });
}

function embeddedEmitDelta(runtime: NativeRuntime, request: WireDecisionRequest): EmitDelta | undefined {
    const { trigger } = request;
    if (trigger.type !== "llm.execute" || !trigger.stream) return undefined;

    const rootSessionId = request.ancestry?.[0] ?? request.session_id;
    let seq = 0;

    return async (delta: LlmTokenDeltaInput) => {
        await runtime.emitTokenDelta(
            JSON.stringify({
                tenant_id: request.identity.tenant_id,
                root_session_id: rootSessionId,
                session_id: request.session_id,
                agent_id: request.agent_id,
                turn_id: request.turn_id,
                call_id: trigger.id,
                attempt: trigger.attempt,
                seq: seq++,
                text: delta.text,
                reasoning: delta.reasoning,
                tool_calls: delta.tool_calls,
                finish_reason: delta.finish_reason,
            }),
        );
    };
}
