import type { Event, SubmitPayloadRequest, WorkerDecisionRequestWire } from "./types";

// ── Runtime interface ────────────────────────────────────────────────────────

/**
 * Minimal interface that the NAPI EmbeddedRuntime must satisfy.
 * This avoids a hard dependency on @substructure.ai/runtime.
 */
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
        authJson: string,
        turnId?: string,
    ): AsyncGenerator<string, void, unknown>;
    shutdown(): Promise<void>;
}

// ── HasHandler (re-exported from worker) ─────────────────────────────────────

export interface DecisionResult {
    actions: unknown[];
    state: string;
}

export type DecisionHandler = (request: WorkerDecisionRequestWire) => Promise<DecisionResult>;

export interface HasHandler {
    readonly id: string;
    handler(): DecisionHandler;
}

// ── InProcessRuntime ─────────────────────────────────────────────────────────

export interface InProcessRuntimeOptions {
    /**
     * The native runtime instance from @substructure.ai/runtime.
     *
     * ```ts
     * import { EmbeddedRuntime } from '@substructure.ai/runtime'
     * const native = new EmbeddedRuntime()
     * const runtime = new InProcessRuntime({ native })
     * ```
     */
    native: NativeRuntime;
}

/**
 * Wraps the NAPI runtime into the same interface as the HTTP clients.
 *
 * ```ts
 * import { EmbeddedRuntime } from '@substructure.ai/runtime'
 * import { InProcessRuntime } from '@substructure.ai/sdk/runtime'
 * import { defineAgent } from '@substructure.ai/sdk/agent'
 *
 * const runtime = new InProcessRuntime({ native: new EmbeddedRuntime() })
 * const worker = defineAgent("my-agent")
 * await runtime.register(worker, 'default')
 *
 * for await (const event of runtime.submitPayload({
 *   agent_id: 'my-agent',
 *   payload: { type: 'message', message: { role: 'user', content: 'hello' } },
 * })) {
 *   console.log(event.payload.type)
 * }
 * ```
 */
export class InProcessRuntime {
    private native: NativeRuntime;

    constructor(options: InProcessRuntimeOptions) {
        this.native = options.native;
    }

    /**
     * Register a Worker (or set of agents) with the in-process runtime.
     * This connects the worker's decision handlers directly — no HTTP.
     */
    async register(
        worker: { agentIds: string[]; handleDecision: (request: WorkerDecisionRequestWire) => Promise<unknown> },
        tenantId: string,
    ): Promise<void> {
        await this.native.registerWorker(tenantId, worker.agentIds, async (decisionJson: string) => {
            const request: WorkerDecisionRequestWire = JSON.parse(decisionJson);
            const result = await worker.handleDecision(request);
            return JSON.stringify(result);
        });
    }

    /**
     * Submit a payload — same interface as UserClient.submitPayload().
     */
    async *submitPayload(request: SubmitPayloadRequest): AsyncGenerator<Event> {
        if (!request.auth?.sub) {
            throw new Error("auth.sub is required for embedded runtime submitPayload");
        }
        const sessionId = request.session_id ?? crypto.randomUUID();

        for await (const json of this.native.submitPayload(
            sessionId,
            request.agent_id,
            JSON.stringify(request.payload),
            JSON.stringify(request.auth),
            request.turn_id,
        )) {
            yield JSON.parse(json) as Event;
        }
    }

    /**
     * Shut down the runtime and all worker loops.
     */
    async shutdown(): Promise<void> {
        await this.native.shutdown();
    }
}
