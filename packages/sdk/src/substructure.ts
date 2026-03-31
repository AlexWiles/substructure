import type { Event, Decimal, TurnCompleted, ClientPayload, ClientIdentity } from "./types";
import type { NativeRuntime } from "./runtime";
import type { Handler, FetchHandlerOptions } from "./worker";
import { Worker } from "./worker";

export { contentText } from "./types";
export {
    defineAgent,
    state,
    stateSlice,
    logging,
    tool,
    messageHistory,
    systemMessage,
    tools,
    llmLoop,
    subAgents,
} from "./worker";
export type { AgentRequest, AgentResponse, MiddlewareFn, FetchHandlerOptions } from "./worker";
export { verifyWebhookSignature, WebhookVerificationError } from "./webhook";
export { BackendClient } from "./backend-client";
export { FrontendClient } from "./frontend-client";

// ── RunStream ────────────────────────────────────────────────────────────────

export interface TurnResult {
    turnId: string;
    data: unknown;
    cost: Decimal;
    tokenUsage: Record<string, number>;
}

export interface SubmitRequest {
    agentId: string;
    payload: ClientPayload;
    auth?: ClientIdentity;
    sessionId?: string;
    turnId?: string;
}

export class RunStream {
    readonly result: Promise<TurnResult>;
    private events: Event[] = [];
    private resolveResult!: (r: TurnResult) => void;
    private rejectResult!: (e: Error) => void;
    private source: AsyncGenerator<Event>;

    constructor(source: AsyncGenerator<Event>) {
        this.source = source;
        this.result = new Promise<TurnResult>((resolve, reject) => {
            this.resolveResult = resolve;
            this.rejectResult = reject;
        });
    }

    async *[Symbol.asyncIterator](): AsyncGenerator<Event> {
        try {
            for await (const event of this.source) {
                this.events.push(event);
                yield event;
            }
            const tc = this.events.findLast(
                (e): e is Event & { payload: TurnCompleted } =>
                    e.payload.type === "turn.completed",
            );
            if (tc) {
                this.resolveResult({
                    turnId: tc.payload.turn_id,
                    data: tc.payload.data,
                    cost: tc.payload.turn_cost ?? "0",
                    tokenUsage: tc.payload.turn_token_usage ?? {},
                });
            } else {
                this.rejectResult(new Error("stream ended without turn.completed"));
            }
        } catch (err) {
            this.rejectResult(err instanceof Error ? err : new Error(String(err)));
            throw err;
        }
    }
}

// ── Config ──────────────────────────────────────────────────────────────────

export interface SubstructureConfig {
    /** A NativeRuntime instance (e.g. from `@substructure.ai/runtime`) */
    runtime: NativeRuntime;
}

// ── Substructure ────────────────────────────────────────────────────────────

export class Substructure {
    private runtime: NativeRuntime;
    private agents: Handler[] = [];
    private worker: Worker | null = null;
    private registered: Promise<void> | null = null;

    constructor(config: SubstructureConfig) {
        this.runtime = config.runtime;
    }

    agent(handler: Handler): this {
        if (this.registered) {
            throw new Error("Cannot register agents after submit() has been called");
        }
        this.agents.push(handler);
        return this;
    }

    private ensureWorker(): Worker {
        if (!this.worker) {
            this.worker = new Worker(this.agents);
        }
        return this.worker;
    }

    private register(): Promise<void> {
        if (this.registered) return this.registered;
        this.registered = (async () => {
            const worker = this.ensureWorker();
            await worker.register(this.runtime, "default");
        })();
        return this.registered;
    }

    private submitPayload(
        agentId: string,
        payload: ClientPayload,
        options?: Omit<SubmitRequest, "agentId" | "payload">,
    ): RunStream {
        const sessionId = options?.sessionId ?? crypto.randomUUID();
        const auth = options?.auth;
        const turnId = options?.turnId;

        const self = this;
        async function* generate(): AsyncGenerator<Event> {
            await self.register();
            if (!auth?.sub) {
                throw new Error("submit.auth.sub is required for embedded runtime");
            }
            for await (const json of self.runtime.submitPayload(
                sessionId,
                agentId,
                JSON.stringify(payload),
                JSON.stringify(auth),
                turnId,
            )) {
                yield JSON.parse(json) as Event;
            }
        }

        return new RunStream(generate());
    }

    submit(request: SubmitRequest): RunStream {
        const { agentId, payload, ...options } = request;
        return this.submitPayload(agentId, payload, options);
    }

    fetchHandler(options?: FetchHandlerOptions): (req: Request) => Promise<Response> {
        return this.ensureWorker().fetchHandler(options);
    }

    async shutdown(): Promise<void> {
        if (this.runtime) {
            await this.runtime.shutdown();
        }
    }
}
