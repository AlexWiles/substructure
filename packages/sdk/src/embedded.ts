// The only entry that touches the native `@substructure.ai/runtime` peer
// dependency, so consumers of the main `@substructure.ai/sdk` entry never pull
// it into their bundle. The runtime is thus imported with a plain literal
// specifier and full static types.

import type {
    ClientIdentity,
    ClientPayload,
    Event,
    PersistedEvent,
    SessionScope,
    SubmitToolCallResultArgs,
    TurnResult,
} from "./types";
import { drainToTurnResult, isTokenDelta } from "./types";
import type { FetchHandlerOptions, Handler, NativeRuntime } from "./worker";
import { Worker } from "./worker";

export interface EmbeddedOptions {
    agents: Handler[];
    /** SQLite database path (default: ":memory:") */
    db?: string;
    /** OpenRouter API base URL (default: "https://openrouter.ai/api") */
    openrouterBaseUrl?: string;
    openrouterApiKey?: string;
    /** Number of concurrent LLM handler tasks (default: 4) */
    llmPoolSize?: number;
    /** Tenant id under which to register the worker (default: "default") */
    tenantId?: string;
}

export interface StartTurnRequest {
    agentId: string;
    payload: ClientPayload;
    identity?: ClientIdentity;
    sessionId?: string;
    turnId?: string;
}

export interface StreamOptions {
    sequenceAfter?: number;
    /** Include transient `llm.token.delta` events. Off by default, so
     *  `stream()` yields only persisted events. Deltas only arrive when
     *  streaming is enabled on the agent's `llm`. */
    tokens?: boolean;
}

export class SubstructureEmbedded {
    private runtime: NativeRuntime;
    private worker: Worker;
    private registered: Promise<void>;
    private tenantId: string;

    /** Requires the optional `@substructure.ai/runtime` native package. The
     *  native runtime loads asynchronously, hence a factory rather than `new`. */
    static async create(options: EmbeddedOptions): Promise<SubstructureEmbedded> {
        const { EmbeddedRuntime } = await import("@substructure.ai/runtime");
        const runtime = new EmbeddedRuntime({
            db: options.db ?? ":memory:",
            openrouterBaseUrl: options.openrouterBaseUrl,
            openrouterApiKey: options.openrouterApiKey,
            llmPoolSize: options.llmPoolSize,
        });
        return new SubstructureEmbedded(runtime, options.agents, options.tenantId ?? "default");
    }

    private constructor(runtime: NativeRuntime, agents: Handler[], tenantId: string) {
        this.runtime = runtime;
        this.worker = new Worker(agents);
        this.tenantId = tenantId;
        this.registered = this.worker.register(runtime, tenantId);
    }

    /** Fire-and-forget: returns as soon as the turn is accepted, not complete. */
    async startTurn(request: StartTurnRequest): Promise<SessionScope> {
        await this.registered;
        const identity = request.identity;
        if (!identity?.id) {
            throw new Error("startTurn.identity.id is required for embedded runtime");
        }
        const sessionId = request.sessionId ?? crypto.randomUUID();
        return this.runtime.submitPayload(
            sessionId,
            request.agentId,
            JSON.stringify(request.payload),
            JSON.stringify(identity),
            request.turnId,
        );
    }

    /** When `scope.turnId` is set, the stream is filtered to that turn and
     *  auto-closes on completion. */
    stream(scope: SessionScope, options?: StreamOptions & { tokens?: false }): AsyncGenerator<PersistedEvent>;
    stream(scope: SessionScope, options: StreamOptions & { tokens: true }): AsyncGenerator<Event>;
    stream(scope: SessionScope, options: StreamOptions): AsyncGenerator<Event>;
    async *stream(scope: SessionScope, options?: StreamOptions): AsyncGenerator<Event> {
        for await (const json of this.runtime.streamSession(
            this.tenantId,
            scope.sessionId,
            scope.turnId,
            options?.sequenceAfter,
        )) {
            const event = JSON.parse(json) as Event;
            if (!options?.tokens && isTokenDelta(event)) continue;
            yield event;
        }
    }

    turnResult(scope: SessionScope): Promise<TurnResult> {
        if (!scope.turnId) {
            throw new Error("turnResult requires scope.turnId");
        }
        return drainToTurnResult(this.stream(scope));
    }

    /** Complete (or fail) a tool call out-of-band, after a tool returns `DEFERRED`. */
    async submitToolCallResult(args: SubmitToolCallResultArgs): Promise<void> {
        if (args.result !== undefined) {
            await this.runtime.submitToolCallResult(
                args.sessionId,
                this.tenantId,
                args.toolCallId,
                args.attempt,
                args.result,
                undefined,
                undefined,
            );
        } else {
            await this.runtime.submitToolCallResult(
                args.sessionId,
                this.tenantId,
                args.toolCallId,
                args.attempt,
                undefined,
                args.error,
                args.retryable,
            );
        }
    }

    fetchHandler(options?: FetchHandlerOptions): (req: Request) => Promise<Response> {
        return this.worker.fetchHandler(options);
    }

    async shutdown(): Promise<void> {
        await this.runtime.shutdown();
    }
}
