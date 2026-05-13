import type { BackendClientOptions } from "./backend-client";
import { BackendClient } from "./backend-client";
import type { FrontendClientOptions } from "./frontend-client";
import { FrontendClient } from "./frontend-client";
import {
    jsonState,
    llmLoop,
    logging,
    messageHistory,
    messageHistoryCurrentTurn,
    stateSlice,
    subAgents,
    systemMessage,
    tool,
    tools,
} from "./middleware";
import type { ClientIdentity, ClientPayload, Event, SessionScope, TurnResult } from "./types";
import { drainToTurnResult } from "./types";
import type { FetchHandlerOptions, Handler, NativeRuntime } from "./worker";
import { HandlerBuilder, Worker } from "./worker";

// ── Agent factory ───────────────────────────────────────────────────────────

export interface AgentOptions {
    id: string;
}

export interface AgentFactory {
    (options: AgentOptions): HandlerBuilder<unknown>;
    jsonState: typeof jsonState;
    stateSlice: typeof stateSlice;
    tool: typeof tool;
    logging: typeof logging;
    messageHistory: typeof messageHistory;
    messageHistoryCurrentTurn: typeof messageHistoryCurrentTurn;
    systemMessage: typeof systemMessage;
    tools: typeof tools;
    llmLoop: typeof llmLoop;
    subAgents: typeof subAgents;
}

function createAgentFactory(): AgentFactory {
    const factory = ((options: AgentOptions) => {
        return new HandlerBuilder(options.id);
    }) as AgentFactory;

    factory.jsonState = jsonState;
    factory.stateSlice = stateSlice;
    factory.tool = tool;
    factory.logging = logging;
    factory.messageHistory = messageHistory;
    factory.messageHistoryCurrentTurn = messageHistoryCurrentTurn;
    factory.systemMessage = systemMessage;
    factory.tools = tools;
    factory.llmLoop = llmLoop;
    factory.subAgents = subAgents;

    return factory;
}

// ── Namespace objects ───────────────────────────────────────────────────────

class BackendNamespace {
    client(options: BackendClientOptions): BackendClient {
        return new BackendClient(options);
    }
}

class FrontendNamespace {
    client(options: FrontendClientOptions): FrontendClient {
        return new FrontendClient(options);
    }
}

// ── Embedded instance ───────────────────────────────────────────────────────

export interface EmbeddedOptions {
    agents: Handler[];
    /** SQLite database path (default: ":memory:") */
    db?: string;
    /** OpenRouter API base URL (default: "https://openrouter.ai/api") */
    openrouterBaseUrl?: string;
    /** OpenRouter API key */
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
}

export class EmbeddedInstance {
    private runtime: NativeRuntime;
    private worker: Worker;
    private registered: Promise<void>;

    constructor(runtime: NativeRuntime, agents: Handler[], tenantId: string) {
        this.runtime = runtime;
        this.worker = new Worker(agents);
        this.registered = this.worker.register(runtime, tenantId);
    }

    /** Fire-and-forget: enqueue a turn, return as soon as it's accepted. */
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

    /** Stream events for a session. If `scope.turnId` is set, the stream is
     *  filtered to that turn and auto-closes on completion. */
    async *stream(scope: SessionScope, options?: StreamOptions): AsyncGenerator<Event> {
        for await (const json of this.runtime.streamSession(scope.sessionId, scope.turnId, options?.sequenceAfter)) {
            yield JSON.parse(json) as Event;
        }
    }

    /** Stream a turn to completion and return its result. Requires `scope.turnId`. */
    turnResult(scope: SessionScope): Promise<TurnResult> {
        if (!scope.turnId) {
            throw new Error("turnResult requires scope.turnId");
        }
        return drainToTurnResult(this.stream(scope));
    }

    fetchHandler(options?: FetchHandlerOptions): (req: Request) => Promise<Response> {
        return this.worker.fetchHandler(options);
    }

    async shutdown(): Promise<void> {
        await this.runtime.shutdown();
    }
}

// ── Substructure ────────────────────────────────────────────────────────────

export class Substructure {
    readonly backend: BackendNamespace;
    readonly frontend: FrontendNamespace;
    readonly agent: AgentFactory;

    constructor() {
        this.backend = new BackendNamespace();
        this.frontend = new FrontendNamespace();
        this.agent = createAgentFactory();
    }

    worker(options: { agents: Handler[] }): Worker {
        return new Worker(options.agents);
    }

    async embedded(options: EmbeddedOptions): Promise<EmbeddedInstance> {
        const { EmbeddedRuntime } = await import("@substructure.ai/runtime");
        const runtime = new EmbeddedRuntime({
            db: options.db ?? ":memory:",
            openrouterBaseUrl: options.openrouterBaseUrl,
            openrouterApiKey: options.openrouterApiKey,
            llmPoolSize: options.llmPoolSize,
        });
        const instance = new EmbeddedInstance(runtime, options.agents, options.tenantId ?? "default");
        return instance;
    }
}
