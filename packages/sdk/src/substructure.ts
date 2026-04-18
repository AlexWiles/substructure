import type { ClientPayload, ClientIdentity, Event } from "./types";
import type { FetchHandlerOptions, NativeRuntime } from "./worker";
import { Worker, HandlerBuilder } from "./worker";
import type { Handler } from "./worker";
import { RunStream } from "./run-stream";
import { BackendClient } from "./backend-client";
import type { BackendClientOptions } from "./backend-client";
import { FrontendClient } from "./frontend-client";
import type { FrontendClientOptions } from "./frontend-client";
import {
    jsonState,
    stateSlice,
    tool,
    logging,
    messageHistory,
    messageHistoryCurrentTurn,
    systemMessage,
    tools,
    llmLoop,
    subAgents,
} from "./middleware";

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
}

export interface SubmitRequest {
    agentId: string;
    payload: ClientPayload;
    identity?: ClientIdentity;
    sessionId?: string;
    turnId?: string;
}

export class EmbeddedInstance {
    private runtime: NativeRuntime;
    private worker: Worker;
    private registered: Promise<void>;

    constructor(runtime: NativeRuntime, agents: Handler[]) {
        this.runtime = runtime;
        this.worker = new Worker(agents);
        this.registered = this.worker.register(runtime, "default");
    }

    submit(request: SubmitRequest): RunStream {
        const sessionId = request.sessionId ?? crypto.randomUUID();
        const identity = request.identity;
        const turnId = request.turnId;

        const self = this;
        async function* generate(): AsyncGenerator<Event> {
            await self.registered;
            if (!identity?.id) {
                throw new Error("submit.identity.id is required for embedded runtime");
            }
            for await (const json of self.runtime.submitPayload(
                sessionId,
                request.agentId,
                JSON.stringify(request.payload),
                JSON.stringify(identity),
                turnId,
            )) {
                yield JSON.parse(json) as Event;
            }
        }

        return new RunStream(generate());
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
        const instance = new EmbeddedInstance(runtime, options.agents);
        return instance;
    }
}
