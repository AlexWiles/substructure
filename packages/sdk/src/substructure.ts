import type { BackendClientOptions } from "./backend-client";
import { BackendClient } from "./backend-client";
import type { FrontendClientOptions } from "./frontend-client";
import { FrontendClient } from "./frontend-client";
import {
    action,
    actions,
    jsonState,
    llmLoop,
    logging,
    messageHistory,
    messageHistoryCurrentTurn,
    stateSlice,
    subAgents,
    tool,
    tools,
} from "./middleware";
import type { Handler } from "./worker";
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
    action: typeof action;
    actions: typeof actions;
    logging: typeof logging;
    messageHistory: typeof messageHistory;
    messageHistoryCurrentTurn: typeof messageHistoryCurrentTurn;
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
    factory.action = action;
    factory.actions = actions;
    factory.logging = logging;
    factory.messageHistory = messageHistory;
    factory.messageHistoryCurrentTurn = messageHistoryCurrentTurn;
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
}
