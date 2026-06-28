import type { BackendClientOptions } from "./backend-client";
import { BackendClient } from "./backend-client";
import type { FrontendClientOptions } from "./frontend-client";
import { FrontendClient } from "./frontend-client";
import {
    action,
    actions,
    jsonState,
    llm,
    logging,
    serverGenerate,
    stateSlice,
    stepCountIs,
    stopWhen,
    subAgents,
    tool,
    tools,
} from "./middleware";
import type { Handler } from "./worker";
import { HandlerBuilder, Worker } from "./worker";

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
    tools: typeof tools;
    llm: typeof llm;
    stopWhen: typeof stopWhen;
    stepCountIs: typeof stepCountIs;
    serverGenerate: typeof serverGenerate;
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
    factory.tools = tools;
    factory.llm = llm;
    factory.stopWhen = stopWhen;
    factory.stepCountIs = stepCountIs;
    factory.serverGenerate = serverGenerate;
    factory.subAgents = subAgents;

    return factory;
}

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
