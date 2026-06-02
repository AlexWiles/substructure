// Cloudflare Worker that serves the agent over HTTP, with per-session
// agent state stored in a Durable Object. Point a Substructure backend
// at this Worker's URL.

import Substructure from "@substructure.ai/sdk";
import type { AgentContext, MiddlewareFn, Next } from "@substructure.ai/sdk";
import { DurableObject } from "cloudflare:workers";

type Todo = { id: string; title: string; done: boolean };
type State = { items: Todo[] };

const INITIAL_STATE: State = { items: [] };

export class AgentState extends DurableObject {
    private state: State = INITIAL_STATE;

    constructor(ctx: DurableObjectState, env: Env) {
        super(ctx, env);
        ctx.blockConcurrencyWhile(async () => {
            const stored = await ctx.storage.get<State>("state");
            if (stored) {
                this.state = stored;
            } else {
                await ctx.storage.put("state", INITIAL_STATE);
            }
        });
    }

    async getState(): Promise<State> {
        return this.state;
    }

    async setState(state: State): Promise<void> {
        this.state = state;
        await this.ctx.storage.put("state", state);
    }
}

function durableObjectState(namespace: DurableObjectNamespace<AgentState>): MiddlewareFn<unknown, State> {
    return async (ctx: AgentContext<unknown>, next: Next<State>) => {
        const sessionId = ctx.request.session_id;
        const stub = namespace.getByName(sessionId);
        const state = await stub.getState();

        const result = await next({ ...ctx, state });

        await stub.setState(result.state as State);

        return {
            ...result,
            workerState: btoa(JSON.stringify({ ref: sessionId })),
        };
    };
}

interface WorkerEnv extends Env {
    AGENT_STATE: DurableObjectNamespace<AgentState>;
    SIGNING_SECRET?: string;
}

const sub = new Substructure();
const { agent } = sub;

const todos = agent.stateSlice<State>({ items: [] });

const addTodo = agent.tool({
    name: "add_todo",
    description: "Add a todo item",
    parameters: {
        type: "object",
        properties: { title: { type: "string" } },
        required: ["title"],
    },
    state: todos,
    execute: (args, state) => {
        const { title } = JSON.parse(args);
        const item: Todo = { id: crypto.randomUUID().slice(0, 8), title, done: false };
        state.items.push(item);
        return JSON.stringify(item);
    },
});

const listTodos = agent.tool({
    name: "list_todos",
    description: "List all todos",
    parameters: { type: "object", properties: {} },
    state: todos,
    execute: (_args, state) => JSON.stringify(state.items),
});

export default {
    async fetch(request: Request, env: WorkerEnv): Promise<Response> {
        const todoAgent = agent({ id: "todo" })
            .use(durableObjectState(env.AGENT_STATE))
            .use(agent.messageHistory("Concise todo assistant. Use tools to manage the list."))
            .use(agent.tools([addTodo, listTodos]))
            .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));

        const handler = sub.worker({ agents: [todoAgent] }).fetchHandler({ signingSecret: env.SIGNING_SECRET });

        return handler(request);
    },
};
