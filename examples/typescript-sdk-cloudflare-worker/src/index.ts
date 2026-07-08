// Cloudflare Worker serving the agent over HTTP, with per-session state in a Durable Object.
// State is worker-managed: each tool reaches its own store (keyed by `request.session_id`) directly.

import { agent, tool, toolLoop, worker } from "@substructure.ai/sdk";
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

interface WorkerEnv extends Env {
    AGENT_STATE: DurableObjectNamespace<AgentState>;
    SIGNING_SECRET?: string;
}

function todoTools(namespace: DurableObjectNamespace<AgentState>) {
    const addTodo = tool({
        name: "add_todo",
        description: "Add a todo item",
        input: {
            type: "object",
            properties: { title: { type: "string" } },
            required: ["title"],
        },
        execute: async (args, request) => {
            const { title } = JSON.parse(args);
            const stub = namespace.getByName(request.session_id);
            const state = await stub.getState();
            const item: Todo = { id: crypto.randomUUID().slice(0, 8), title, done: false };
            await stub.setState({ items: [...state.items, item] });
            return JSON.stringify(item);
        },
    });

    const listTodos = tool({
        name: "list_todos",
        description: "List all todos",
        execute: async (_args, request) => {
            const stub = namespace.getByName(request.session_id);
            const state = await stub.getState();
            return JSON.stringify(state.items);
        },
    });

    return [addTodo, listTodos];
}

export default {
    async fetch(request: Request, env: WorkerEnv): Promise<Response> {
        const todoAgent = agent({
            name: "todo",
            decide: toolLoop({
                llm: { model: "anthropic/claude-sonnet-4-6" },
                instructions: "Concise todo assistant. Use tools to manage the list.",
                tools: todoTools(env.AGENT_STATE),
            }),
        });

        const handler = worker([todoAgent]).fetch({ signingSecret: env.SIGNING_SECRET });

        return handler(request);
    },
};
