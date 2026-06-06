// Shared substructure setup. Server-only (never the client bundle).

import Substructure from "@substructure.ai/sdk";

export const AGENT_ID = "todo-agent";

const sub = new Substructure();
const { agent } = sub;

// Every tool here is a frontend (client-handled) tool: `handler: "client"` +
// `ctx.defer()` suspends the turn so the browser executes it against the shared
// to-do store. The matching executors live in each chat client.
const addTodo = agent.tool({
    name: "add_todo",
    description: "Add a task to the user's on-screen to-do list. Runs in the user's browser.",
    parameters: {
        type: "object",
        properties: { title: { type: "string", description: "The task text." } },
        required: ["title"],
    },
    handler: "client",
    execute: (_args: string, ctx) => ctx.defer(),
});

const toggleTodo = agent.tool({
    name: "toggle_todo",
    description:
        "Check off or reopen a task by id. Pass `done` to set it explicitly, or omit it to flip. " +
        "Use list_todos first to get ids. Runs in the user's browser.",
    parameters: {
        type: "object",
        properties: {
            id: { type: "string" },
            done: { type: "boolean" },
        },
        required: ["id"],
    },
    handler: "client",
    execute: (_args: string, ctx) => ctx.defer(),
});

const removeTodo = agent.tool({
    name: "remove_todo",
    description: "Delete a task by id. Use list_todos first to get ids. Runs in the user's browser.",
    parameters: {
        type: "object",
        properties: { id: { type: "string" } },
        required: ["id"],
    },
    handler: "client",
    execute: (_args: string, ctx) => ctx.defer(),
});

const clearCompleted = agent.tool({
    name: "clear_completed",
    description: "Remove every completed task from the list. Runs in the user's browser.",
    parameters: { type: "object", properties: {} },
    handler: "client",
    execute: (_args: string, ctx) => ctx.defer(),
});

const listTodos = agent.tool({
    name: "list_todos",
    description:
        "Read the user's current to-do list. Returns each task's id, title, and done flag. " +
        "Runs in the user's browser.",
    parameters: { type: "object", properties: {} },
    handler: "client",
    execute: (_args: string, ctx) => ctx.defer(),
});

const todoAgent = agent({ id: AGENT_ID })
    .use(
        agent.messageHistory(
            "You are a concise, friendly to-do list assistant. The user has an on-screen to-do " +
                "list you drive with tools. Use add_todo to add a task (call it once per item when " +
                "adding several). Call list_todos to see the current tasks and their ids before you " +
                "toggle or remove anything — toggle_todo and remove_todo take an id. Use toggle_todo " +
                "to check off or reopen a task, and clear_completed to drop all finished tasks. When " +
                "the user asks what's on their list, call list_todos and summarize it.",
        ),
    )
    .use(agent.tools([addTodo, toggleTodo, removeTodo, clearCompleted, listTodos]))
    .use(
        agent.llmLoop({
            request: { model: "minimax/minimax-m3" },
            stream: true,
        }),
    );

const worker = sub.worker({ agents: [todoAgent] });

export const substructureHandler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

/** Handle a decision webhook from the engine. */
export function handleAgentRequest(request: Request): Promise<Response> {
    return substructureHandler(request);
}

/** Mint a short-lived, identity-locked client token for the browser. In a real
 *  app, authenticate first and bind identity.id to that user. */
export async function mintBrowserToken(): Promise<{ token: string; substructureUrl: string; agentId: string }> {
    const backend = sub.backend.client({
        url: process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000",
        apiKey: process.env.SUBSTRUCTURE_API_KEY ?? "dev-worker-key",
    });
    const { token } = await backend.mintClientToken({
        identity: { id: "demo-user" },
        ttlSeconds: 60 * 15,
    });
    // The browser talks to the engine directly, so it needs the public URL.
    const substructureUrl =
        process.env.SUBSTRUCTURE_PUBLIC_URL ?? process.env.SUBSTRUCTURE_URL ?? "http://localhost:9000";
    return { token, substructureUrl, agentId: AGENT_ID };
}
