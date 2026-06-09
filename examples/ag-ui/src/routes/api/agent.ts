// The substructure worker webhook. The ENGINE posts decision requests here
// (server-to-server); this returns the worker's actions. Point the engine at
// it: `substructure start --worker-url https://<app>/api/agent`.
import Substructure from "@substructure.ai/sdk";
import { createFileRoute } from "@tanstack/react-router";

export const AGENT_ID = "todo-agent";

const sub = new Substructure();
const { agent } = sub;

// Every tool is a frontend (client-handled) tool: `handler: "client"` +
// `ctx.defer()` suspends the turn so the browser runs it against the shared
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
    .use(agent.logging())
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
        agent.llmToolLoop({
            generator: agent.serverGenerate({ model: "minimax/minimax-m3" }),
            stream: true,
        }),
    );

const worker = sub.worker({ agents: [todoAgent] });

export const handler = worker.fetchHandler({ signingSecret: process.env.SIGNING_SECRET });

export const Route = createFileRoute("/api/agent")({
    server: {
        handlers: {
            POST: ({ request }) => handler(request),
        },
    },
});
