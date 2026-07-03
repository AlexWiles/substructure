// The Substructure worker webhook: the engine posts decision requests here.
// Point the engine at it: `substructure start --worker-url https://<app>/api/agent`.
import { agent, tool, toolLoop, worker } from "@substructure.ai/sdk";
import { createFileRoute } from "@tanstack/react-router";

export const AGENT_ID = "todo-agent";

// Every tool is a client-handled tool: `handler: "client"`, no `execute` — the engine suspends
// the turn and the browser runs it. Executors live in each chat client.
const addTodo = tool({
    name: "add_todo",
    description: "Add a task to the user's on-screen to-do list. Runs in the user's browser.",
    parameters: {
        type: "object",
        properties: { title: { type: "string", description: "The task text." } },
        required: ["title"],
    },
    handler: "client",
});

const toggleTodo = tool({
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
});

const removeTodo = tool({
    name: "remove_todo",
    description: "Delete a task by id. Use list_todos first to get ids. Runs in the user's browser.",
    parameters: {
        type: "object",
        properties: { id: { type: "string" } },
        required: ["id"],
    },
    handler: "client",
});

const clearCompleted = tool({
    name: "clear_completed",
    description: "Remove every completed task from the list. Runs in the user's browser.",
    parameters: { type: "object", properties: {} },
    handler: "client",
});

const listTodos = tool({
    name: "list_todos",
    description:
        "Read the user's current to-do list. Returns each task's id, title, and done flag. " +
        "Runs in the user's browser.",
    parameters: { type: "object", properties: {} },
    handler: "client",
});

const todoAgent = agent({
    name: AGENT_ID,
    decide: toolLoop({
        llm: { model: "minimax/minimax-m3", stream: true },
        instructions:
            "You are a concise, friendly to-do list assistant. The user has an on-screen to-do " +
            "list you drive with tools. Use add_todo to add a task (call it once per item when " +
            "adding several). Call list_todos to see the current tasks and their ids before you " +
            "toggle or remove anything — toggle_todo and remove_todo take an id. Use toggle_todo " +
            "to check off or reopen a task, and clear_completed to drop all finished tasks. When " +
            "the user asks what's on their list, call list_todos and summarize it.",
        tools: [addTodo, toggleTodo, removeTodo, clearCompleted, listTodos],
    }),
});

export const handler = worker([todoAgent]).fetch({ signingSecret: process.env.SIGNING_SECRET });

export const Route = createFileRoute("/api/agent")({
    server: {
        handlers: {
            POST: ({ request }) => handler(request),
        },
    },
});
