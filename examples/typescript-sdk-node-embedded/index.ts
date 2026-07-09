// Embedded runtime: no separate server, the agent runs in-process.
// State and event log persist to a local SQLite file (`agent.db`).

import { agent, tool, toolLoop } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";
import { randomUUID } from "node:crypto";

type Todo = { id: string; title: string; done: boolean };
const todos: { items: Todo[] } = { items: [] };

const addTodo = tool({
    name: "add_todo",
    description: "Add a todo item",
    parameters: {
        type: "object",
        properties: { title: { type: "string" } },
        required: ["title"],
    },
    execute: (args) => {
        const { title } = JSON.parse(args);
        const item: Todo = { id: randomUUID().slice(0, 8), title, done: false };
        todos.items.push(item);
        return JSON.stringify(item);
    },
});

const listTodos = tool({
    name: "list_todos",
    description: "List all todos",
    parameters: { type: "object", properties: {} },
    execute: () => JSON.stringify(todos.items),
});

const todoAgent = agent({
    name: "todo",
    decide: toolLoop({
        llm: { model: "anthropic/claude-sonnet-4-6" },
        instructions: "You are a concise todo assistant. Use tools to manage the list.",
        tools: [addTodo, listTodos],
    }),
});

const embedded = await SubstructureEmbedded.create({
    agents: [todoAgent],
    db: "agent.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: "todo",
    payload: {
        type: "client.message",
        message: { role: "user", content: "Add 'buy groceries' and 'walk the dog', then list my todos" },
    },
    identity: { tenant_id: "default", id: "demo" },
});

const { data } = await embedded.turnResult(scope);
console.log(data);

await embedded.shutdown();
