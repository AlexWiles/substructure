// Embedded runtime: no separate server, the agent runs in-process.
// State and event log persist to a local SQLite file (`agent.db`).

import Substructure from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";
import { randomUUID } from "node:crypto";

const sub = new Substructure();
const { agent } = sub;

type Todo = { id: string; title: string; done: boolean };
const todos = agent.stateSlice<{ items: Todo[] }>({ items: [] });

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
        const item: Todo = { id: randomUUID().slice(0, 8), title, done: false };
        state.items.push(item);
        return item;
    },
});

const listTodos = agent.tool({
    name: "list_todos",
    description: "List all todos",
    parameters: { type: "object", properties: {} },
    state: todos,
    execute: (_args, state) => state.items,
});

const todoAgent = agent({ id: "todo" })
    .use(agent.jsonState())
    .use(agent.systemMessage("You are a concise todo assistant. Use tools to manage the list."))
    .use(agent.messageHistory())
    .use(agent.tools([addTodo, listTodos]))
    .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));

const embedded = await SubstructureEmbedded.create({
    agents: [todoAgent],
    db: "agent.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: todoAgent.agentId,
    payload: {
        type: "message",
        message: { role: "user", content: "Add 'buy groceries' and 'walk the dog', then list my todos" },
    },
    identity: { tenant_id: "default", id: "demo" },
});

const { data } = await embedded.turnResult(scope);
console.log(data);

await embedded.shutdown();
