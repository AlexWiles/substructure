// Hybrid state: the todo list lives in its own database, keyed by user id, not
// in SDK-held state. There is no built-in tool state — each tool reaches the
// store itself (here a directory of JSON files keyed by `ctx.request.identity.id`),
// so todos persist across all of a user's sessions and never ride the wire.
//
// Swap loadTodos/saveTodos for a real client (Postgres, Durable Object, S3, ...)
// without touching the agent or the tools.

import { agent, server, tool, toolLoop } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

// ── Domain ──────────────────────────────────────────────────────────────────

type Todo = { id: string; title: string; done: boolean };
type TodoData = { items: Todo[] };

// ── "Database" (one JSON file per user) ─────────────────────────────────────

const DB_DIR = "./todo-db";

async function loadTodos(userId: string): Promise<TodoData> {
    const path = join(DB_DIR, `${userId}.json`);
    if (!existsSync(path)) return { items: [] };
    return JSON.parse(await readFile(path, "utf8")) as TodoData;
}

async function saveTodos(userId: string, data: TodoData): Promise<void> {
    await mkdir(DB_DIR, { recursive: true });
    await writeFile(join(DB_DIR, `${userId}.json`), JSON.stringify(data, null, 2));
}

// ── Tools (read and write the store directly, keyed by user) ────────────────

const addTodo = tool({
    name: "add_todo",
    description: "Add a todo item",
    parameters: {
        type: "object",
        properties: { title: { type: "string" } },
        required: ["title"],
    },
    execute: async (args, ctx) => {
        const { title } = JSON.parse(args);
        const userId = ctx.request.identity.id;
        const data = await loadTodos(userId);
        const item: Todo = { id: randomUUID().slice(0, 8), title, done: false };
        data.items.push(item);
        await saveTodos(userId, data);
        return JSON.stringify(item);
    },
});

const listTodos = tool({
    name: "list_todos",
    description: "List all todos",
    parameters: { type: "object", properties: {} },
    execute: async (_args, ctx) => JSON.stringify((await loadTodos(ctx.request.identity.id)).items),
});

// ── Agent ───────────────────────────────────────────────────────────────────

const todoAgent = agent({
    name: "todo",
    decide: toolLoop({
        model: server("anthropic/claude-sonnet-4-6"),
        instructions: "Concise todo assistant. Use tools to manage the list.",
        tools: [addTodo, listTodos],
    }),
});

// ── Run ─────────────────────────────────────────────────────────────────────

const embedded = await SubstructureEmbedded.create({
    agents: [todoAgent],
    db: "agent.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: "todo",
    payload: {
        type: "message",
        message: {
            role: "user",
            content: "Add 'buy groceries' and 'walk the dog'",
        },
    },
    identity: { tenant_id: "default", id: "demo" },
});

const { data } = await embedded.turnResult(scope);
console.log(data);

await embedded.shutdown();
