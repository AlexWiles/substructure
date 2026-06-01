// Hybrid state: most state rides the wire as base64 JSON (via jsonState),
// but the todo list lives in its own database, keyed by user id. The
// todoSlice middleware loads `todos` from the DB on the way in and saves
// on the way out, then empties the field so it doesn't ride the wire.
// Tools that opt into the slice see `state.todos` as if it were ordinary
// in-memory data, and todos persist across all of a user's sessions.
//
// The "database" here is a directory of JSON files keyed by user id, so
// the example has no infra to set up. Swap loadTodos/saveTodos for a real
// client (Postgres, Durable Object, S3, ...) without touching the
// middleware or the tools.

import Substructure, { middleware } from "@substructure.ai/sdk";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const sub = new Substructure();
const { agent } = sub;

// ── Domain ──────────────────────────────────────────────────────────────────

type Todo = { id: string; title: string; done: boolean };
type TodoData = { items: Todo[] };

// ── "Database" (one JSON file per user) ─────────────────────────────────────

const DB_DIR = "./todo-db";

async function loadTodos(userId: string): Promise<TodoData | null> {
    const path = join(DB_DIR, `${userId}.json`);
    if (!existsSync(path)) return null;
    return JSON.parse(await readFile(path, "utf8")) as TodoData;
}

async function saveTodos(userId: string, data: TodoData): Promise<void> {
    await mkdir(DB_DIR, { recursive: true });
    await writeFile(join(DB_DIR, `${userId}.json`), JSON.stringify(data, null, 2));
}

// ── Slice with built-in hydration ───────────────────────────────────────────
// One middleware that both contributes the typed `todos` slice and persists
// it to the database keyed by user id. On the way in we load; on the way
// out we save, then empty the field so the wire stays small. The DB is the
// source of truth and the key is fresh from identity on every decision, so
// no bookkeeping has to live in state.

const todoSlice = middleware<{ todos: TodoData }>({
    state: { todos: { items: [] } },
    handler: async (ctx, next) => {
        const userId = ctx.request.identity.id;
        ctx.state.todos = (await loadTodos(userId)) ?? { items: [] };

        const res = await next(ctx);

        await saveTodos(userId, ctx.state.todos);
        ctx.state.todos = { items: [] }; // DB has the items; don't ship them again
        return res;
    },
});

// ── Tools (operate on the slice as if it lived in memory) ───────────────────

const addTodo = agent.tool({
    name: "add_todo",
    description: "Add a todo item",
    parameters: {
        type: "object",
        properties: { title: { type: "string" } },
        required: ["title"],
    },
    state: todoSlice,
    execute: (args, state) => {
        const { title } = JSON.parse(args);
        const item: Todo = { id: randomUUID().slice(0, 8), title, done: false };
        state.todos.items.push(item);
        return item;
    },
});

const listTodos = agent.tool({
    name: "list_todos",
    description: "List all todos",
    parameters: { type: "object", properties: {} },
    state: todoSlice,
    execute: (_args, state) => state.todos.items,
});

// ── Agent ───────────────────────────────────────────────────────────────────

const todoAgent = agent({ id: "todo" })
    .use(agent.jsonState()) // request.worker_state <-> ctx.state (everything below)
    .use(todoSlice) // contributes + hydrates `todos`
    .use(agent.systemMessage("Concise todo assistant. Use tools to manage the list."))
    .use(agent.messageHistory()) // wire-backed (rides along in jsonState)
    .use(agent.tools([addTodo, listTodos]))
    .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));

// ── Run ─────────────────────────────────────────────────────────────────────

const embedded = await sub.embedded({
    agents: [todoAgent],
    db: "agent.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: todoAgent.agentId,
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
