// State hydration: the persisted (wire) state holds only compact *references*
// — a history id and a todos id. A hydration middleware loads the referenced
// data into rich objects the agent works with, then on the way out saves them
// back and restores the references, so only the ids ever ride the wire.
//
// This is a *type transform*, not a contribution. A state slice ADDS keys
// (`S -> S & A`); this middleware SWAPS the whole shape (`Refs -> Hydrated`)
// for everything downstream. It's a plain `MiddlewareFn<In, Out>` — the
// `.use()` transform overload replaces the builder's state type with `Out`, so
// every middleware and tool below sees the hydrated shape, fully typed.
//
// What this shows:
//   - Middleware that changes the state TYPE, not just adds keys.
//   - Compact wire state (two ids) backed by a database; the heavy data
//     (messages, todos) lives in the DB and is rehydrated each turn.
//   - The mandatory dehydrate step: whatever state we return is serialized
//     onto the wire, so we persist the data and hand back only the ids.

import Substructure, { type AgentContext, type Message, type MiddlewareFn, type Next } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const sub = new Substructure();
const { agent } = sub;

// ── Domain ──────────────────────────────────────────────────────────────────

type Todo = { id: string; title: string; done: boolean };

// What persists on the wire: just references. Tiny and cheap to ship every turn.
type Refs = { historyId: string; todosId: string };

// What the agent operates on after hydration: the loaded data.
type Hydrated = Refs & { messages: Message[]; todos: Todo[] };

// ── "Database": one JSON file per id, in two namespaces ──────────────────────
// Swap load/save for Postgres, Redis, S3, a Durable Object — the middleware and
// tools don't change.

const DB_DIR = "./hydration-db";

async function load<T>(ns: string, id: string, fallback: T): Promise<T> {
    const path = join(DB_DIR, `${ns}-${id}.json`);
    if (!existsSync(path)) return fallback;
    return JSON.parse(await readFile(path, "utf8")) as T;
}

async function save<T>(ns: string, id: string, data: T): Promise<void> {
    await mkdir(DB_DIR, { recursive: true });
    await writeFile(join(DB_DIR, `${ns}-${id}.json`), JSON.stringify(data, null, 2));
}

// ── Hydration middleware: Refs -> Hydrated ───────────────────────────────────
// A plain `MiddlewareFn<In, Out>`. `In` is what the upstream Refs slice
// provides; passing a different `Out` to `next` swaps the state type for
// everything downstream. `dehydrate` on the way out is mandatory: without it,
// the hydrated objects would be serialized onto the wire.

const hydrate: MiddlewareFn<Refs, Hydrated> = async (ctx: AgentContext<Refs>, next: Next<Hydrated>) => {
    // First turn the refs are empty — mint stable ids: todos per user (persist
    // across a user's sessions), history per session.
    const historyId = ctx.state.historyId || ctx.request.session_id;
    const todosId = ctx.state.todosId || ctx.request.identity.id;

    const hydrated: Hydrated = {
        historyId,
        todosId,
        messages: await load<Message[]>("history", historyId, []),
        todos: await load<Todo[]>("todos", todosId, []),
    };

    const res = await next({ ...ctx, state: hydrated });

    // Persist the heavy data, then hand the wire back only the references.
    const final = res.state as Hydrated;
    await save("history", historyId, final.messages);
    await save("todos", todosId, final.todos);
    return { ...res, state: { historyId, todosId } satisfies Refs };
};

// ── Tools (operate on the hydrated `todos: Todo[]`) ──────────────────────────
// The tool's state token is typed to the hydrated shape; at runtime it receives
// the live, already-hydrated state, and mutations are saved by `dehydrate`.

const todosState = agent.stateSlice<{ todos: Todo[] }>({ todos: [] });

function formatTodo(todo: Todo): string {
    return `[${todo.done ? "x" : " "}] ${todo.title} (${todo.id})`;
}

const addTodo = agent.tool({
    name: "add_todo",
    description: "Add a todo item",
    parameters: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
    state: todosState,
    execute: (args, state) => {
        const { title } = JSON.parse(args) as { title: string };
        const todo: Todo = { id: randomUUID().slice(0, 8), title, done: false };
        state.todos.push(todo);
        return formatTodo(todo);
    },
});

const listTodos = agent.tool({
    name: "list_todos",
    description: "List all todos",
    parameters: { type: "object", properties: {} },
    state: todosState,
    execute: (_args, state) => {
        return state.todos.map(formatTodo).join("\n");
    },
});

// ── Agent ────────────────────────────────────────────────────────────────────
// Refs slice (establishes the id shape) -> hydrate (Refs -> Hydrated) ->
// everything below sees Hydrated.

const todoAgent = agent({ id: "todo" })
    .use(agent.stateSlice<Refs>({ historyId: "", todosId: "" }))
    .use(hydrate)
    .use(agent.messageHistory("Concise todo assistant. Use the tools to manage the list."))
    .use(agent.tools([addTodo, listTodos]))
    .use(agent.llmToolLoop({ model: "anthropic/claude-sonnet-4-6" }));

// ── Run ───────────────────────────────────────────────────────────────────────

const embedded = await SubstructureEmbedded.create({
    agents: [todoAgent],
    db: ":memory:",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: todoAgent.agentId,
    payload: {
        type: "message",
        message: { role: "user", content: "Add 'buy groceries' and 'walk the dog', then list them" },
    },
    identity: { tenant_id: "default", id: "demo" },
});

for await (const event of embedded.stream(scope)) {
    switch (event.payload.type) {
        case "message.new":
            if (["user", "assistant"].includes(event.payload.message.role)) {
                console.log(event.payload.message.role + ":\n" + event.payload.message.content + "\n");
            }
            continue;
        case "tool.call.requested":
            console.log(event.payload.name + ":\n" + event.payload.arguments);
            continue;
        case "tool.call.completed":
            console.log(event.payload.name + " result:\n" + event.payload.result);
            continue;
    }
}

await embedded.shutdown();
