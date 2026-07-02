// State on the wire: the todo list rides the decision envelope as `worker_state`,
// round-tripped every turn. There is no built-in tool state, so the tools are
// built per decision, closing over the live list. `toolLoop` turns them into the
// model's tool schemas and runs them; passing `state` into the loop is what it
// persists back out — `loop({ ...req, state })` — so the tools' edits ride the
// wire with no manual plumbing.

import { agent, tool, toolLoop } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";
import { randomUUID } from "node:crypto";

// ── Domain ──────────────────────────────────────────────────────────────────

type Todo = { id: string; title: string; done: boolean };
type State = { todos: Todo[] };

const formatTodo = (todo: Todo): string => `[${todo.done ? "x" : " "}] ${todo.title} (${todo.id})`;

// Built fresh each decision so `execute` closes over the live list; `toolLoop`
// runs them on `tool.execute`, and the mutations land in `state.todos`.
function todoTools(state: State) {
    return [
        tool({
            name: "add_todo",
            description: "Add a todo item",
            parameters: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
            execute: (args) => {
                const todo: Todo = { id: randomUUID().slice(0, 8), title: JSON.parse(args).title, done: false };
                state.todos.push(todo);
                return formatTodo(todo);
            },
        }),
        tool({
            name: "list_todos",
            description: "List all todos",
            parameters: { type: "object", properties: {} },
            execute: () => state.todos.map(formatTodo).join("\n") || "(no todos yet)",
        }),
    ];
}

// ── Agent: the default loop over state that rides the wire ────────────────────

const todoAgent = agent<State>({
    name: "todo",
    decide: async (req) => {
        const state: State = { todos: req.state?.todos ?? [] };
        const loop = toolLoop<State>({
            llm: { model: "anthropic/claude-sonnet-4-6" },
            instructions: "Concise todo assistant. Use the tools to manage the list.",
            tools: todoTools(state),
        });
        return loop({ ...req, state });
    },
});

// ── Run ───────────────────────────────────────────────────────────────────────

const embedded = await SubstructureEmbedded.create({
    agents: [todoAgent],
    db: ":memory:",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: "todo",
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
