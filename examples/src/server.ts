// Substructure SDK – HTTP Worker Server Example
// This example shows how to run agents as an HTTP worker that connects to a
// Substructure backend server, rather than using the embedded runtime.
// Requires a running Substructure backend at http://localhost:8080 started
// with `--worker-url http://localhost:4444` so it pushes decisions here.

import Substructure from "@substructure.ai/sdk";
import { randomUUID } from "crypto";

const sub = new Substructure();
const { agent } = sub;

// --- Define a simple agent (same as basic.ts) ---

const todoState = agent.stateSlice<{ todos: { id: string; title: string; done: boolean }[] }>({ todos: [] });

const addTask = agent.tool({
    name: "add_task",
    description: "Add a new task to the todo list",
    parameters: {
        type: "object",
        properties: { title: { type: "string", description: "Task title" } },
        required: ["title"],
    },
    state: todoState,
    execute: (args, state) => {
        const { title } = JSON.parse(args);
        const task = { id: randomUUID().slice(0, 8), title, done: false };
        state.todos.push(task);
        return task;
    },
});

const todoAgent = agent({ id: "todo-agent" })
    .use(agent.jsonState())
    .use(agent.systemMessage("You are a todo list assistant. Use the provided tools to manage tasks. Be concise."))
    .use(agent.messageHistory())
    .use(agent.tools([addTask]))
    .use(
        agent.llmLoop({
            request: { model: "anthropic/claude-sonnet-4" },
            llm_client: "openrouter",
        }),
    );

// --- Start an HTTP worker server ---

const worker = sub.worker({ agents: [todoAgent] });
const WORKER_PORT = 4444;
const server = Bun.serve({ port: WORKER_PORT, fetch: worker.fetchHandler() });
console.log(`Worker listening on http://localhost:${WORKER_PORT}`);

// --- Mint a client token and submit via the frontend client ---

const backend = sub.backend.client({
    url: "http://localhost:8080",
    apiKey: "dev-worker-key",
});

const clientToken = (
    await backend.mintClientToken({
        tenantId: "default",
        sub: "frontend-user",
        ttlSeconds: 600,
    })
).token;

const frontend = sub.frontend.client({
    url: "http://localhost:8080",
    token: clientToken,
});

const stream = frontend.submit({
    agentId: "todo-agent",
    payload: {
        type: "message",
        message: { role: "user", content: "Add buy groceries and walk the dog to my list" },
    },
    sessionId: randomUUID(),
    turnId: randomUUID(),
});

for await (const event of stream) {
    if (
        event.payload.type === "message.new" &&
        event.payload.message.role === "assistant" &&
        event.payload.message.content
    ) {
        console.log(event.payload.message.content);
    }
}

server.stop();
