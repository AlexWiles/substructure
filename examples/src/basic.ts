// Substructure SDK – Todo List Agent Example
// This example builds a simple conversational todo-list agent that persists
// state across turns using Substructure's embedded runtime.

import Substructure, { type RunStream } from "@substructure.ai/sdk";
import { randomUUID } from "crypto";

// Create a Substructure instance – the entry point for building agents.
const sub = new Substructure();
const { agent } = sub;

// Define the shape of our agent's state. A state slice is a typed, mutable
// piece of data that tools can read and write during execution.
type Todo = { id: string; title: string; done: boolean };
type TodoList = { todos: Todo[] };

const todoState = agent.stateSlice<TodoList>({ todos: [] });

// --- Tools ---
// Tools give the LLM actions it can take. Each tool declares a JSON Schema for
// its parameters, binds to a state slice, and provides an execute function.
// The execute function receives the raw JSON args string and the current state.

const addTask = agent.tool({
    name: "add_task",
    description: "Add a new task to the todo list",
    parameters: {
        type: "object",
        properties: {
            title: { type: "string", description: "Task title" },
        },
        required: ["title"],
    },
    state: todoState,
    execute: (args, state) => {
        const { title } = JSON.parse(args);
        const task: Todo = { id: randomUUID().slice(0, 8), title, done: false };
        state.todos.push(task);
        return task;
    },
});

const listTasks = agent.tool({
    name: "list_tasks",
    description: "List all tasks on the todo list",
    parameters: { type: "object", properties: {} },
    state: todoState,
    execute: (_args, state) => ({ tasks: state.todos, total: state.todos.length }),
});

const completeTask = agent.tool({
    name: "complete_task",
    description: "Mark a task as done by its id",
    parameters: {
        type: "object",
        properties: {
            id: { type: "string", description: "Task id to complete" },
        },
        required: ["id"],
    },
    state: todoState,
    execute: (args, state) => {
        const { id } = JSON.parse(args);
        const task = state.todos.find((t) => t.id === id);
        if (!task) return { error: `Task "${id}" not found` };
        task.done = true;
        return task;
    },
});

// --- Agent Definition ---
// Compose the agent by chaining middleware with `.use()`. Middleware runs in
// order for each turn and handles concerns like state persistence, message
// history, tool registration, and the LLM request/response loop.

const todoAgent = agent({ id: "todo-agent" })
    // Log all agent req/responses
    .use(agent.logging())
    // Serialize/deserialize state as JSON between turns
    .use(agent.jsonState())
    .use(agent.systemMessage("You are a todo list assistant. Use the provided tools to manage tasks. Be concise."))
    // Automatically track conversation history in state
    .use(agent.messageHistory())
    // Setup tool execution
    .use(agent.tools([addTask, listTasks, completeTask]))
    // Drives the loop, calling the LLM and requesting tool execution.
    .use(
        agent.llmLoop({
            request: { model: "anthropic/claude-sonnet-4" },
            llm_client: "openrouter",
        }),
    );

const embedded = await sub.embedded({
    agents: [todoAgent],
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const sessionId = randomUUID();
const identity = { tenant_id: "default", id: "example-user" };

async function turn(message: string) {
    console.log(`\n> ${message}`);
    const stream = embedded.submitAndListen({
        agentId: todoAgent.agentId,
        payload: { type: "message", message: { role: "user", content: message } },
        sessionId,
        identity,
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
}

await turn("Add buy groceries and walk my dog to my list");
await turn("What's on my list?");
await turn("Mark buy groceries as done");
await turn("What's on my list?");

await embedded.shutdown();
