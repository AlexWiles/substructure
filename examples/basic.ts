import Substructure, { type RunStream } from "@substructure.ai/sdk";
import { randomUUID } from "crypto";

const sub = new Substructure();
const { agent } = sub;

type Todo = { id: string; title: string; done: boolean };
type TodoList = { todos: Todo[] };

const todoState = agent.stateSlice<TodoList>({ todos: [] });

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

const todoAgent = agent({ id: "todo-agent" })
    .use(agent.logging())
    .use(agent.jsonState())
    .use(agent.systemMessage("You are a todo list assistant. Use the provided tools to manage tasks. Be concise."))
    .use(agent.messageHistory())
    .use(agent.tools([addTask, listTasks, completeTask]))
    .use(
        agent.llmLoop({
            request: { model: "anthropic/claude-sonnet-4" },
            llm_client: "openrouter",
        }),
    );

const embedded = await sub.embedded({
    agents: [todoAgent],
    db: ":memory:",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const sessionId = randomUUID();
const auth = { tenant_id: "default", sub: "example-user" };

async function turn(message: string) {
    console.log(`\n> ${message}`);
    const stream = embedded.submit({
        agentId: todoAgent.agentId,
        payload: { type: "message", message: { role: "user", content: message } },
        sessionId,
        auth,
        turnId: randomUUID(),
    });
    for await (const event of stream) {
        if (event.payload.type === "message.new" && event.payload.message.role === "assistant") {
            console.log(event.payload.message.content);
        }
    }
}

await turn("Add buy groceries and walk my dog to my list");
await turn("What's on my list?");
await turn("Mark buy groceries as done");
await turn("What's on my list?");

await embedded.shutdown();
