import { HttpAgent } from "@ag-ui/client";
import {
    AssistantRuntimeProvider,
    makeAssistantTool,
    makeAssistantToolUI,
    type ToolCallMessagePartProps,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { Thread } from "@assistant-ui/react-ui";
import "@assistant-ui/react-ui/styles/index.css";
import { useMemo } from "react";

import { makeAssistantUiHistory } from "../lib/assistant-ui-history";
import { useSessions } from "../lib/sessions";
import { addTodo, clearCompleted, getTodos, removeTodo, type Todo, toggleTodo } from "../lib/todo-store";
import type { BrowserSession } from "../lib/token";
import { MarkdownText } from "./markdown-text";
import { ChatLoading } from "./page-shell";

type TodoArgs = { title?: string; id?: string; done?: boolean };
type TodoResult = Record<string, unknown>;

// One-line summary of a to-do tool call for its inline card, derived from the
// result (or the streaming args before the result lands).
function summarize(toolName: string, args: TodoArgs, result?: TodoResult): string | undefined {
    switch (toolName) {
        case "add_todo":
            return (result as Todo | undefined)?.title ?? args.title;
        case "toggle_todo": {
            const r = result as Todo | undefined;
            return r ? `${r.done ? "✓" : "○"} ${r.title}` : args.id;
        }
        case "remove_todo":
            return args.id ? `removed ${args.id}` : "removed";
        case "clear_completed": {
            const n = (result as { removed?: number } | undefined)?.removed;
            return typeof n === "number" ? `${n} cleared` : undefined;
        }
        case "list_todos": {
            const n = (result as { todos?: Todo[] } | undefined)?.todos?.length;
            return typeof n === "number" ? `${n} ${n === 1 ? "task" : "tasks"}` : undefined;
        }
        default:
            return undefined;
    }
}

// Shared inline card for every to-do tool call — same look live and rehydrated
// from history. Without a registered UI a restored call falls back to
// assistant-ui's generic JSON card.
function todoCard({ toolName, args, result, status }: ToolCallMessagePartProps<TodoArgs, TodoResult>) {
    const label = summarize(toolName, args ?? {}, result);
    return (
        <div className="toolcard">
            <span className="toolcard-name">{toolName}</span>
            {label ? <span className="toolcard-status">{label}</span> : null}
            {status.type !== "complete" ? <span className="toolcard-status">…</span> : null}
        </div>
    );
}

const TODO_TOOL_NAMES = ["add_todo", "toggle_todo", "remove_todo", "clear_completed", "list_todos"] as const;
const TodoToolUIs = TODO_TOOL_NAMES.map((toolName) => makeAssistantToolUI<TodoArgs, TodoResult>({ toolName, render: todoCard }));

// The execute() delay works around an upstream race: assistant-ui resumes a
// frontend tool only when no run is in flight, but a fast tool can resolve
// before the originating run's SSE stream closes, dropping the resume and
// hanging the chat. The delay lets the run close first. Tracked upstream in
// @assistant-ui/react-ag-ui.
const settle = () => new Promise((r) => setTimeout(r, 400));

const AddTodo = makeAssistantTool({
    toolName: "add_todo",
    type: "frontend",
    description: "Add a task to the on-screen to-do list.",
    parameters: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
    execute: async (args) => {
        await settle();
        return addTodo(String((args as TodoArgs).title ?? ""));
    },
});

const ToggleTodo = makeAssistantTool({
    toolName: "toggle_todo",
    type: "frontend",
    description: "Check off or reopen a task by id (omit `done` to flip).",
    parameters: {
        type: "object",
        properties: { id: { type: "string" }, done: { type: "boolean" } },
        required: ["id"],
    },
    execute: async (args) => {
        await settle();
        const { id, done } = args as TodoArgs;
        return toggleTodo(String(id), done) ?? { error: `No task with id ${String(id)}` };
    },
});

const RemoveTodo = makeAssistantTool({
    toolName: "remove_todo",
    type: "frontend",
    description: "Delete a task by id.",
    parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    execute: async (args) => {
        await settle();
        const { id } = args as TodoArgs;
        return { id: String(id), removed: removeTodo(String(id)) };
    },
});

const ClearCompleted = makeAssistantTool({
    toolName: "clear_completed",
    type: "frontend",
    description: "Remove every completed task.",
    parameters: { type: "object", properties: {} },
    execute: async () => {
        await settle();
        return { removed: clearCompleted() };
    },
});

const ListTodos = makeAssistantTool({
    toolName: "list_todos",
    type: "frontend",
    description: "Read the current to-do list.",
    parameters: { type: "object", properties: {} },
    execute: async () => {
        await settle();
        return { todos: getTodos() };
    },
});

// Mount the thread keyed on the active session so switching gives a fresh
// runtime. History is loaded by assistant-ui's history adapter (it ignores the
// agent's initialMessages); the agent's threadId targets the same session.
export function AssistantUiChat({ session }: { session: BrowserSession }) {
    const { activeId } = useSessions();
    if (!activeId) return <ChatLoading />;
    return <AssistantUiThread key={activeId} session={session} sessionId={activeId} />;
}

function AssistantUiThread({ session, sessionId }: { session: BrowserSession; sessionId: string }) {
    const { token, substructureUrl, agentId } = session;

    const agent = useMemo(
        () =>
            new HttpAgent({
                url: `${substructureUrl}/api/client/ag-ui/agents/${agentId}/run`,
                headers: { Authorization: `Bearer ${token}` },
                threadId: sessionId,
                // Bound fetch — HttpAgent calls `this.fetch(...)`, which throws
                // "Illegal invocation" in Firefox with an unbound reference.
                fetch: (url, init) => fetch(url, init),
            }),
        [token, substructureUrl, agentId, sessionId],
    );

    const adapters = useMemo(() => ({ history: makeAssistantUiHistory(session, sessionId) }), [session, sessionId]);

    const runtime = useAgUiRuntime({ agent, adapters });
    return (
        <AssistantRuntimeProvider runtime={runtime}>
            <AddTodo />
            <ToggleTodo />
            <RemoveTodo />
            <ClearCompleted />
            <ListTodos />
            {TodoToolUIs.map((ToolUI, i) => (
                <ToolUI key={TODO_TOOL_NAMES[i]} />
            ))}
            <Thread assistantMessage={{ components: { Text: MarkdownText } }} />
        </AssistantRuntimeProvider>
    );
}
