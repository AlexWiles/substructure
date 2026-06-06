import { CopilotChat, CopilotKitProvider } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import { useMemo } from "react";

import { useSessionHistory } from "../lib/history";
import { HistoryHttpAgent } from "../lib/history-http-agent";
import { useSessions } from "../lib/sessions";
import { addTodo, clearCompleted, getTodos, removeTodo, toggleTodo } from "../lib/todo-store";
import type { BrowserSession } from "../lib/token";
import { ChatLoading } from "./page-shell";

// Frontend tools. The schema is declared on the worker; these handlers just
// read/write the shared to-do store.
const frontendTools = [
    {
        name: "add_todo",
        description: "Add a task to the on-screen to-do list.",
        handler: async (args: Record<string, unknown>) => addTodo(String(args.title ?? "")),
    },
    {
        name: "toggle_todo",
        description: "Check off or reopen a task by id (omit `done` to flip).",
        handler: async (args: Record<string, unknown>) => {
            const t = toggleTodo(String(args.id), args.done as boolean | undefined);
            return t ?? { error: `No task with id ${String(args.id)}` };
        },
    },
    {
        name: "remove_todo",
        description: "Delete a task by id.",
        handler: async (args: Record<string, unknown>) => ({
            id: String(args.id),
            removed: removeTodo(String(args.id)),
        }),
    },
    {
        name: "clear_completed",
        description: "Remove every completed task.",
        handler: async () => ({ removed: clearCompleted() }),
    },
    {
        name: "list_todos",
        description: "Read the current to-do list.",
        handler: async () => ({ todos: getTodos() }),
    },
];

// Mount the chat keyed on the active session so switching gives a fresh provider.
// History is restored by the agent's connect() (see HistoryHttpAgent), not seeded
// here; useSessionHistory runs only to title the session from its first message.
export function CopilotKitChat({ session }: { session: BrowserSession }) {
    const { activeId } = useSessions();
    const { loading } = useSessionHistory(session, activeId);
    if (!activeId || loading) return <ChatLoading />;
    return <CopilotKitThread key={activeId} session={session} sessionId={activeId} />;
}

// CopilotKit v2 connects straight to the AG-UI endpoint via a self-managed
// HttpAgent (re-exported from @copilotkit/react-core/v2) — no CopilotRuntime broker.
function CopilotKitThread({ session, sessionId }: { session: BrowserSession; sessionId: string }) {
    const { token, substructureUrl, agentId } = session;
    const base = `${substructureUrl}/api/client/ag-ui/agents/${agentId}`;

    const agent = useMemo(
        () =>
            new HistoryHttpAgent({
                url: `${base}/run`,
                // connect() restores prior history on mount; it carries its own
                // auth since CopilotKit only injects provider headers on run().
                connectUrl: `${base}/connect`,
                connectHeaders: { Authorization: `Bearer ${token}` },
                threadId: sessionId,
            }),
        [base, sessionId, token],
    );

    // Auth goes on the provider, not the agent: CopilotKit overwrites a managed
    // agent's headers from the provider config.
    //
    // threadId MUST be set explicitly on CopilotChat: it's both what the agent
    // runs under (so the engine persists the session under our id) AND what makes
    // CopilotChat call the agent's connect() on mount to restore history. Without
    // it, CopilotKit mints a random thread id and never connects — no history.
    return (
        <CopilotKitProvider
            headers={{ Authorization: `Bearer ${token}` }}
            selfManagedAgents={{ [agentId]: agent }}
            frontendTools={frontendTools}
        >
            <div className="copilot-pane">
                <CopilotChat agentId={agentId} threadId={sessionId} />
            </div>
        </CopilotKitProvider>
    );
}
