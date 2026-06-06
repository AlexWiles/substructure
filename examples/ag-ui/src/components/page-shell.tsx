import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";

import { TodoPanel } from "./todo-panel";

// The client examples, one per route, linked from the sidebar.
const CLIENTS = [
    { to: "/assistant-ui", name: "assistant-ui", note: "useAgUiRuntime" },
    { to: "/copilotkit", name: "CopilotKit", note: "v2 self-managed" },
] as const;

// Shown while a session's history is being fetched on open/switch.
export function ChatLoading() {
    return <div className="chat-loading">Loading conversation…</div>;
}

export function PageShell({ children }: { children: ReactNode }) {
    return (
        <div className="shell">
            <aside className="sidebar">
                <div className="brand">
                    <b>substructure.ai × AG-UI</b>
                </div>
                <nav className="nav">
                    {CLIENTS.map((c) => (
                        <Link
                            key={c.to}
                            to={c.to}
                            className="navlink"
                            activeProps={{ className: "active" }}
                        >
                            <b>{c.name}</b>
                            <span>{c.note}</span>
                        </Link>
                    ))}
                </nav>
                <p className="hint">
                    Each client calls the same frontend to-do tools — watch the agent drive the
                    list on the right.
                </p>
            </aside>
            <main className="pagebody">
                <div className="chatpane">{children}</div>
                <TodoPanel />
            </main>
        </div>
    );
}
