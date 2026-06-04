import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";

import { ColorPanel } from "./color-panel";

// The client examples, one per route, linked from the sidebar.
const CLIENTS = [
    { to: "/assistant-ui", name: "assistant-ui", note: "useAgUiRuntime" },
    { to: "/copilotkit", name: "CopilotKit", note: "v2 self-managed" },
] as const;

export function PageShell({ children }: { children: ReactNode }) {
    return (
        <div className="shell">
            <aside className="sidebar">
                <div className="brand">
                    <b>Substructure × AG-UI</b>
                    <span>One endpoint, multiple clients</span>
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
                    Each client calls the same frontend <code>set_color</code> tool — watch the
                    agent drive the mixer on the right.
                </p>
            </aside>
            <main className="pagebody">
                <div className="chatpane">{children}</div>
                <ColorPanel />
            </main>
        </div>
    );
}
