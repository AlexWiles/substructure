import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";

import { ColorPanel } from "./color-panel";

// The three client examples, one per route. The sidebar links between them;
// each is its own route, so switching mounts a fresh chat with its own token.
const CLIENTS = [
    { to: "/tanstack", name: "TanStack AI", note: "useChat · dialect: tanstack" },
    { to: "/assistant-ui", name: "assistant-ui", note: "useAgUiRuntime · dialect: spec" },
    { to: "/copilotkit", name: "CopilotKit", note: "v2 self-managed · dialect: spec" },
] as const;

export function PageShell({ children }: { children: ReactNode }) {
    return (
        <div className="shell">
            <aside className="sidebar">
                <div className="brand">
                    <b>Substructure × AG-UI</b>
                    <span>One endpoint, three clients</span>
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
