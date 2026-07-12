import { HttpAgent } from "@ag-ui/client";
import { CopilotChat, CopilotKit, useFrontendTool } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import { useEffect, useMemo, useState } from "react";
import { z } from "zod";

type Session = { token: string; url: string; agentId: string };

// A client tool: it runs in the browser, not on the worker. CopilotKit forwards
// its declaration up on every run (in the AG-UI run's `tools`); the engine layers
// it onto the agent and suspends the turn until this returns.
function TimezoneTool() {
    useFrontendTool({
        name: "get_timezone",
        description: "Get the user's IANA time zone from their browser.",
        parameters: z.object({}),
        handler: async () => ({ timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone }),
        render: ({ name, status, result }) => {
            const done = status === "complete";
            return (
                <div style={{ margin: "6px 0", border: "1px solid rgba(120,120,120,0.25)", borderRadius: 8, padding: "8px 10px", fontSize: 13 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ flex: "none", width: 7, height: 7, borderRadius: "50%", background: done ? "#22c55e" : "#f59e0b" }} />
                        <code>{name}</code>
                        {!done && <span style={{ opacity: 0.6 }}>running…</span>}
                    </div>
                    {done && <pre style={{ margin: "6px 0 0", whiteSpace: "pre-wrap", wordBreak: "break-word", opacity: 0.8 }}>{result}</pre>}
                </div>
            );
        },
    });
    return null;
}

export function Chat() {
    const [session, setSession] = useState<Session>();
    useEffect(() => {
        fetch("/token")
            .then((r) => r.json())
            .then(setSession);
    }, []);
    return session ? <ChatUI session={session} /> : <p style={{ padding: 24 }}>Connecting…</p>;
}

function ChatUI({ session }: { session: Session }) {
    // Connect the browser straight to substructure's native AG-UI endpoint — no
    // CopilotKit runtime in between. `agents__unsafe_dev_only` is CopilotKit's
    // direct-to-agent escape hatch for a self-hosted AG-UI backend.
    const agent = useMemo(
        () =>
            new HttpAgent({
                url: `${session.url}/api/client/ag-ui/agents/${session.agentId}/run`,
                headers: { Authorization: `Bearer ${session.token}` },
            }),
        [session],
    );
    return (
        <CopilotKit agents__unsafe_dev_only={{ [session.agentId]: agent }}>
            <TimezoneTool />
            <main style={{ height: "100dvh" }}>
                <CopilotChat agentId={session.agentId} labels={{ chatInputPlaceholder: "what time is it in my time zone?" }} />
            </main>
        </CopilotKit>
    );
}
