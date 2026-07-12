import { HttpAgent } from "@ag-ui/client";
import {
    AssistantRuntimeProvider,
    AuiProvider,
    defineToolkit,
    Tools,
    type ToolCallMessagePartProps,
    useAui,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { makeMarkdownText, Thread } from "@assistant-ui/react-ui";
import "@assistant-ui/react-ui/styles/index.css";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { z } from "zod";

const MarkdownText = makeMarkdownText();

function ToolFallback({ toolName, argsText, result, isError, status }: ToolCallMessagePartProps) {
    const done = status.type === "complete" || status.type === "incomplete";
    const args = argsText && argsText !== "{}" ? `(${argsText})` : "";
    return (
        <div style={{ margin: "6px 0", border: "1px solid rgba(120,120,120,0.25)", borderRadius: 8, padding: "8px 10px", fontSize: 13 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ flex: "none", width: 7, height: 7, borderRadius: "50%", background: isError ? "#ef4444" : done ? "#22c55e" : "#f59e0b" }} />
                <code>{toolName}{args}</code>
                {!done && <span style={{ opacity: 0.6 }}>running…</span>}
            </div>
            {result !== undefined && (
                <pre style={{ margin: "6px 0 0", whiteSpace: "pre-wrap", wordBreak: "break-word", opacity: 0.8 }}>
                    {typeof result === "string" ? result : JSON.stringify(result, null, 2)}
                </pre>
            )}
        </div>
    );
}

// A client tool: it runs in the browser, not on the worker. The worker declares
// it with handler:"client"; the engine suspends the turn until this returns.
const toolkit = defineToolkit({
    get_timezone: {
        type: "frontend",
        description: "Get the user's IANA time zone from their browser.",
        parameters: z.object({}),
        execute: () => ({ timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone }),
        render: ToolFallback,
    },
});

type Session = { token: string; url: string; agentId: string };

export function Chat() {
    const [session, setSession] = useState<Session>();
    useEffect(() => {
        fetch("/token")
            .then((r) => r.json())
            .then(setSession);
    }, []);
    return session ? <Runtime session={session} /> : <p style={{ padding: 24 }}>Connecting…</p>;
}

function Runtime({ session }: { session: Session }) {
    const agent = useMemo(
        () =>
            new HttpAgent({
                url: `${session.url}/api/client/ag-ui/agents/${session.agentId}/run`,
                headers: { Authorization: `Bearer ${session.token}` },
            }),
        [session],
    );
    const runtime = useAgUiRuntime({ agent });
    return (
        <AssistantRuntimeProvider runtime={runtime}>
            <ClientTools>
                <main style={{ height: "100dvh" }}>
                    <Thread assistantMessage={{ components: { Text: MarkdownText, ToolFallback } }} />
                </main>
            </ClientTools>
        </AssistantRuntimeProvider>
    );
}

// useAui must run inside the runtime provider so it registers the toolkit's
// tools into the active runtime; AuiProvider then exposes them to the thread.
function ClientTools({ children }: { children: ReactNode }) {
    const aui = useAui({ tools: Tools({ toolkit }) });
    return <AuiProvider value={aui}>{children}</AuiProvider>;
}
