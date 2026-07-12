import { HttpAgent } from "@ag-ui/client";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { makeMarkdownText, Thread } from "@assistant-ui/react-ui";
import "@assistant-ui/react-ui/styles/index.css";
import { useEffect, useMemo, useState } from "react";

const MarkdownText = makeMarkdownText();

// What GET /token hands the browser: a client token and where to stream from.
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
    // assistant-ui speaks AG-UI to substructure's native endpoint — no translation layer.
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
            <main style={{ height: "100dvh" }}>
                <Thread assistantMessage={{ components: { Text: MarkdownText } }} />
            </main>
        </AssistantRuntimeProvider>
    );
}
