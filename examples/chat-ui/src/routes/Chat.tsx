import { useState, useEffect } from "react";
import { CopilotKit, useRenderToolCall } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8080";
const AGENT_NAME = import.meta.env.VITE_AGENT_NAME || "default";

function ToolCallRenderer() {
    useRenderToolCall({
        name: "*",
        render: ({ args, status, name }: { args: Record<string, unknown>; status: string; name: string }) => {
            const label = name.replace(/_/g, " ");
            const detail = args.message ?? args.query ?? args.input;
            return (
                <div style={{
                    padding: "0.5rem 0.75rem",
                    margin: "0.25rem 0",
                    borderRadius: "6px",
                    background: "#f4f4f5",
                    fontSize: "0.85rem",
                    color: "#52525b",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                }}>
                    {status !== "complete" && (
                        <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", border: "2px solid #a1a1aa", borderTopColor: "transparent" }} />
                    )}
                    <span>
                        <strong>{label}</strong>
                        {detail ? `: ${detail}` : ""}
                        {status === "complete" ? " — done" : ""}
                    </span>
                </div>
            );
        },
    });
    return null;
}

export function Chat() {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetch(`${BACKEND_URL}/sessions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ agent: AGENT_NAME }),
        })
            .then((r) => {
                if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
                return r.json();
            })
            .then((data) => setSessionId(data.session_id))
            .catch((e) => setError(e.message));
    }, []);

    if (error) {
        return (
            <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
                <p>Failed to create session: {error}</p>
                <p>
                    Make sure the backend is running:
                    <br />
                    <code>cargo run --features http -- serve --config config.toml</code>
                </p>
            </div>
        );
    }

    if (!sessionId) {
        return (
            <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
                Creating session...
            </div>
        );
    }

    return (
        <CopilotKit
            runtimeUrl={`/api/copilotkit?sessionId=${sessionId}`}
            agent="default"
        >
            <ToolCallRenderer />
            <div style={{ height: "100vh" }}>
                <CopilotChat
                    labels={{ title: "Chat", initial: "Send a message to get started." }}
                />
            </div>
        </CopilotKit>
    );
}
