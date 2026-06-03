import { fetchServerSentEvents, useChat } from "@tanstack/ai-react";
import { toolDefinition } from "@tanstack/ai/client";
import { useMemo, useState } from "react";

import { setColor, toHex } from "../lib/color-store";
import type { BrowserSession } from "../lib/token";

// Frontend (client-handled) tool — runs in the browser. Matches the worker's
// `handler: "client"` tool of the same name, and drives the shared color mixer.
const setColorTool = toolDefinition({
    name: "set_color",
    description: "Set the color in the on-screen color mixer (red/green/blue, 0–255).",
    inputSchema: {
        type: "object",
        properties: {
            red: { type: "integer" },
            green: { type: "integer" },
            blue: { type: "integer" },
        },
        required: ["red", "green", "blue"],
    },
}).client(async (args) => {
    const { red, green, blue } = args as { red: number; green: number; blue: number };
    const c = setColor({ r: red, g: green, b: blue }, { animate: true });
    return { red: c.r, green: c.g, blue: c.b, hex: toHex(c) };
});

const tools = [setColorTool] as const;

// TanStack AI speaks AG-UI natively. It needs the `tool-input-available` CUSTOM
// event to execute frontend tools, so it declares its dialect via forwardedProps.
export function TanStackChat({ session }: { session: BrowserSession }) {
    const { token, substructureUrl, agentId } = session;

    const connection = useMemo(
        () =>
            fetchServerSentEvents(`${substructureUrl}/api/client/ag-ui/agents/${agentId}/run`, {
                headers: { Authorization: `Bearer ${token}` },
                body: { aguiClient: "tanstack" },
            }),
        [substructureUrl, agentId, token],
    );

    const { messages, sendMessage, isLoading } = useChat({ connection, tools });
    const [input, setInput] = useState("");

    const onSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const text = input.trim();
        if (!text || isLoading) return;
        setInput("");
        void sendMessage(text);
    };

    return (
        <div className="chat">
            <div className="messages">
                {messages.map((m) => (
                    <div key={m.id} className={`msg ${m.role}`}>
                        {m.parts.map((part, i) => {
                            switch (part.type) {
                                case "text":
                                    return <span key={i}>{part.content}</span>;
                                case "tool-call": {
                                    const out = part.output as { hex?: string } | undefined;
                                    return (
                                        <em key={i} className="tool">
                                            🎨 {part.name}
                                            {out?.hex ? (
                                                <>
                                                    {" → "}
                                                    <span
                                                        className="swatchchip"
                                                        style={{ background: out.hex }}
                                                    />{" "}
                                                    {out.hex}
                                                </>
                                            ) : (
                                                " …"
                                            )}
                                        </em>
                                    );
                                }
                                default:
                                    return null;
                            }
                        })}
                    </div>
                ))}
                {isLoading && <div className="msg assistant pending">…</div>}
            </div>
            <form onSubmit={onSubmit} className="composer">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask to mix a color, or about the weather…"
                    // eslint-disable-next-line jsx-a11y/no-autofocus
                    autoFocus
                />
                <button type="submit" disabled={isLoading}>
                    Send
                </button>
            </form>
        </div>
    );
}
