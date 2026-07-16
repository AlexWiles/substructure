import { HttpAgent } from "@ag-ui/client";
import {
    AssistantRuntimeProvider,
    AuiProvider,
    defineToolkit,
    ExportedMessageRepository,
    Tools,
    type ToolCallMessagePartProps,
    useAui,
} from "@assistant-ui/react";
import { fromAgUiMessages, useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { makeMarkdownText, Thread } from "@assistant-ui/react-ui";
import "@assistant-ui/react-ui/styles/index.css";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { z } from "zod";
import type { Message } from "../protocol";

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

// The session id lives in the URL (?session=…) so a refresh restores the chat.
function urlSessionId(): string {
    let id = new URLSearchParams(location.search).get("session");
    if (!id) {
        id = crypto.randomUUID();
        history.replaceState(null, "", `?session=${id}`);
    }
    return id;
}

export function Chat() {
    const [session, setSession] = useState<Session>();
    const [sessionId, setSessionId] = useState(urlSessionId);
    useEffect(() => {
        fetch("/token")
            .then((r) => r.json())
            .then(setSession);
    }, []);
    useEffect(() => {
        const onPop = () => setSessionId(urlSessionId());
        window.addEventListener("popstate", onPop);
        return () => window.removeEventListener("popstate", onPop);
    }, []);
    const newChat = () => {
        const id = crypto.randomUUID();
        history.pushState(null, "", `?session=${id}`);
        setSessionId(id);
    };
    if (!session) return null;
    // Keyed by session id: switching chats remounts the runtime, which reloads history.
    return <Runtime key={sessionId} session={session} sessionId={sessionId} onNewChat={newChat} />;
}

type SessionInterrupt = { interrupt_id: string; reason: string; anchor?: string | null };
type SessionView = { messages: Message[]; interrupts?: SessionInterrupt[] };

// `messages` is the active conversation, ready to fold into assistant-ui
// messages. Interrupts parking it — no anchor, or anchored on this path — ride
// as metadata on the newest assistant message, the one place the runtime looks
// for pending interrupts; the converter derives interrupt UI from it.
function toMessages({ messages, interrupts = [] }: SessionView) {
    const onPath = new Set(messages.map((m) => m.id));
    const parking = interrupts
        .filter((i) => !i.anchor || onPath.has(i.anchor))
        .map((i) => ({ id: i.interrupt_id, reason: i.reason }));

    const newestAssistant = parking.length
        ? messages.filter((m) => m.role === "assistant").at(-1)
        : undefined;

    const attached = messages
        .map((m) => {
            return m === newestAssistant
                ? { ...m, metadata: { custom: { agui: { interrupts: parking } } } }
                : m
        });

    return fromAgUiMessages(attached);
}

function Runtime({ session, sessionId, onNewChat }: { session: Session; sessionId: string; onNewChat: () => void }) {
    // threadId is the substructure session id — the engine keys the session by it.
    const agent = useMemo(
        () =>
            new HttpAgent({
                url: `${session.url}/api/client/ag-ui/agents/${session.agentId}/run`,
                headers: { Authorization: `Bearer ${session.token}` },
                threadId: sessionId,
            }),
        [session, sessionId],
    );
    // The history adapter reloads the head branch on mount; the engine already
    // records every message, so append is a no-op.
    const history = useMemo(
        () => ({
            async load() {
                const res = await fetch(`${session.url}/api/client/sessions/${sessionId}`, {
                    headers: { authorization: `Bearer ${session.token}` },
                });
                if (!res.ok) return { messages: [] }; // 404: not created yet
                return ExportedMessageRepository.fromArray(toMessages(await res.json()));
            },
            async append() { },
        }),
        [session, sessionId],
    );
    const runtime = useAgUiRuntime({ agent, adapters: { history } });
    return (
        <AssistantRuntimeProvider runtime={runtime}>
            <ClientTools>
                <main style={{ height: "100dvh", position: "relative" }}>
                    <button
                        type="button"
                        onClick={onNewChat}
                        style={{
                            position: "absolute",
                            top: 12,
                            right: 16,
                            zIndex: 10,
                            padding: "6px 12px",
                            borderRadius: 8,
                            border: "1px solid rgba(120,120,120,0.35)",
                            background: "#fff",
                            fontSize: 13,
                            cursor: "pointer",
                        }}
                    >
                        New chat
                    </button>
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
