// Loads a session's history from the engine's AG-UI connect endpoint, the single
// hydration path shared by both chat clients. `/connect` streams
// RUN_STARTED → MESSAGES_SNAPSHOT → RUN_FINISHED; we read the snapshot's message
// list out of that (small, finite) SSE response.

import { useEffect, useState } from "react";

import type { BrowserSession } from "./token";

export type AgUiMessage = {
    id: string;
    role: "user" | "assistant" | "tool" | "system";
    content?: string;
    toolCalls?: unknown[];
    toolCallId?: string;
};

/** Fetch a session's history as the AG-UI message list the engine persists. */
export async function fetchSessionSnapshot(session: BrowserSession, sessionId: string): Promise<AgUiMessage[]> {
    if (!sessionId) return [];
    try {
        const res = await fetch(`${session.substructureUrl}/api/client/ag-ui/agents/${session.agentId}/connect`, {
            method: "POST",
            headers: { Authorization: `Bearer ${session.token}`, "Content-Type": "application/json" },
            // /connect reads only threadId; runId just labels the synthetic snapshot run.
            body: JSON.stringify({ threadId: sessionId, runId: "snapshot", messages: [] }),
        });
        if (!res.ok) return [];
        return snapshotMessages(await res.text());
    } catch {
        return [];
    }
}

// Pull the MESSAGES_SNAPSHOT payload out of the connect SSE. Each frame is a
// `data: {json}` line; we want the one whose type is MESSAGES_SNAPSHOT (and skip
// the `event:`/keep-alive lines around it).
function snapshotMessages(sse: string): AgUiMessage[] {
    for (const line of sse.split("\n")) {
        if (!line.startsWith("data:")) continue;
        try {
            const ev = JSON.parse(line.slice(5).trim());
            if (ev?.type === "MESSAGES_SNAPSHOT") return ev.messages ?? [];
        } catch {
            // non-JSON / partial line — skip
        }
    }
    return [];
}

type History = { loading: boolean; messages: AgUiMessage[] };

export function useSessionHistory(session: BrowserSession, sessionId: string): History {
    const [history, setHistory] = useState<History>({ loading: true, messages: [] });

    useEffect(() => {
        if (!sessionId) {
            setHistory({ loading: true, messages: [] });
            return;
        }
        let cancelled = false;
        setHistory({ loading: true, messages: [] });

        fetchSessionSnapshot(session, sessionId).then((messages) => {
            if (cancelled) return;
            setHistory({ loading: false, messages });
        });

        return () => {
            cancelled = true;
        };
    }, [session.substructureUrl, session.token, session.agentId, sessionId]);

    return history;
}
