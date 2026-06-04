// Loads a session's history from the engine's AG-UI hydration endpoint so a
// thread can be (re)opened with its past messages. The shape returned matches
// AG-UI's message union, so it drops straight into HttpAgent's `initialMessages`.

import { useEffect, useState } from "react";

import { setSessionTitle } from "./sessions";
import type { BrowserSession } from "./token";

export type AgUiMessage = {
    id: string;
    role: "user" | "assistant" | "tool" | "system";
    content?: string;
    toolCalls?: unknown[];
    toolCallId?: string;
};

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

        const url = `${session.substructureUrl}/api/client/ag-ui/sessions/${sessionId}/messages`;
        fetch(url, { headers: { Authorization: `Bearer ${session.token}` } })
            .then((r) => (r.ok ? r.json() : { messages: [] }))
            .then((data: { messages?: AgUiMessage[] }) => {
                if (cancelled) return;
                const messages = data.messages ?? [];
                setHistory({ loading: false, messages });
                // Label the session by its opening question, once.
                const firstUser = messages.find((m) => m.role === "user");
                if (firstUser?.content) setSessionTitle(sessionId, firstUser.content);
            })
            .catch(() => {
                if (!cancelled) setHistory({ loading: false, messages: [] });
            });

        return () => {
            cancelled = true;
        };
    }, [session.substructureUrl, session.token, sessionId]);

    return history;
}
