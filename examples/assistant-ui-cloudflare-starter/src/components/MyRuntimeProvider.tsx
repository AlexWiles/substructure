import { HttpAgent } from "@ag-ui/client";
import { AssistantRuntimeProvider, type ThreadMessage } from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { BrowserSession } from "@/lib/token";

type StoredThread = {
    id: string;
    messages: readonly ThreadMessage[];
};

// AG-UI runtime with a threadList adapter for multi-thread support, wired to the
// Substructure engine. The browser streams from the engine directly: the AG-UI
// runtime POSTs to /api/client/ag-ui/agents/{agentId}/run with the short-lived
// client token. Mirrors assistant-ui's with-ag-ui example, adapted to carry the
// minted Substructure session.
export function MyRuntimeProvider({ session, children }: { session: BrowserSession; children: ReactNode }) {
    const { token, substructureUrl, agentId } = session;

    // In-memory thread store. The thread id is the AG-UI threadId / Substructure
    // session_id; the engine owns each conversation, this just tracks the set so
    // onSwitchToThread can rehydrate the on-screen messages.
    const threadsRef = useRef<Map<string, StoredThread>>(new Map());
    const [currentThreadId, setCurrentThreadId] = useState<string>(() => {
        const id = crypto.randomUUID();
        threadsRef.current.set(id, { id, messages: [] });
        return id;
    });

    const agent = useMemo(
        () =>
            new HttpAgent({
                url: `${substructureUrl}/api/client/ag-ui/agents/${agentId}/run`,
                headers: { Authorization: `Bearer ${token}` },
                threadId: currentThreadId,
                // Bound fetch: HttpAgent calls `this.fetch(...)`, which throws
                // "Illegal invocation" in Firefox with an unbound reference.
                fetch: (url, init) => fetch(url, init),
            }),
        [token, substructureUrl, agentId, currentThreadId],
    );

    const threadListAdapter = useMemo(
        () => ({
            threadId: currentThreadId,
            onSwitchToNewThread: async () => {
                const newId = crypto.randomUUID();
                threadsRef.current.set(newId, { id: newId, messages: [] });
                setCurrentThreadId(newId);
            },
            onSwitchToThread: async (threadId: string) => {
                const thread = threadsRef.current.get(threadId);
                if (!thread) {
                    throw new Error(`Thread ${threadId} not found`);
                }
                setCurrentThreadId(threadId);
                return { messages: thread.messages };
            },
        }),
        [currentThreadId],
    );

    const runtime = useAgUiRuntime({
        agent,
        adapters: { threadList: threadListAdapter },
    });

    // Persist each thread's messages so switching back rehydrates them.
    useEffect(() => {
        return runtime.thread.subscribe(() => {
            threadsRef.current.set(currentThreadId, {
                id: currentThreadId,
                messages: runtime.thread.getState().messages,
            });
        });
    }, [runtime, currentThreadId]);

    return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>;
}
