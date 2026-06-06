// The current chat thread for this page load, in memory only — nothing is
// persisted, so a refresh mints a fresh thread and the conversation resets. The
// id IS the AG-UI threadId / substructure session_id; the server owns the actual
// conversation, this just holds the active thread id.

import { useSyncExternalStore } from "react";

let activeId = "";
let initialized = false;

// Lazily mint the thread id on the client. On the server it stays "";
// useSyncExternalStore reconciles the two on hydration. Because the id isn't
// persisted, each page load starts a brand-new thread.
function ensureInit(): void {
    if (initialized || typeof window === "undefined") return;
    initialized = true;
    activeId = crypto.randomUUID();
}

function subscribe(): () => void {
    ensureInit();
    return () => {};
}

export function useSessions(): { activeId: string } {
    const id = useSyncExternalStore(
        subscribe,
        () => {
            ensureInit();
            return activeId;
        },
        () => "",
    );
    return { activeId: id };
}
