// Tracks the user's chat sessions (AG-UI threads) in localStorage so they
// survive reloads and can be switched between. The id IS the AG-UI threadId /
// substructure session_id; the server owns the actual conversation, this just
// remembers which sessions exist and which is active.

import { useSyncExternalStore } from "react";

export type SessionMeta = { id: string; title: string; createdAt: number };
type State = { sessions: SessionMeta[]; activeId: string };

const LIST_KEY = "substructure.sessions";
const ACTIVE_KEY = "substructure.activeSession";
const DEFAULT_TITLE = "New chat";

let state: State = { sessions: [], activeId: "" };
let initialized = false;
const listeners = new Set<() => void>();

function makeSession(): SessionMeta {
    return { id: crypto.randomUUID(), title: DEFAULT_TITLE, createdAt: Date.now() };
}

// Lazily load from localStorage on the client. On the server `state` stays the
// empty default; useSyncExternalStore reconciles the two on hydration.
function ensureInit(): void {
    if (initialized || typeof window === "undefined") return;
    initialized = true;
    let sessions: SessionMeta[] = [];
    try {
        sessions = JSON.parse(localStorage.getItem(LIST_KEY) ?? "[]");
    } catch {
        sessions = [];
    }
    let activeId = localStorage.getItem(ACTIVE_KEY) ?? "";
    if (sessions.length === 0) {
        const s = makeSession();
        sessions = [s];
        activeId = s.id;
    } else if (!sessions.some((s) => s.id === activeId)) {
        activeId = sessions[0].id;
    }
    state = { sessions, activeId };
    persist();
}

function persist(): void {
    try {
        localStorage.setItem(LIST_KEY, JSON.stringify(state.sessions));
        localStorage.setItem(ACTIVE_KEY, state.activeId);
    } catch {
        // ignore quota / private-mode failures
    }
}

function commit(next: State): void {
    state = next;
    persist();
    for (const notify of listeners) notify();
}

export function newSession(): string {
    ensureInit();
    const s = makeSession();
    commit({ sessions: [s, ...state.sessions], activeId: s.id });
    return s.id;
}

export function switchSession(id: string): void {
    ensureInit();
    if (id === state.activeId) return;
    commit({ ...state, activeId: id });
}

// Name a session from its first user message — only once, while still untitled.
export function setSessionTitle(id: string, title: string): void {
    ensureInit();
    const trimmed = title.trim().slice(0, 60);
    if (!trimmed) return;
    const current = state.sessions.find((s) => s.id === id);
    if (!current || current.title !== DEFAULT_TITLE) return;
    commit({
        ...state,
        sessions: state.sessions.map((s) => (s.id === id ? { ...s, title: trimmed } : s)),
    });
}

const SERVER_SNAPSHOT: State = { sessions: [], activeId: "" };

function subscribe(cb: () => void): () => void {
    ensureInit();
    listeners.add(cb);
    return () => {
        listeners.delete(cb);
    };
}

export function useSessions(): State {
    return useSyncExternalStore(
        subscribe,
        () => {
            ensureInit();
            return state;
        },
        () => SERVER_SNAPSHOT,
    );
}
