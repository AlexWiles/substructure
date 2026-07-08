// assistant-ui hydrates a thread from a history adapter, NOT from the agent's
// `initialMessages` (its runtime keeps its own message list). Its AG-UI runtime
// also only folds AG-UI messages for the live MESSAGES_SNAPSHOT *event* — that
// fold isn't exported for the adapter path — so we fetch the session's snapshot
// (the shared /connect endpoint) and do the Camp A→B fold ourselves: shape the
// messages as ThreadMessageLike and hand them to
// ExportedMessageRepository.fromBranchableArray (which does the ThreadMessage
// conversion). `load()` runs once when the runtime mounts for a thread.

import { ExportedMessageRepository, type ThreadHistoryAdapter, type ThreadMessageLike } from "@assistant-ui/react";

import { fetchSessionSnapshot, type AgUiMessage } from "./history";
import type { BrowserSession } from "./token";

type ToolCallPart = {
    type: "tool-call";
    toolCallId: string;
    toolName: string;
    argsText: string;
    args?: Record<string, unknown>;
    result?: unknown;
};
type Part = ToolCallPart | { type: "text"; text: string };
type Draft =
    | { id: string; role: "user" | "system"; content: string }
    | { id: string; role: "assistant"; content: string | Part[] };

function parse(text: string | undefined): unknown {
    if (text == null) return text;
    try {
        return JSON.parse(text);
    } catch {
        return text;
    }
}

function toToolCallPart(
    tc: { id?: string; function?: { name?: string; arguments?: string } },
    fallbackId: string,
): ToolCallPart {
    const fn = tc.function ?? {};
    const argsText = fn.arguments ?? "{}";
    const parsed = parse(argsText);
    const args =
        parsed && typeof parsed === "object" && !Array.isArray(parsed)
            ? (parsed as Record<string, unknown>)
            : undefined;
    return {
        type: "tool-call",
        toolCallId: tc.id ?? fallbackId,
        toolName: fn.name ?? "tool",
        argsText,
        ...(args ? { args } : {}),
    };
}

// Fold AG-UI messages into ThreadMessageLike. Tool results merge into their
// assistant tool-call part — assistant-ui has no standalone tool messages.
function toThreadMessages(messages: AgUiMessage[]): Draft[] {
    const out: Draft[] = [];
    for (const m of messages) {
        if (m.role === "tool") {
            for (let i = out.length - 1; i >= 0; i--) {
                const prev = out[i];
                if (prev.role !== "assistant" || !Array.isArray(prev.content)) continue;
                const part = prev.content.find(
                    (p): p is ToolCallPart => p.type === "tool-call" && p.toolCallId === m.toolCallId,
                );
                if (part) {
                    part.result = parse(m.content);
                    break;
                }
            }
            continue;
        }
        if (m.role === "assistant") {
            const parts: Part[] = [];
            if (m.content) parts.push({ type: "text", text: m.content });
            const calls = (m.toolCalls ?? []) as { id?: string; function?: { name?: string; arguments?: string } }[];
            calls.forEach((tc, i) => parts.push(toToolCallPart(tc, `${m.id}-tool-${i}`)));
            out.push({ id: m.id, role: "assistant", content: parts.length ? parts : "" });
            continue;
        }
        out.push({ id: m.id, role: m.role, content: m.content ?? "" });
    }
    return out;
}

export function makeAssistantUiHistory(session: BrowserSession, sessionId: string): ThreadHistoryAdapter {
    return {
        async load() {
            const raw = await fetchSessionSnapshot(session, sessionId);
            const drafts = toThreadMessages(raw);
            // Linear thread: each message's parent is the one before it.
            // fromBranchableArray converts each ThreadMessageLike and picks the head.
            return ExportedMessageRepository.fromBranchableArray(
                drafts.map((message, i) => ({
                    // Our tool-call args are parsed JSON; cast to the lib's ThreadMessageLike.
                    message: message as ThreadMessageLike & { id: string },
                    parentId: i === 0 ? null : drafts[i - 1].id,
                })),
            );
        },
        async append() {
            // The engine persists the conversation; nothing to store client-side.
        },
    };
}
