// Pure header-parsing helpers for inbound email.

function stripAngles(s: string): string {
    return s.replace(/^<|>$/g, "").toLowerCase();
}

// RFC 5322 threading: the thread root is the first id in References,
// falling back to In-Reply-To, then this message's own Message-ID.
export function threadRoot(headers: Headers): string | null {
    const references = headers.get("References");
    if (references) {
        const first = references.trim().split(/\s+/)[0];
        if (first) return stripAngles(first);
    }
    const inReplyTo = headers.get("In-Reply-To");
    if (inReplyTo) return stripAngles(inReplyTo.trim());
    const messageId = headers.get("Message-ID");
    if (messageId) return stripAngles(messageId.trim());
    return null;
}

export function inReplyToOf(headers: Headers): string | null {
    const v = headers.get("In-Reply-To");
    return v ? stripAngles(v.trim()) : null;
}

// Same thread → same session → same agent state → message history persists.
export function sessionIdForThread(headers: Headers): string {
    return `thread:${threadRoot(headers) ?? crypto.randomUUID()}`;
}
