export interface SseMessage {
    event?: string;
    id?: string;
    retry?: number;
    data: unknown;
}

export interface SseStream {
    readable: ReadableStream<Uint8Array>;
    writeSSE(message: SseMessage): Promise<void>;
    onAbort(listener: () => void): void;
    close(): Promise<void>;
}

export function createSseStream(): SseStream {
    const encoder = new TextEncoder();
    const transform = new TransformStream<Uint8Array, Uint8Array>();
    const writer = transform.writable.getWriter();
    const reader = transform.readable.getReader();

    let closed = false;
    let aborted = false;
    const abortListeners: Array<() => void> = [];
    const abort = () => {
        if (aborted) return;
        aborted = true;
        void reader.cancel();
        for (const listener of abortListeners) listener();
    };

    const readable = new ReadableStream<Uint8Array>({
        async pull(controller) {
            const { done, value } = await reader.read();
            if (done) controller.close();
            else controller.enqueue(value);
        },
        cancel() {
            if (!closed) abort();
        },
    });

    return {
        readable,
        async writeSSE(message) {
            try {
                await writer.write(encoder.encode(encodeSse(message)));
            } catch {
                // The peer is gone; nothing left to write to.
            }
        },
        onAbort(listener) {
            abortListeners.push(listener);
        },
        async close() {
            closed = true;
            try {
                await writer.close();
            } catch {
                // Already torn down by an abort.
            }
        },
    };
}

function encodeSse(message: SseMessage): string {
    for (const field of ["event", "id", "retry"] as const) {
        const value = message[field];
        if (value != null && /[\r\n]/.test(String(value))) {
            throw new Error(`SSE ${field} must not contain a newline`);
        }
    }
    const data = typeof message.data === "string" ? message.data : JSON.stringify(message.data);
    const lines = [
        message.event != null && `event: ${message.event}`,
        message.id != null && `id: ${message.id}`,
        message.retry != null && `retry: ${message.retry}`,
        ...data.split(/\r\n|\r|\n/).map((line) => `data: ${line}`),
    ].filter(Boolean);
    return `${lines.join("\n")}\n\n`;
}
