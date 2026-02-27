import type { BaseEvent } from "@ag-ui/client";

export type QueueItem =
  | { source: "parent"; event: BaseEvent }
  | { source: "child"; toolCallId: string; data: any };

export function createEventQueue(signal: AbortSignal) {
  const queue: QueueItem[] = [];
  let resolve: (() => void) | null = null;
  let done = false;

  function push(item: QueueItem) {
    if (done) return;
    queue.push(item);
    resolve?.();
    resolve = null;
  }

  function finish() {
    done = true;
    resolve?.();
    resolve = null;
  }

  signal.addEventListener("abort", finish, { once: true });

  const iterable: AsyncIterable<QueueItem> = {
    [Symbol.asyncIterator]() {
      return {
        async next(): Promise<IteratorResult<QueueItem>> {
          while (queue.length === 0 && !done) {
            await new Promise<void>((r) => {
              resolve = r;
            });
          }
          if (queue.length > 0) {
            return { value: queue.shift()!, done: false };
          }
          return { value: undefined as any, done: true };
        },
      };
    },
  };

  return { push, finish, iterable };
}

function delay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

export async function observeChild(
  childSessionId: string,
  toolCallId: string,
  push: (item: QueueItem) => void,
  signal: AbortSignal,
  baseUrl: string,
  headers: Record<string, string>,
) {
  let retries = 0;
  const maxRetries = 20;

  while (!signal.aborted && retries < maxRetries) {
    try {
      const resp = await fetch(
        `${baseUrl}/sessions/${childSessionId}/ag-ui/observe`,
        { signal, headers },
      );
      if (!resp.ok) {
        retries++;
        await delay(Math.min(300 * Math.pow(2, retries), 5000), signal);
        continue;
      }

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop()!;
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              push({ source: "child", toolCallId, data });
            } catch {}
          }
        }
      }
      return;
    } catch (e: any) {
      if (signal.aborted) return;
      retries++;
      await delay(Math.min(300 * Math.pow(2, retries), 5000), signal);
    }
  }
}
