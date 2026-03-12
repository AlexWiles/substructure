/** Client for sending messages to a substructure2 runtime. */

import type { SendMessageRequest, SessionEvent } from "./types.js";

export class Client {
  readonly url: string;
  readonly tenantId: string;

  constructor(url: string, options?: { tenantId?: string }) {
    this.url = url.replace(/\/+$/, "");
    this.tenantId = options?.tenantId ?? "default";
  }

  async *send(
    agentId: string,
    message: string,
    options?: { sessionId?: string },
  ): AsyncIterable<SessionEvent> {
    const req: SendMessageRequest = {
      agent_id: agentId,
      message,
      tenant_id: this.tenantId,
      session_id: options?.sessionId,
    };

    // Strip undefined values
    const body = JSON.parse(JSON.stringify(req)) as SendMessageRequest;

    const resp = await fetch(`${this.url}/sessions/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      throw new Error(`send failed: ${resp.status} ${resp.statusText}`);
    }

    if (!resp.body) return;

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop()!;

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) {
          yield JSON.parse(trimmed) as SessionEvent;
        }
      }
    }

    // Flush remaining buffer
    const trimmed = buffer.trim();
    if (trimmed) {
      yield JSON.parse(trimmed) as SessionEvent;
    }
  }
}
