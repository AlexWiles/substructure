import type {
  Event,
  SendMessageRequest,
  SubmitRequest,
  SubmitResponse,
  RegisterRequest,
  RegisterResponse,
} from "./types";

export interface ClientOptions {
  baseUrl: string;
}

export class Client {
  private baseUrl: string;

  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
  }

  async submit(request: SubmitRequest): Promise<SubmitResponse> {
    const resp = await fetch(`${this.baseUrl}/workers/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    return (await resp.json()) as SubmitResponse;
  }

  async *sendMessage(request: SendMessageRequest): AsyncGenerator<Event> {
    const resp = await fetch(`${this.baseUrl}/sessions/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`sendMessage failed (${resp.status}): ${text}`);
    }

    const reader = resp.body!.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop()!;
      for (const line of lines) {
        if (line.length > 0) {
          yield JSON.parse(line) as Event;
        }
      }
    }

    if (buf.length > 0) {
      yield JSON.parse(buf) as Event;
    }
  }

  async register(request: RegisterRequest): Promise<RegisterResponse> {
    const resp = await fetch(`${this.baseUrl}/workers/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    return (await resp.json()) as RegisterResponse;
  }
}
