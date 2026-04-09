type ClientHeaders = Record<string, string>;

export interface BaseClientOptions {
  baseUrl: string;
  headers?: ClientHeaders;
}

export interface RequestOptions {
  signal?: AbortSignal;
  headers?: ClientHeaders;
}

export class BaseClient {
  protected baseUrl: string;
  protected headers?: ClientHeaders;

  constructor(options: BaseClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.headers = options.headers;
  }

  protected mergeHeaders(headers?: ClientHeaders): Headers {
    const merged = new Headers(this.headers);
    if (headers) {
      const extra = new Headers(headers);
      for (const [k, v] of extra.entries()) {
        merged.set(k, v);
      }
    }
    return merged;
  }

  protected buildUrl(path: string, query?: Record<string, string | undefined>): string {
    const url = `${this.baseUrl}${path}`;
    if (!query) return url;
    const search = new URLSearchParams(
      Object.entries(query).filter(([, v]) => v !== undefined) as [string, string][]
    );
    return `${url}?${search}`;
  }

  protected async fetch(url: string, init?: RequestInit): Promise<Response> {
    const resp = await fetch(url, init);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${init?.method ?? "GET"} ${url} failed (${resp.status}): ${text}`);
    }
    return resp;
  }

  protected async get<T>(path: string, params?: Record<string, string | undefined>, opts?: RequestOptions): Promise<T> {
    const resp = await this.fetch(this.buildUrl(path, params), {
      signal: opts?.signal,
      headers: this.mergeHeaders(opts?.headers),
    });
    return (await resp.json()) as T;
  }

  protected async post<T>(path: string, body: unknown, opts?: RequestOptions): Promise<T> {
    const resp = await this.fetch(this.buildUrl(path), {
      method: "POST",
      headers: this.mergeHeaders({ "Content-Type": "application/json", ...opts?.headers }),
      body: JSON.stringify(body),
      signal: opts?.signal,
    });
    return (await resp.json()) as T;
  }

  protected async *streamSSEGet<T>(path: string, params?: Record<string, string | undefined>, opts?: RequestOptions): AsyncGenerator<T> {
    const resp = await this.fetch(this.buildUrl(path, params), {
      signal: opts?.signal,
      headers: this.mergeHeaders({ Accept: "text/event-stream", ...opts?.headers }),
    });
    yield* this.readSSE<T>(resp);
  }

  protected async *streamSSE<T>(path: string, body: unknown, opts?: RequestOptions): AsyncGenerator<T> {
    const resp = await this.fetch(this.buildUrl(path), {
      method: "POST",
      headers: this.mergeHeaders({ "Content-Type": "application/json", Accept: "text/event-stream", ...opts?.headers }),
      body: JSON.stringify(body),
      signal: opts?.signal,
    });
    yield* this.readSSE<T>(resp);
  }

  private async *readSSE<T>(resp: Response): AsyncGenerator<T> {
    const body = resp.body;
    if (!body) throw new Error("Response body is null");

    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const messages = buf.split("\n\n");
        buf = messages.pop()!;

        for (const msg of messages) {
          if (!msg.trim()) continue;
          let data = "";
          for (const line of msg.split("\n")) {
            if (line.startsWith("data: ")) {
              data += line.slice(6);
            } else if (line.startsWith("data:")) {
              data += line.slice(5);
            }
          }
          if (data) {
            yield JSON.parse(data) as T;
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}
