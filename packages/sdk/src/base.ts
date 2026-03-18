export interface BaseClientOptions {
  baseUrl: string;
}

export interface RequestOptions {
  signal?: AbortSignal;
}

export class BaseClient {
  protected baseUrl: string;

  constructor(options: BaseClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
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
    const resp = await this.fetch(this.buildUrl(path, params), { signal: opts?.signal });
    return (await resp.json()) as T;
  }

  protected async post<T>(path: string, body: unknown, opts?: RequestOptions): Promise<T> {
    const resp = await this.fetch(this.buildUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: opts?.signal,
    });
    return (await resp.json()) as T;
  }

  protected async *streamNdjsonGet<T>(path: string, params?: Record<string, string | undefined>, opts?: RequestOptions): AsyncGenerator<T> {
    const resp = await this.fetch(this.buildUrl(path, params), { signal: opts?.signal });
    yield* this.readNdjson<T>(resp);
  }

  protected async *streamNdjson<T>(path: string, body: unknown, opts?: RequestOptions): AsyncGenerator<T> {
    const resp = await this.fetch(this.buildUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: opts?.signal,
    });
    yield* this.readNdjson<T>(resp);
  }

  private async *readNdjson<T>(resp: Response): AsyncGenerator<T> {
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
        const lines = buf.split("\n");
        buf = lines.pop()!;
        for (const line of lines) {
          if (line.length > 0) {
            yield JSON.parse(line) as T;
          }
        }
      }

      if (buf.length > 0) {
        yield JSON.parse(buf) as T;
      }
    } finally {
      reader.releaseLock();
    }
  }
}
