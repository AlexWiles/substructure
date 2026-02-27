import type { SessionSummary, AgentInfo } from "./types";

export interface SubstructureClientConfig {
  baseUrl: string;
  token?: string;
}

export interface SubstructureClient {
  listSessions(filter?: { status?: string[]; agent?: string }): Promise<SessionSummary[]>;
  createSession(agent: string): Promise<{ session_id: string }>;
  getSession(id: string): Promise<SessionSummary>;
  listAgents(): Promise<AgentInfo[]>;
  baseUrl: string;
  headers(): Record<string, string>;
}

export function createClient(config: SubstructureClientConfig): SubstructureClient {
  const { baseUrl, token } = config;

  function headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (token) h["Authorization"] = `Bearer ${token}`;
    return h;
  }

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const resp = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers: { ...headers(), ...init?.headers },
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
    }
    return resp.json();
  }

  return {
    baseUrl,
    headers,
    listSessions(filter) {
      const params = new URLSearchParams();
      if (filter?.status) filter.status.forEach((s) => params.append("status", s));
      if (filter?.agent) params.set("agent", filter.agent);
      const qs = params.toString();
      return request<SessionSummary[]>(`/sessions${qs ? `?${qs}` : ""}`);
    },
    createSession(agent) {
      return request<{ session_id: string }>("/sessions", {
        method: "POST",
        body: JSON.stringify({ agent }),
      });
    },
    getSession(id) {
      return request<SessionSummary>(`/sessions/${id}`);
    },
    listAgents() {
      return request<AgentInfo[]>("/agents");
    },
  };
}
