import { useState, useEffect, useCallback, useRef } from "react";
import type { SubstructureClient } from "./client";
import type { SessionSummary, AgentInfo } from "./types";

export interface UseSessionsResult {
  agents: AgentInfo[];
  sessions: SessionSummary[];
  drafting: boolean;
  loading: boolean;
  error: string | null;
  startNewChat(): void;
  createSession(agent: string): Promise<string>;
  selectSession(id: string): void;
  refresh(): Promise<void>;
}

export function useSessions(client: SubstructureClient, initialSessionId?: string | null): UseSessionsResult {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [drafting, setDrafting] = useState(!initialSessionId);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [agentList, sessionList] = await Promise.all([
          client.listAgents().catch(() => [] as AgentInfo[]),
          client.listSessions(),
        ]);
        if (cancelled) return;

        setAgents(agentList);
        setSessions(sessionList.sort((a, b) => b.stream_version - a.stream_version));
      } catch (e: any) {
        if (!cancelled) setError(e.message || "Failed to load sessions");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [client]);

  const refresh = useCallback(async () => {
    try {
      const list = await client.listSessions();
      setSessions(list.sort((a, b) => b.stream_version - a.stream_version));
    } catch (e: any) {
      setError(e.message || "Failed to refresh sessions");
    }
  }, [client]);

  const startNewChat = useCallback(() => {
    setDrafting(true);
  }, []);

  const createSession = useCallback(
    async (agent: string): Promise<string> => {
      setError(null);
      try {
        const { session_id } = await client.createSession(agent);
        setSessions((prev) => [
          {
            session_id,
            status: "idle",
            agent_name: agent,
            message_count: 0,
            token_usage: 0,
            stream_version: Date.now(),
          },
          ...prev,
        ]);
        setDrafting(false);
        return session_id;
      } catch (e: any) {
        const msg = e.message || "Failed to create session";
        setError(msg);
        throw e;
      }
    },
    [client],
  );

  const selectSession = useCallback((id: string) => {
    setDrafting(false);
  }, []);

  return {
    agents,
    sessions,
    drafting,
    loading,
    error,
    startNewChat,
    createSession,
    selectSession,
    refresh,
  };
}
