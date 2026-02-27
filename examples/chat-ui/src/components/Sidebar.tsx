import type { SessionSummary } from "../lib/types";

interface SidebarProps {
  sessions: SessionSummary[];
  activeSessionId: string | null;
  drafting: boolean;
  loading: boolean;
  error: string | null;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
}

export function Sidebar({
  sessions,
  activeSessionId,
  drafting,
  loading,
  error,
  onNewChat,
  onSelectSession,
}: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <button
          className="new-chat-btn"
          onClick={onNewChat}
          disabled={loading}
        >
          + New chat
        </button>
        {error && <div className="sidebar-error">{error}</div>}
      </div>
      <div className="session-list">
        {loading && sessions.length === 0 && (
          <div className="session-list-empty">Loading...</div>
        )}
        {sessions.map((s) => (
          <button
            key={s.session_id}
            className={`session-item ${!drafting && s.session_id === activeSessionId ? "active" : ""}`}
            onClick={() => onSelectSession(s.session_id)}
          >
            <span className="session-agent">{s.agent_name}</span>
            <span className="session-meta">
              {s.message_count} msg{s.message_count !== 1 ? "s" : ""}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
