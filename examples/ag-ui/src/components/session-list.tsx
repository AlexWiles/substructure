import { newSession, switchSession, useSessions } from "../lib/sessions";

// The conversation switcher. Each entry is an AG-UI thread; selecting one
// re-hydrates the chat from the engine, "+ New" starts a fresh thread.
export function SessionList() {
    const { sessions, activeId } = useSessions();

    return (
        <div className="sessions">
            <div className="sessions-head">
                <span>Conversations</span>
                <button type="button" className="newchat" onClick={() => newSession()}>
                    + New
                </button>
            </div>
            <div className="session-list">
                {sessions.map((s) => (
                    <button
                        key={s.id}
                        type="button"
                        className={`session-item${s.id === activeId ? " active" : ""}`}
                        onClick={() => switchSession(s.id)}
                    >
                        {s.title}
                    </button>
                ))}
            </div>
        </div>
    );
}
