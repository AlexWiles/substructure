import { useMemo, useCallback } from "react";
import { useParams, useNavigate } from "@tanstack/react-router";
import { createClient } from "../lib/client";
import { useSessions } from "../lib/use-sessions";
import { useChat } from "../lib/use-chat";
import { Sidebar } from "../components/Sidebar";
import { MessageList } from "../components/MessageList";
import { MessageInput } from "../components/MessageInput";
import { NewChat } from "../components/NewChat";

export function Chat() {
  const client = useMemo(() => createClient({ baseUrl: "/client" }), []);
  const { sessionId: urlSessionId } = useParams({ strict: false }) as {
    sessionId?: string;
  };
  const navigate = useNavigate();

  const {
    agents,
    sessions,
    drafting,
    loading,
    error,
    startNewChat: startNewChatState,
    createSession: createSessionState,
    selectSession: selectSessionState,
    refresh,
  } = useSessions(client, urlSessionId);

  const activeSessionId = urlSessionId ?? null;

  const { messages, isStreaming, loadingHistory, sendMessage } = useChat(
    client,
    activeSessionId,
  );

  const selectSession = useCallback(
    (id: string) => {
      selectSessionState(id);
      navigate({ to: "/sessions/$sessionId", params: { sessionId: id } });
    },
    [selectSessionState, navigate],
  );

  const startNewChat = useCallback(() => {
    startNewChatState();
    navigate({ to: "/" });
  }, [startNewChatState, navigate]);

  const handleDraftSend = useCallback(
    async (agent: string, text: string) => {
      try {
        const sessionId = await createSessionState(agent);
        navigate({
          to: "/sessions/$sessionId",
          params: { sessionId },
          replace: true,
        });
        sendMessage(text, sessionId);
      } catch {
        // error surfaced via hook
      }
    },
    [createSessionState, sendMessage, navigate],
  );

  return (
    <div className="chat-layout">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        drafting={drafting}
        loading={loading}
        error={error}
        onNewChat={startNewChat}
        onSelectSession={selectSession}
      />
      <div className="chat-main">
        {activeSessionId && !drafting ? (
          <>
            <MessageList messages={messages} isStreaming={isStreaming} loadingHistory={loadingHistory} />
            <MessageInput onSend={sendMessage} disabled={isStreaming} />
          </>
        ) : (
          <NewChat
            agents={agents}
            onSend={handleDraftSend}
            disabled={loading}
          />
        )}
      </div>
    </div>
  );
}
