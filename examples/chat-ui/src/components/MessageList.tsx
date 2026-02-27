import { useEffect, useRef } from "react";
import Markdown from "react-markdown";
import type { ChatMessage } from "../lib/types";
import { ToolCard } from "./ToolCard";

interface MessageListProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  loadingHistory: boolean;
}

export function MessageList({ messages, isStreaming, loadingHistory }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  if (loadingHistory) {
    return (
      <div className="message-list empty">
        <p className="empty-text">Loading messages...</p>
      </div>
    );
  }

  if (messages.length === 0) {
    return (
      <div className="message-list empty">
        <p className="empty-text">Send a message to get started.</p>
      </div>
    );
  }

  return (
    <div className="message-list">
      <div className="message-list-inner">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <div className="message-bubble">
              {msg.parts.map((part, i) =>
                part.type === "text" ? (
                  msg.role === "assistant" ? (
                    <div key={i} className="message-text markdown">
                      <Markdown>{part.text}</Markdown>
                    </div>
                  ) : (
                    <div key={i} className="message-text">
                      {part.text}
                    </div>
                  )
                ) : (
                  <ToolCard key={part.toolCall.id} toolCall={part.toolCall} />
                ),
              )}
            </div>
          </div>
        ))}
        {isStreaming && messages[messages.length - 1]?.role !== "assistant" && (
          <div className="message assistant">
            <div className="message-bubble">
              <div className="typing-indicator">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
