import { useState, useRef, useCallback } from "react";
import type { AgentInfo } from "../lib/types";

interface NewChatProps {
  agents: AgentInfo[];
  onSend: (agent: string, text: string) => void;
  disabled: boolean;
}

export function NewChat({ agents, onSend, disabled }: NewChatProps) {
  const [selectedAgent, setSelectedAgent] = useState(agents[0]?.name ?? "");
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Keep selectedAgent in sync if agents load after mount
  if (!selectedAgent && agents.length > 0) {
    setSelectedAgent(agents[0].name);
  }

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || !selectedAgent || disabled) return;
    onSend(selectedAgent, trimmed);
    setText("");
  }, [text, selectedAgent, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleInput = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, []);

  return (
    <div className="new-chat-view">
      <div className="new-chat-center">
        <h2 className="new-chat-title">New conversation</h2>
        {agents.length > 1 && (
          <div className="new-chat-agent-picker">
            {agents.map((a) => (
              <button
                key={a.name}
                className={`agent-chip ${a.name === selectedAgent ? "active" : ""}`}
                onClick={() => setSelectedAgent(a.name)}
              >
                <span className="agent-chip-name">{a.name}</span>
                {a.description && (
                  <span className="agent-chip-desc">{a.description}</span>
                )}
              </button>
            ))}
          </div>
        )}
        {agents.length === 1 && (
          <p className="new-chat-agent-single">{agents[0].name}</p>
        )}
      </div>
      <div className="message-input-container">
        <div className="message-input-card">
          <textarea
            ref={textareaRef}
            className="message-textarea"
            value={text}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            rows={1}
            disabled={disabled || agents.length === 0}
          />
          <div className="message-input-toolbar">
            <div className="message-input-toolbar-left" />
            <button
              className="send-btn"
              onClick={handleSubmit}
              disabled={disabled || !text.trim() || !selectedAgent}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
