import { useState } from "react";
import type { ToolCall } from "../lib/types";

interface ToolCardProps {
  toolCall: ToolCall;
}

function filterArgs(args: any): any {
  if (!args || typeof args !== "object") return args;
  const { __sub, ...rest } = args;
  return Object.keys(rest).length > 0 ? rest : undefined;
}

export function ToolCard({ toolCall }: ToolCardProps) {
  const { name, args, result, subAgent, status } = toolCall;
  const displayArgs = filterArgs(args);
  const label = name.replace(/_/g, " ");
  const isDone = status === "complete";
  const [open, setOpen] = useState(false);
  const isSubAgentActive = subAgent && !isDone;

  return (
    <div className="tool-card">
      <button onClick={() => setOpen(!open)} className="tool-card-header">
        <span className={isDone ? "status-dot done" : "status-dot spinning"} />
        <span className="tool-card-name">{label}</span>
        <span className="tool-card-toggle">{open ? "\u25B2" : "\u25BC"}</span>
      </button>

      {isSubAgentActive && (
        <div className="tool-card-sub-agent">
          {subAgent.text ? (
            <div className="sub-agent-text">{subAgent.text}</div>
          ) : subAgent.toolCalls.length === 0 ? (
            <div className="sub-agent-starting">Starting sub-agent...</div>
          ) : null}
          {subAgent.toolCalls.map((tc) => (
            <div key={tc.id} className="sub-agent-tool">
              <span
                className={
                  tc.result !== undefined
                    ? "status-dot-sm done"
                    : "status-dot-sm spinning"
                }
              />
              <span>{tc.name.replace(/_/g, " ")}</span>
            </div>
          ))}
        </div>
      )}

      {open && (
        <div className="tool-card-detail">
          {displayArgs !== undefined && (
            <>
              <div className="detail-label">args</div>
              <div>{JSON.stringify(displayArgs, null, 2)}</div>
            </>
          )}
          {result !== undefined && (
            <>
              <div className="detail-label mt-2">result</div>
              <div>
                {typeof result === "string"
                  ? result
                  : JSON.stringify(result, null, 2)}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
