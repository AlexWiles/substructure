export interface SessionSummary {
  session_id: string;
  status: string;
  agent_name: string;
  message_count: number;
  token_usage: number;
  stream_version: number;
}

export interface AgentInfo {
  name: string;
  description?: string;
}

export type MessagePart =
  | { type: "text"; text: string }
  | { type: "tool-call"; toolCall: ToolCall };

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  parts: MessagePart[];
}

export interface ToolCall {
  id: string;
  name: string;
  argsText: string;
  args: any;
  result?: unknown;
  subAgent?: SubAgentState;
  status: "pending" | "complete";
}

export interface SubAgentState {
  text: string;
  toolCalls: { id: string; name: string; result?: string }[];
  done: boolean;
}
