import { useState, useRef, useCallback, useEffect } from "react";
import { HttpAgent, EventType } from "@ag-ui/client";
import type {
  RunAgentInput,
  TextMessageContentEvent,
  ToolCallStartEvent,
  ToolCallArgsEvent,
  ToolCallResultEvent,
  RunErrorEvent,
} from "@ag-ui/client";
import type { SubstructureClient } from "./client";
import type { ChatMessage, MessagePart, ToolCall, SubAgentState } from "./types";
import { createEventQueue, observeChild } from "./event-queue";

interface InternalToolCall {
  id: string;
  name: string;
  argsText: string;
  args: any;
  result?: unknown;
  isError?: boolean;
}

interface InternalSubAgent {
  text: string;
  toolCalls: Map<string, { name: string; argsText: string; args: any; result?: string }>;
  done: boolean;
}

function serializeSubAgent(sa: InternalSubAgent): SubAgentState {
  return {
    text: sa.text,
    toolCalls: Array.from(sa.toolCalls.entries()).map(([id, tc]) => ({
      id,
      name: tc.name,
      result: tc.result,
    })),
    done: sa.done,
  };
}

function tryParseJSON(str: string | undefined): any {
  if (!str) return {};
  try {
    return JSON.parse(str);
  } catch {
    return {};
  }
}

type OrderedPart =
  | { kind: "text"; text: string }
  | { kind: "tool"; toolCallId: string };

// ---------------------------------------------------------------------------
// ag-ui wire message type (what the backend expects)
// ---------------------------------------------------------------------------

interface AgUiWireMessage {
  id?: string;
  role: string;
  content?: string;
  toolCalls?: { id: string; function: { name: string; arguments: string } }[];
  toolCallId?: string;
}

// ---------------------------------------------------------------------------
// Hydrate ChatMessage[] + raw wire messages from a MESSAGES_SNAPSHOT
// ---------------------------------------------------------------------------

interface HydrateResult {
  chatMessages: ChatMessage[];
  wireMessages: AgUiWireMessage[];
}

function hydrateMessages(snapshot: AgUiWireMessage[]): HydrateResult {
  const chatMessages: ChatMessage[] = [];

  // Collect tool results keyed by toolCallId
  const toolResults = new Map<string, string>();
  for (const msg of snapshot) {
    if (msg.role === "tool" && msg.toolCallId) {
      toolResults.set(msg.toolCallId, msg.content ?? "");
    }
  }

  for (const msg of snapshot) {
    if (msg.role === "user") {
      chatMessages.push({
        id: msg.id ?? crypto.randomUUID(),
        role: "user",
        parts: [{ type: "text", text: msg.content ?? "" }],
      });
    } else if (msg.role === "assistant") {
      const parts: MessagePart[] = [];
      if (msg.content) {
        parts.push({ type: "text", text: msg.content });
      }
      if (msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          const args = tryParseJSON(tc.function?.arguments);
          const resultContent = toolResults.get(tc.id);
          let parsedResult: unknown | undefined;
          if (resultContent !== undefined) {
            try {
              parsedResult = JSON.parse(resultContent);
            } catch {
              parsedResult = resultContent;
            }
          }
          parts.push({
            type: "tool-call",
            toolCall: {
              id: tc.id,
              name: tc.function?.name ?? "",
              argsText: tc.function?.arguments ?? "",
              args,
              result: parsedResult,
              status: parsedResult !== undefined ? "complete" : "pending",
            },
          });
        }
      }
      if (parts.length > 0) {
        chatMessages.push({
          id: msg.id ?? crypto.randomUUID(),
          role: "assistant",
          parts,
        });
      }
    }
  }

  return { chatMessages, wireMessages: snapshot };
}

// ---------------------------------------------------------------------------
// Load existing messages via the observe SSE endpoint
// ---------------------------------------------------------------------------

async function loadExistingMessages(
  baseUrl: string,
  headers: Record<string, string>,
  sessionId: string,
  signal: AbortSignal,
): Promise<HydrateResult> {
  const empty: HydrateResult = { chatMessages: [], wireMessages: [] };
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;

  try {
    const resp = await fetch(
      `${baseUrl}/sessions/${sessionId}/ag-ui/observe`,
      { signal, headers },
    );
    if (!resp.ok || !resp.body) return empty;

    reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop()!;
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === "MESSAGES_SNAPSHOT") {
            reader.cancel().catch(() => {});
            return hydrateMessages(data.messages ?? []);
          }
          if (data.type === "RUN_FINISHED" || data.type === "RUN_ERROR") {
            reader.cancel().catch(() => {});
            return empty;
          }
        } catch {}
      }
    }
  } catch {
    // AbortError, TypeError from cancelled stream, network errors — all fine
  } finally {
    reader?.cancel().catch(() => {});
  }

  return empty;
}

// ---------------------------------------------------------------------------
// useChat hook
// ---------------------------------------------------------------------------

export function useChat(client: SubstructureClient, sessionId: string | null) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // Raw ag-ui wire messages for SyncConversation — kept in a ref to avoid
  // re-renders and stale closure issues.
  const wireRef = useRef<AgUiWireMessage[]>([]);

  // Load existing messages when session changes.
  // Use a ref to track streaming so the effect can skip cleanup when
  // sendMessage() has already kicked off a run for this session.
  const streamingRef = useRef(false);

  useEffect(() => {
    // If we're already streaming into this session (new-chat flow),
    // skip the reset — sendMessage has already set up state.
    if (streamingRef.current) {
      return;
    }

    abortRef.current?.abort();
    setMessages([]);
    setIsStreaming(false);
    wireRef.current = [];

    if (!sessionId) {
      setLoadingHistory(false);
      return;
    }

    const controller = new AbortController();
    setLoadingHistory(true);

    loadExistingMessages(
      client.baseUrl,
      client.headers(),
      sessionId,
      controller.signal,
    )
      .then((result) => {
        if (!controller.signal.aborted) {
          setMessages(result.chatMessages);
          wireRef.current = result.wireMessages;
          setLoadingHistory(false);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setLoadingHistory(false);
        }
      });

    return () => controller.abort();
  }, [client, sessionId]);

  const sendMessage = useCallback(
    (text: string, targetSessionId?: string) => {
      const sid = targetSessionId ?? sessionId;
      if (!sid || isStreaming) return;

      // Signal to the useEffect that we're actively streaming into this
      // session so it doesn't wipe state when sessionId changes.
      streamingRef.current = true;

      const userMsgId = crypto.randomUUID();
      const userMsg: ChatMessage = {
        id: userMsgId,
        role: "user",
        parts: [{ type: "text", text }],
      };

      // Build the full conversation for SyncConversation:
      // existing wire messages + new user message
      const newWireMsg: AgUiWireMessage = {
        id: userMsgId,
        role: "user",
        content: text,
      };
      const fullHistory = [...wireRef.current, newWireMsg];

      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);

      const abortController = new AbortController();
      abortRef.current = abortController;
      const signal = abortController.signal;

      const runId = crypto.randomUUID();
      const agent = new HttpAgent({ url: `${client.baseUrl}/sessions/${sid}/ag-ui` });
      const events$ = agent.run({
        threadId: sid,
        runId,
        messages: fullHistory,
        tools: [],
        context: [],
      } as RunAgentInput);

      const { push, finish, iterable } = createEventQueue(signal);

      const subscription = events$.subscribe({
        next(event) {
          push({ source: "parent", event });
        },
        error(err) {
          console.error("[ag-ui] observable error", err);
          finish();
        },
        complete() {
          finish();
        },
      });
      signal.addEventListener("abort", () => subscription.unsubscribe(), { once: true });

      const childAbort = new AbortController();
      signal.addEventListener("abort", () => childAbort.abort(), { once: true });

      const orderedParts: OrderedPart[] = [];
      const toolCalls = new Map<string, InternalToolCall>();
      const subAgents = new Map<string, InternalSubAgent>();

      function appendText(delta: string) {
        const last = orderedParts[orderedParts.length - 1];
        if (last && last.kind === "text") {
          last.text += delta;
        } else {
          orderedParts.push({ kind: "text", text: delta });
        }
      }

      function buildAssistantMessage(): ChatMessage {
        const parts: MessagePart[] = [];
        for (const op of orderedParts) {
          if (op.kind === "text") {
            parts.push({ type: "text", text: op.text });
          } else {
            const tc = toolCalls.get(op.toolCallId);
            if (tc) {
              const sa = subAgents.get(tc.id);
              parts.push({
                type: "tool-call",
                toolCall: {
                  id: tc.id,
                  name: tc.name,
                  argsText: tc.argsText,
                  args: tc.args,
                  result: tc.result,
                  subAgent: sa ? serializeSubAgent(sa) : undefined,
                  status: tc.result !== undefined ? "complete" : "pending",
                },
              });
            }
          }
        }
        return { id: `assistant-${runId}`, role: "assistant", parts };
      }

      function emitUpdate() {
        const msg = buildAssistantMessage();
        setMessages((prev) => {
          const idx = prev.findIndex((m) => m.id === msg.id);
          if (idx >= 0) {
            const next = [...prev];
            next[idx] = msg;
            return next;
          }
          return [...prev, msg];
        });
      }

      (async () => {
        try {
          // Accumulate wire messages for the assistant turn so we can append
          // them to wireRef when the run finishes.
          let assistantText = "";

          for await (const item of iterable) {
            if (item.source === "parent") {
              const event = item.event;
              switch (event.type) {
                case EventType.TEXT_MESSAGE_CONTENT: {
                  const e = event as TextMessageContentEvent;
                  assistantText += e.delta;
                  appendText(e.delta);
                  emitUpdate();
                  break;
                }
                case EventType.TOOL_CALL_START: {
                  const e = event as ToolCallStartEvent;
                  const childSessionId = (event as any).childSessionId as string | undefined;
                  toolCalls.set(e.toolCallId, {
                    id: e.toolCallId,
                    name: e.toolCallName,
                    argsText: "",
                    args: {},
                  });
                  orderedParts.push({ kind: "tool", toolCallId: e.toolCallId });
                  if (childSessionId) {
                    subAgents.set(e.toolCallId, {
                      text: "",
                      toolCalls: new Map(),
                      done: false,
                    });
                    observeChild(
                      childSessionId,
                      e.toolCallId,
                      push,
                      childAbort.signal,
                      client.baseUrl,
                      client.headers(),
                    );
                  }
                  emitUpdate();
                  break;
                }
                case EventType.TOOL_CALL_ARGS: {
                  const e = event as ToolCallArgsEvent;
                  const tc = toolCalls.get(e.toolCallId);
                  if (tc) {
                    tc.argsText += e.delta;
                    try {
                      tc.args = JSON.parse(tc.argsText);
                    } catch {}
                    emitUpdate();
                  }
                  break;
                }
                case EventType.TOOL_CALL_RESULT: {
                  const e = event as ToolCallResultEvent;
                  const tc = toolCalls.get(e.toolCallId);
                  if (tc) {
                    try {
                      tc.result = JSON.parse(e.content);
                    } catch {
                      tc.result = e.content;
                    }
                    const sa = subAgents.get(e.toolCallId);
                    if (sa) sa.done = true;
                    emitUpdate();
                  }
                  break;
                }
                case EventType.RUN_ERROR: {
                  const e = event as RunErrorEvent;
                  console.error("[ag-ui] run error", e.message);
                  childAbort.abort();
                  break;
                }
              }
            } else {
              const { toolCallId, data } = item;
              const sa = subAgents.get(toolCallId);
              if (!sa) continue;

              switch (data.type) {
                case "MESSAGES_SNAPSHOT": {
                  const msgs = data.messages || [];
                  for (const msg of msgs) {
                    if (msg.role === "assistant") {
                      if (msg.content) sa.text += msg.content;
                      if (msg.toolCalls) {
                        for (const tc of msg.toolCalls) {
                          sa.toolCalls.set(tc.id, {
                            name: tc.function?.name || "",
                            argsText: tc.function?.arguments || "",
                            args: tryParseJSON(tc.function?.arguments),
                          });
                        }
                      }
                    } else if (msg.role === "tool") {
                      const stc = sa.toolCalls.get(msg.toolCallId);
                      if (stc) stc.result = msg.content;
                    }
                  }
                  break;
                }
                case "TEXT_MESSAGE_CONTENT":
                  sa.text += data.delta || "";
                  break;
                case "TOOL_CALL_START":
                  sa.toolCalls.set(data.toolCallId, {
                    name: data.toolCallName || "",
                    argsText: "",
                    args: {},
                  });
                  break;
                case "TOOL_CALL_ARGS": {
                  const stc = sa.toolCalls.get(data.toolCallId);
                  if (stc) {
                    stc.argsText += data.delta || "";
                    try {
                      stc.args = JSON.parse(stc.argsText);
                    } catch {}
                  }
                  break;
                }
                case "TOOL_CALL_RESULT": {
                  const stc = sa.toolCalls.get(data.toolCallId);
                  if (stc) stc.result = data.content;
                  break;
                }
                case "RUN_FINISHED":
                  sa.done = true;
                  break;
              }
              emitUpdate();
            }
          }

          // Run finished — reload wire messages from the server so they
          // match the backend's internal state exactly.  The client-side
          // accumulation can diverge (e.g. multiple assistant turns in an
          // agentic loop get collapsed into one), which causes
          // SyncConversation prefix-match to fail on the next send.
          try {
            const refreshed = await loadExistingMessages(
              client.baseUrl,
              client.headers(),
              sid,
              new AbortController().signal,
            );
            wireRef.current = refreshed.wireMessages;
          } catch {
            // Best-effort — if it fails, the next send will still attempt
            // SyncConversation with whatever wireRef already has.
          }
        } finally {
          childAbort.abort();
          streamingRef.current = false;
          setIsStreaming(false);
        }
      })();
    },
    [client, sessionId, isStreaming],
  );

  const abort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { messages, isStreaming, loadingHistory, sendMessage, abort };
}
