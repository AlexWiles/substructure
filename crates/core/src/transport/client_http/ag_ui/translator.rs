use std::collections::HashSet;

use axum::response::sse::Event as SseEvent;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::events::AgUiEvent;
use crate::event_store::Event;
use crate::llm::TokenDelta;
use crate::session::events::{EventPayload, ToolHandler};

struct ToolBatch {
    /// Client tool calls not yet seen as `ToolCallRequested`.
    pending_client_tool_calls: HashSet<String>,
    /// Worker tool calls awaiting their result.
    pending_worker_tool_calls: HashSet<String>,
    /// Whether the batch has a client tool (so the turn yields to the browser).
    has_client: bool,
}

impl ToolBatch {
    fn is_yield_point(&self) -> bool {
        self.has_client
            && self.pending_client_tool_calls.is_empty()
            && self.pending_worker_tool_calls.is_empty()
    }
}

/// Translates substructure session events + token deltas into the AG-UI event
/// sequence. Stateful: tracks open `TEXT_MESSAGE` / `TOOL_CALL` / `REASONING`
/// brackets so every `*_START` is matched by a `*_END` before any terminal
/// event — AG-UI clients are strict state machines.
pub struct AgUiTranslator {
    thread_id: String,
    run_id: String,
    /// The engine turn this run follows. Learned from `turn.started`, or adopted
    /// from the first token delta on a resumed turn (which emits no
    /// `turn.started`). The engine turn id need not equal the client `runId`.
    turn_id: Option<String>,
    /// call_ids with an open `TEXT_MESSAGE`.
    open_text: HashSet<String>,
    /// call_ids that produced a text delta, so the final `llm.call.completed`
    /// content is not also synthesized into a message.
    streamed_calls: HashSet<String>,
    /// call_ids with an open reasoning block.
    open_reasoning: HashSet<String>,
    /// tool_call_ids whose START/ARGS already streamed from deltas, so the later
    /// `tool.call.requested` only closes the bracket.
    streamed_tool_calls: HashSet<String>,
    /// Streamed tool_call_ids that emitted a non-empty ARGS fragment. If none
    /// streamed (e.g. a no-arg tool), the close step emits the args once.
    streamed_tool_call_args: HashSet<String>,
    /// tool_call_ids already finalized by `tool.call.requested`; late delta
    /// fragments for them are dropped (re-opening would reset the client's args).
    closed_tool_calls: HashSet<String>,
    /// The current llm `call_id`, used as a tool call's `parentMessageId` so the
    /// client binds it to a stable assistant message from the moment it appears.
    current_call_id: Option<String>,
    /// tool_call_ids mid-bracket, for the close-all finalizer.
    open_tools: HashSet<String>,
    /// Tool calls of the current assistant message; the run yields to the browser
    /// only once the whole batch is on the wire (see [`ToolBatch`]).
    batch: Option<ToolBatch>,
    pub terminated: bool,
}

impl AgUiTranslator {
    pub fn new(thread_id: String, run_id: String) -> Self {
        Self {
            thread_id,
            run_id,
            turn_id: None,
            open_text: HashSet::new(),
            streamed_calls: HashSet::new(),
            open_reasoning: HashSet::new(),
            streamed_tool_calls: HashSet::new(),
            streamed_tool_call_args: HashSet::new(),
            closed_tool_calls: HashSet::new(),
            current_call_id: None,
            open_tools: HashSet::new(),
            batch: None,
            terminated: false,
        }
    }

    /// The opening `RUN_STARTED`. Call once before the event loop.
    pub fn start(&self) -> Vec<AgUiEvent> {
        vec![AgUiEvent::RunStarted {
            thread_id: self.thread_id.clone(),
            run_id: self.run_id.clone(),
        }]
    }

    pub fn on_delta(&mut self, delta: TokenDelta) -> Vec<AgUiEvent> {
        if self.terminated {
            return vec![];
        }
        // Scope the session-wide delta transport to this run's turn; adopt the
        // turn id from the first delta on a resumed turn.
        match (&self.turn_id, &delta.turn_id) {
            (Some(known), Some(d)) if known != d => return vec![],
            (None, Some(d)) => self.turn_id = Some(d.clone()),
            _ => {}
        }
        let mut out = Vec::new();

        // Reasoning gets its own message id (distinct from the answer's call_id)
        // so clients render it as a separate message.
        if let Some(reasoning) = delta.reasoning {
            if !reasoning.is_empty() {
                let rid = reasoning_id(&delta.call_id);
                if self.open_reasoning.insert(delta.call_id.clone()) {
                    out.push(AgUiEvent::ReasoningStart {
                        message_id: rid.clone(),
                    });
                    out.push(AgUiEvent::ReasoningMessageStart {
                        message_id: rid.clone(),
                        role: "reasoning",
                    });
                }
                out.push(AgUiEvent::ReasoningMessageContent {
                    message_id: rid,
                    delta: reasoning,
                });
            }
        }

        // An EMPTY text delta is the provider's priming chunk, sent before
        // reasoning streams — acting on it would open the text message ahead of
        // reasoning and render the thinking below the answer. Only real output
        // (non-empty text or a tool call) ends reasoning.
        let has_text = delta.text.as_deref().is_some_and(|t| !t.is_empty());
        if has_text || !delta.tool_calls.is_empty() {
            out.extend(self.close_reasoning(&delta.call_id));
        }

        if has_text {
            let text = delta.text.unwrap_or_default();
            if self.open_text.insert(delta.call_id.clone()) {
                self.streamed_calls.insert(delta.call_id.clone());
                out.push(AgUiEvent::TextMessageStart {
                    message_id: delta.call_id.clone(),
                    role: "assistant",
                });
            }
            out.push(AgUiEvent::TextMessageContent {
                message_id: delta.call_id.clone(),
                delta: text,
            });
        }

        // Tool-call args stream too: first fragment carries id (+ name) and opens
        // the bracket; the matching `tool.call.requested` closes it (see `on_event`).
        for tc in delta.tool_calls {
            if self.closed_tool_calls.contains(&tc.id) {
                continue;
            }
            if self.open_tools.insert(tc.id.clone()) {
                self.streamed_tool_calls.insert(tc.id.clone());
                out.push(AgUiEvent::ToolCallStart {
                    tool_call_id: tc.id.clone(),
                    tool_call_name: tc.name.unwrap_or_default(),
                    parent_message_id: self.current_call_id.clone(),
                });
            }
            if let Some(args) = tc.arguments {
                if !args.is_empty() {
                    self.streamed_tool_call_args.insert(tc.id.clone());
                    out.push(AgUiEvent::ToolCallArgs {
                        tool_call_id: tc.id,
                        delta: args,
                    });
                }
            }
        }

        out
    }

    pub fn on_event(&mut self, event: EventPayload) -> Vec<AgUiEvent> {
        if self.terminated {
            return vec![];
        }
        match event {
            EventPayload::TurnStarted(t) => {
                self.turn_id = Some(t.turn_id);
                vec![]
            }
            EventPayload::LlmCallRequested(r) => {
                self.current_call_id = Some(r.call_id);
                vec![]
            }
            EventPayload::LlmCallCompleted(c) => {
                // Record the response's tool calls so a client-tool yield waits
                // for the whole batch (see `ToolBatch`).
                let ids: HashSet<String> = c
                    .response
                    .tool_calls
                    .iter()
                    .map(|tc| tc.id.clone())
                    .collect();
                self.batch = (!ids.is_empty()).then(|| ToolBatch {
                    pending_client_tool_calls: ids,
                    pending_worker_tool_calls: HashSet::new(),
                    has_client: false,
                });
                self.on_llm_completed(c.call_id, c.response.content)
            }
            EventPayload::ToolCallRequested(t) => {
                // If args already streamed, just close the bracket; otherwise
                // emit the whole call now.
                let mut out = if self.streamed_tool_calls.remove(&t.tool_call_id) {
                    self.open_tools.remove(&t.tool_call_id);
                    let mut closed = Vec::new();
                    // Nothing streamed for the args (e.g. a no-arg tool) → emit
                    // the complete args once so the client ends with valid JSON.
                    if !self.streamed_tool_call_args.remove(&t.tool_call_id)
                        && !t.arguments.is_empty()
                    {
                        closed.push(AgUiEvent::ToolCallArgs {
                            tool_call_id: t.tool_call_id.clone(),
                            delta: t.arguments.clone(),
                        });
                    }
                    closed.push(AgUiEvent::ToolCallEnd {
                        tool_call_id: t.tool_call_id.clone(),
                    });
                    closed
                } else {
                    vec![
                        AgUiEvent::ToolCallStart {
                            tool_call_id: t.tool_call_id.clone(),
                            tool_call_name: t.name.clone(),
                            parent_message_id: self.current_call_id.clone(),
                        },
                        AgUiEvent::ToolCallArgs {
                            tool_call_id: t.tool_call_id.clone(),
                            delta: t.arguments.clone(),
                        },
                        AgUiEvent::ToolCallEnd {
                            tool_call_id: t.tool_call_id.clone(),
                        },
                    ]
                };
                self.closed_tool_calls.insert(t.tool_call_id.clone());
                let is_client = t.handler == ToolHandler::Client;
                // Yield only once the whole batch is on the wire (every client
                // tool announced, every worker tool resolved). With no batch
                // context (a resumed stream), a lone client call yields at once.
                let yield_now = if let Some(batch) = self.batch.as_mut() {
                    if batch.pending_client_tool_calls.remove(&t.tool_call_id) {
                        if is_client {
                            batch.has_client = true;
                        } else {
                            batch
                                .pending_worker_tool_calls
                                .insert(t.tool_call_id.clone());
                        }
                        batch.is_yield_point()
                    } else {
                        is_client
                    }
                } else {
                    is_client
                };
                if yield_now {
                    out.extend(self.finish_client_yield());
                }
                out
            }
            EventPayload::ToolCallCompleted(t) => {
                let mut out = vec![tool_result(t.tool_call_id.clone(), t.result)];
                // A worker result may be the last thing the batch waits on.
                let yield_now = if let Some(batch) = self.batch.as_mut() {
                    batch.pending_worker_tool_calls.remove(&t.tool_call_id);
                    batch.is_yield_point()
                } else {
                    false
                };
                if yield_now {
                    out.extend(self.finish_client_yield());
                }
                out
            }
            EventPayload::ToolCallErrored(t) => {
                let mut out = vec![tool_result(t.tool_call_id.clone(), t.error)];
                let yield_now = if let Some(batch) = self.batch.as_mut() {
                    batch.pending_worker_tool_calls.remove(&t.tool_call_id);
                    batch.is_yield_point()
                } else {
                    false
                };
                if yield_now {
                    out.extend(self.finish_client_yield());
                }
                out
            }
            EventPayload::LlmCallErrored(e) if !e.retryable => self.finalize_error(e.error),
            EventPayload::SessionCancelled => self.finalize_error("session cancelled".to_string()),
            EventPayload::TurnCompleted(t) => {
                let mut out = self.close_all_open();
                out.push(AgUiEvent::RunFinished {
                    thread_id: self.thread_id.clone(),
                    run_id: self.run_id.clone(),
                    result: if t.data.is_null() { None } else { Some(t.data) },
                });
                self.terminated = true;
                out
            }
            _ => vec![],
        }
    }

    fn on_llm_completed(&mut self, call_id: String, content: Option<String>) -> Vec<AgUiEvent> {
        // A reasoning-only response leaves the block open — close it now.
        let mut out = self.close_reasoning(&call_id);
        if self.open_text.remove(&call_id) {
            out.push(AgUiEvent::TextMessageEnd {
                message_id: call_id,
            });
            return out;
        }
        if self.streamed_calls.contains(&call_id) {
            return out;
        }
        // Non-streaming: synthesize the whole message from the final content.
        if let Some(text) = content {
            if !text.is_empty() {
                out.push(AgUiEvent::TextMessageStart {
                    message_id: call_id.clone(),
                    role: "assistant",
                });
                out.push(AgUiEvent::TextMessageContent {
                    message_id: call_id.clone(),
                    delta: text,
                });
                out.push(AgUiEvent::TextMessageEnd {
                    message_id: call_id,
                });
            }
        }
        out
    }

    fn close_reasoning(&mut self, call_id: &str) -> Vec<AgUiEvent> {
        if self.open_reasoning.remove(call_id) {
            let rid = reasoning_id(call_id);
            vec![
                AgUiEvent::ReasoningMessageEnd {
                    message_id: rid.clone(),
                },
                AgUiEvent::ReasoningEnd { message_id: rid },
            ]
        } else {
            vec![]
        }
    }

    /// Close open brackets and emit `RUN_FINISHED` for a turn that yielded to the
    /// browser on a client tool. The result re-enters on the resumed run.
    fn finish_client_yield(&mut self) -> Vec<AgUiEvent> {
        let mut out = self.close_all_open();
        out.push(AgUiEvent::RunFinished {
            thread_id: self.thread_id.clone(),
            run_id: self.run_id.clone(),
            result: None,
        });
        self.terminated = true;
        self.batch = None;
        out
    }

    /// Close every open bracket in a deterministic order.
    fn close_all_open(&mut self) -> Vec<AgUiEvent> {
        let mut out = Vec::new();
        let mut reasoning: Vec<String> = self.open_reasoning.drain().collect();
        reasoning.sort();
        for call_id in reasoning {
            let rid = reasoning_id(&call_id);
            out.push(AgUiEvent::ReasoningMessageEnd {
                message_id: rid.clone(),
            });
            out.push(AgUiEvent::ReasoningEnd { message_id: rid });
        }
        let mut texts: Vec<String> = self.open_text.drain().collect();
        texts.sort();
        for id in texts {
            out.push(AgUiEvent::TextMessageEnd { message_id: id });
        }
        let mut tools: Vec<String> = self.open_tools.drain().collect();
        tools.sort();
        for id in tools {
            out.push(AgUiEvent::ToolCallEnd { tool_call_id: id });
        }
        out
    }

    /// Close open brackets and emit a terminal `RUN_ERROR`.
    pub fn finalize_error(&mut self, message: String) -> Vec<AgUiEvent> {
        if self.terminated {
            return vec![];
        }
        let mut out = self.close_all_open();
        out.push(AgUiEvent::RunError { message });
        self.terminated = true;
        out
    }
}

fn reasoning_id(call_id: &str) -> String {
    format!("{call_id}-reasoning")
}

fn tool_result(tool_call_id: String, content: String) -> AgUiEvent {
    AgUiEvent::ToolCallResult {
        message_id: tool_call_id.clone(),
        tool_call_id,
        content,
        role: "tool",
    }
}

fn to_sse(event: &AgUiEvent) -> SseEvent {
    let data = serde_json::to_string(event).unwrap_or_default();
    SseEvent::default().event(event.type_name()).data(data)
}

/// Spawn the translation task and return the SSE receiver. The task ends on the
/// first terminal event, when the event channel closes, or on shutdown.
pub fn run_ag_ui_translation(
    mut event_rx: mpsc::Receiver<Event>,
    mut delta_rx: mpsc::Receiver<TokenDelta>,
    thread_id: String,
    run_id: String,
    shutdown: CancellationToken,
) -> mpsc::Receiver<SseEvent> {
    let (out_tx, out_rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let mut t = AgUiTranslator::new(thread_id, run_id);
        for v in t.start() {
            if out_tx.send(to_sse(&v)).await.is_err() {
                return;
            }
        }
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    for v in t.finalize_error("server shutting down".to_string()) {
                        let _ = out_tx.send(to_sse(&v)).await;
                    }
                    return;
                }
                ev = event_rx.recv() => match ev {
                    Some(raw) => {
                        let payload: EventPayload = match serde_json::from_value(raw.payload) {
                            Ok(p) => p,
                            Err(_) => continue,
                        };
                        // Drain queued deltas before handling `llm.call.completed`
                        // (and the tool.call.requested events that follow) so no
                        // closing event outruns its last streamed fragment.
                        if matches!(payload, EventPayload::LlmCallCompleted(_)) {
                            while let Ok(d) = delta_rx.try_recv() {
                                for v in t.on_delta(d) {
                                    if out_tx.send(to_sse(&v)).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                        for v in t.on_event(payload) {
                            if out_tx.send(to_sse(&v)).await.is_err() {
                                return;
                            }
                        }
                        if t.terminated {
                            return;
                        }
                    }
                    None => {
                        if !t.terminated {
                            let msg = "run stream closed before completion".to_string();
                            for v in t.finalize_error(msg) {
                                let _ = out_tx.send(to_sse(&v)).await;
                            }
                        }
                        return;
                    }
                },
                delta = delta_rx.recv() => match delta {
                    Some(d) => {
                        for v in t.on_delta(d) {
                            if out_tx.send(to_sse(&v)).await.is_err() {
                                return;
                            }
                        }
                    }
                    // The delta transport outlives any single turn.
                    None => continue,
                },
            }
        }
    });
    out_rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolCallChunk;
    use serde_json::{json, Value};

    fn ev(v: Value) -> EventPayload {
        serde_json::from_value(v).expect("valid EventPayload")
    }

    fn base_delta(call_id: &str, run_id: &str) -> TokenDelta {
        TokenDelta {
            tenant_id: "t".into(),
            root_session_id: "s".into(),
            session_id: "s".into(),
            agent_id: "a".into(),
            turn_id: Some(run_id.into()),
            call_id: call_id.into(),
            attempt: 0,
            seq: 0,
            text: None,
            reasoning: None,
            tool_calls: vec![],
            finish_reason: None,
        }
    }

    fn delta(call_id: &str, run_id: &str, text: &str) -> TokenDelta {
        TokenDelta {
            text: Some(text.into()),
            ..base_delta(call_id, run_id)
        }
    }

    fn reasoning_delta(call_id: &str, run_id: &str, text: &str) -> TokenDelta {
        TokenDelta {
            reasoning: Some(text.into()),
            ..base_delta(call_id, run_id)
        }
    }

    fn tool_args_delta(
        call_id: &str,
        run_id: &str,
        id: &str,
        name: Option<&str>,
        args: Option<&str>,
    ) -> TokenDelta {
        TokenDelta {
            tool_calls: vec![ToolCallChunk {
                id: id.into(),
                name: name.map(Into::into),
                arguments: args.map(Into::into),
            }],
            ..base_delta(call_id, run_id)
        }
    }

    fn vals(out: Vec<AgUiEvent>) -> Vec<Value> {
        out.iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .collect()
    }

    fn kinds(out: &[Value]) -> Vec<String> {
        out.iter()
            .map(|v| v["type"].as_str().unwrap().to_string())
            .collect()
    }

    const RETRY: &str =
        r#"{"timeout_secs":null,"max_retries":0,"backoff_base_secs":1,"backoff_max_secs":1}"#;

    fn tool_requested(id: &str, name: &str, args: &str, handler: &str) -> EventPayload {
        ev(json!({
            "type": "tool.call.requested",
            "tool_call_id": id,
            "attempt": 0,
            "name": name,
            "arguments": args,
            "handler": handler,
            "retry": serde_json::from_str::<Value>(RETRY).unwrap(),
        }))
    }

    #[test]
    fn streaming_text_turn() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        assert_eq!(kinds(&vals(t.start())), ["RUN_STARTED"]);

        let a = vals(t.on_delta(delta("c1", "r1", "Hel")));
        assert_eq!(kinds(&a), ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT"]);
        assert_eq!(a[0]["messageId"], "c1");
        assert_eq!(a[0]["role"], "assistant");
        assert_eq!(a[1]["delta"], "Hel");

        let b = vals(t.on_delta(delta("c1", "r1", "lo")));
        assert_eq!(kinds(&b), ["TEXT_MESSAGE_CONTENT"]);

        let c = vals(t.on_event(ev(json!({
            "type": "llm.call.completed", "call_id": "c1", "attempt": 0,
            "response": {"model": "m"},
        }))));
        assert_eq!(kinds(&c), ["TEXT_MESSAGE_END"]);

        let d = vals(t.on_event(ev(json!({"type": "turn.completed", "turn_id": "r1"}))));
        assert_eq!(kinds(&d), ["RUN_FINISHED"]);
        assert_eq!(d[0]["threadId"], "t1");
        assert_eq!(d[0]["runId"], "r1");
        assert!(t.terminated);
        assert!(t.open_text.is_empty());
    }

    #[test]
    fn reasoning_streams_then_text_closes_it() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());

        let a = vals(t.on_delta(reasoning_delta("c1", "r1", "Let me think")));
        assert_eq!(
            kinds(&a),
            [
                "REASONING_START",
                "REASONING_MESSAGE_START",
                "REASONING_MESSAGE_CONTENT"
            ]
        );
        // Reasoning carries its own message id, distinct from the answer's call id.
        assert_eq!(a[0]["messageId"], "c1-reasoning");
        // AG-UI's client Zod schema requires this literal.
        assert_eq!(a[1]["role"], "reasoning");
        assert_eq!(a[2]["delta"], "Let me think");

        let b = vals(t.on_delta(reasoning_delta("c1", "r1", " harder")));
        assert_eq!(kinds(&b), ["REASONING_MESSAGE_CONTENT"]);

        let c = vals(t.on_delta(delta("c1", "r1", "Hi")));
        assert_eq!(
            kinds(&c),
            [
                "REASONING_MESSAGE_END",
                "REASONING_END",
                "TEXT_MESSAGE_START",
                "TEXT_MESSAGE_CONTENT"
            ]
        );

        let d = vals(t.on_event(ev(json!({
            "type": "llm.call.completed", "call_id": "c1", "attempt": 0,
            "response": {"model": "m"},
        }))));
        assert_eq!(kinds(&d), ["TEXT_MESSAGE_END"]);
        assert!(t.open_reasoning.is_empty());
    }

    #[test]
    fn leading_empty_text_does_not_preempt_reasoning() {
        // The provider's empty priming chunk must not open the text message ahead
        // of reasoning, which would render the thinking below the answer.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());

        let prime = vals(t.on_delta(delta("c1", "r1", "")));
        assert!(prime.is_empty(), "empty priming text must emit nothing");
        assert!(t.open_text.is_empty(), "text bracket must not open yet");

        let r = vals(t.on_delta(reasoning_delta("c1", "r1", "thinking")));
        assert_eq!(
            kinds(&r),
            [
                "REASONING_START",
                "REASONING_MESSAGE_START",
                "REASONING_MESSAGE_CONTENT"
            ],
            "reasoning must open first, before any text message"
        );

        let a = vals(t.on_delta(delta("c1", "r1", "Hi")));
        assert_eq!(
            kinds(&a),
            [
                "REASONING_MESSAGE_END",
                "REASONING_END",
                "TEXT_MESSAGE_START",
                "TEXT_MESSAGE_CONTENT"
            ],
            "text opens only after reasoning, so it renders below the thinking"
        );
    }

    #[test]
    fn leading_empty_text_does_not_close_reasoning() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(reasoning_delta("c1", "r1", "still thinking"));
        let empty = vals(t.on_delta(delta("c1", "r1", "")));
        assert!(empty.is_empty(), "empty text mid-reasoning emits nothing");
        assert!(
            t.open_reasoning.contains("c1"),
            "reasoning must stay open through an empty text delta"
        );
    }

    #[test]
    fn reasoning_only_response_closes_on_completion() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(reasoning_delta("c1", "r1", "hmm"));
        let done = vals(t.on_event(ev(json!({
            "type": "llm.call.completed", "call_id": "c1", "attempt": 0,
            "response": {"model": "m"},
        }))));
        assert_eq!(kinds(&done), ["REASONING_MESSAGE_END", "REASONING_END"]);
        assert!(t.open_reasoning.is_empty());
    }

    #[test]
    fn streamed_tool_args_then_requested_only_closes() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        t.on_event(ev(json!({
            "type": "llm.call.requested", "call_id": "c1", "attempt": 0,
            "request": {"model": "m", "messages": []}, "stream": true,
            "retry": serde_json::from_str::<Value>(RETRY).unwrap(),
        })));

        let a = vals(t.on_delta(tool_args_delta(
            "c1",
            "r1",
            "call-1",
            Some("set_color"),
            Some("{\"red\":"),
        )));
        assert_eq!(kinds(&a), ["TOOL_CALL_START", "TOOL_CALL_ARGS"]);
        assert_eq!(a[0]["toolCallId"], "call-1");
        assert_eq!(a[0]["toolCallName"], "set_color");
        assert_eq!(a[0]["parentMessageId"], "c1");
        assert_eq!(a[1]["delta"], "{\"red\":");

        let b = vals(t.on_delta(tool_args_delta("c1", "r1", "call-1", None, Some("255}"))));
        assert_eq!(kinds(&b), ["TOOL_CALL_ARGS"]);
        assert_eq!(b[0]["delta"], "255}");

        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]));
        let r = vals(t.on_event(tool_requested(
            "call-1",
            "set_color",
            r#"{"red":255}"#,
            "worker",
        )));
        assert_eq!(kinds(&r), ["TOOL_CALL_END"]);
        assert!(t.open_tools.is_empty());
        assert!(t.streamed_tool_calls.is_empty());
    }

    #[test]
    fn streamed_tool_with_no_args_emits_complete_on_close() {
        // A no-arg tool that streamed only id+name: the close emits the complete
        // args once so the client gets valid JSON, not an empty argsText.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let a = vals(t.on_delta(tool_args_delta(
            "c1",
            "r1",
            "call-1",
            Some("get_color"),
            None,
        )));
        assert_eq!(kinds(&a), ["TOOL_CALL_START"]);
        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]));
        let r = vals(t.on_event(tool_requested("call-1", "get_color", "{}", "worker")));
        assert_eq!(kinds(&r), ["TOOL_CALL_ARGS", "TOOL_CALL_END"]);
        assert_eq!(r[0]["delta"], "{}");
    }

    #[test]
    fn late_tool_fragment_after_requested_is_ignored() {
        // A fragment after the call was finalized must be dropped — re-opening it
        // would reset the client's accumulated args.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(tool_args_delta("c1", "r1", "call-1", Some("f"), Some("{}")));
        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]));
        let _ = t.on_event(tool_requested("call-1", "f", "{}", "worker"));
        let late = vals(t.on_delta(tool_args_delta("c1", "r1", "call-1", None, Some("X"))));
        assert!(late.is_empty());
    }

    #[test]
    fn streamed_client_tool_yields_after_close() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(tool_args_delta(
            "c1",
            "r1",
            "call-1",
            Some("get_color"),
            Some("{}"),
        ));
        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]));
        let r = vals(t.on_event(tool_requested("call-1", "get_color", "{}", "client")));
        assert_eq!(kinds(&r), ["TOOL_CALL_END", "RUN_FINISHED"]);
        assert!(t.terminated);
    }

    #[test]
    fn run_finished_carries_result() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let d = vals(t.on_event(ev(json!({
            "type": "turn.completed", "turn_id": "r1", "data": {"answer": 42},
        }))));
        assert_eq!(d[0]["result"], json!({"answer": 42}));
    }

    #[test]
    fn ignores_deltas_from_other_runs() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(ev(json!({"type": "turn.started", "turn_id": "r1"})));
        assert!(t.on_delta(delta("c1", "OTHER", "x")).is_empty());
    }

    #[test]
    fn resumed_turn_adopts_turn_id_from_first_delta() {
        // A continuation run (runId r2) resuming turn r1 has no turn.started, so
        // the first delta's turn id is adopted, then other turns are rejected.
        let mut t = AgUiTranslator::new("t1".into(), "r2".into());
        let a = vals(t.on_delta(delta("c3", "r1", "ok")));
        assert_eq!(kinds(&a), ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT"]);
        assert!(t.on_delta(delta("c9", "OTHER", "x")).is_empty());
    }

    #[test]
    fn non_streaming_synthesizes_text() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let c = vals(t.on_event(ev(json!({
            "type": "llm.call.completed", "call_id": "c2", "attempt": 0,
            "response": {"model": "m", "content": "hi there"},
        }))));
        assert_eq!(
            kinds(&c),
            [
                "TEXT_MESSAGE_START",
                "TEXT_MESSAGE_CONTENT",
                "TEXT_MESSAGE_END"
            ]
        );
        assert_eq!(c[0]["messageId"], "c2");
        assert_eq!(c[1]["delta"], "hi there");
    }

    #[test]
    fn server_tool_call_and_result() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let s = tool_requested("x", "get_weather", r#"{"city":"SF"}"#, "worker");
        let s = vals(t.on_event(s));
        assert_eq!(
            kinds(&s),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"]
        );
        assert_eq!(s[0]["toolCallId"], "x");
        assert_eq!(s[0]["toolCallName"], "get_weather");
        assert_eq!(s[1]["delta"], r#"{"city":"SF"}"#);

        let r = vals(t.on_event(ev(json!({
            "type": "tool.call.completed", "tool_call_id": "x",
            "name": "get_weather", "result": r#"{"temp":62}"#,
        }))));
        assert_eq!(kinds(&r), ["TOOL_CALL_RESULT"]);
        assert_eq!(r[0]["toolCallId"], "x");
        assert_eq!(r[0]["content"], r#"{"temp":62}"#);
        assert_eq!(r[0]["role"], "tool");
    }

    #[test]
    fn tool_call_carries_parent_message_id_from_llm_call() {
        // The tool call's `parentMessageId` must be the producing llm call_id so
        // the client binds it to a stable assistant message (no phantom branch).
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        t.on_event(ev(json!({
            "type": "llm.call.requested", "call_id": "c9", "attempt": 0,
            "request": {"model": "m", "messages": []}, "stream": true,
            "retry": serde_json::from_str::<Value>(RETRY).unwrap(),
        })));
        let s = vals(t.on_event(tool_requested("x", "get_user_timezone", "{}", "client")));
        assert_eq!(s[0]["type"], "TOOL_CALL_START");
        assert_eq!(s[0]["parentMessageId"], "c9");
        assert_eq!(s[0]["toolCallId"], "x");
    }

    #[test]
    fn tool_call_without_llm_call_omits_parent_message_id() {
        // With no llm.call.requested seen, omit the field rather than emit a bogus parent.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let s = vals(t.on_event(tool_requested("x", "f", "{}", "worker")));
        assert!(s[0].get("parentMessageId").is_none());
    }

    #[test]
    fn parallel_tool_calls_bracket_independently() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let a = vals(t.on_event(tool_requested("a", "f", "{}", "worker")));
        let b = vals(t.on_event(tool_requested("b", "g", "{}", "worker")));
        assert_eq!(a[0]["toolCallId"], "a");
        assert_eq!(b[0]["toolCallId"], "b");
        assert_eq!(
            kinds(&a),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"]
        );
        assert_eq!(
            kinds(&b),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"]
        );
    }

    #[test]
    fn lone_client_tool_call_yields_run() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let s = vals(t.on_event(tool_requested("x", "get_user_timezone", "{}", "client")));
        assert_eq!(
            kinds(&s),
            [
                "TOOL_CALL_START",
                "TOOL_CALL_ARGS",
                "TOOL_CALL_END",
                "RUN_FINISHED"
            ]
        );
        assert!(t.terminated);
    }

    fn llm_completed_with_tools(call_id: &str, tool_ids: &[&str]) -> EventPayload {
        let tool_calls: Vec<Value> = tool_ids
            .iter()
            .map(|id| {
                json!({
                    "id": id, "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                })
            })
            .collect();
        ev(json!({
            "type": "llm.call.completed", "call_id": call_id, "attempt": 0,
            "response": {"model": "m", "tool_calls": tool_calls},
        }))
    }

    #[test]
    fn parallel_client_tools_yield_once_after_whole_batch() {
        // Both client tools must be announced before the run yields, or the
        // browser never sees the second and the turn deadlocks.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(llm_completed_with_tools("c1", &["a", "b"]));

        let first = vals(t.on_event(tool_requested("a", "set_color", "{}", "client")));
        assert_eq!(
            kinds(&first),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"],
            "first parallel client tool must not end the run"
        );
        assert!(!t.terminated);

        let second = vals(t.on_event(tool_requested("b", "get_color", "{}", "client")));
        assert_eq!(
            kinds(&second),
            [
                "TOOL_CALL_START",
                "TOOL_CALL_ARGS",
                "TOOL_CALL_END",
                "RUN_FINISHED"
            ],
            "the last call in the batch yields the run once"
        );
        assert!(t.terminated);
    }

    #[test]
    fn mixed_batch_waits_for_worker_result_before_yielding() {
        // The worker tool's RESULT must be delivered before the run yields on the
        // client tool.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(llm_completed_with_tools("c1", &["w", "c"]));

        assert!(!t
            .on_event(tool_requested("w", "get_weather", "{}", "worker"))
            .iter()
            .any(|e| matches!(e, AgUiEvent::RunFinished { .. })));
        let after_client = vals(t.on_event(tool_requested("c", "set_color", "{}", "client")));
        assert_eq!(
            kinds(&after_client),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"],
            "must not yield while a worker tool in the batch is unresolved"
        );
        assert!(!t.terminated);

        let done = vals(t.on_event(ev(json!({
            "type": "tool.call.completed", "tool_call_id": "w",
            "name": "get_weather", "result": r#"{"temp":62}"#,
        }))));
        assert_eq!(kinds(&done), ["TOOL_CALL_RESULT", "RUN_FINISHED"]);
        assert!(t.terminated);
    }

    #[test]
    fn pure_worker_batch_does_not_yield() {
        // No client tool → the run ends only at turn.completed.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(llm_completed_with_tools("c1", &["w1", "w2"]));
        let _ = t.on_event(tool_requested("w1", "f", "{}", "worker"));
        let _ = t.on_event(tool_requested("w2", "g", "{}", "worker"));
        let r1 = vals(t.on_event(ev(json!({
            "type": "tool.call.completed", "tool_call_id": "w1", "name": "f", "result": "1",
        }))));
        let r2 = vals(t.on_event(ev(json!({
            "type": "tool.call.completed", "tool_call_id": "w2", "name": "g", "result": "2",
        }))));
        assert_eq!(kinds(&r1), ["TOOL_CALL_RESULT"]);
        assert_eq!(kinds(&r2), ["TOOL_CALL_RESULT"]);
        assert!(!t.terminated);
        let end = vals(t.on_event(ev(json!({"type": "turn.completed", "turn_id": "r1"}))));
        assert_eq!(kinds(&end), ["RUN_FINISHED"]);
    }

    #[test]
    fn client_tool_result_emitted_on_resume() {
        let mut t = AgUiTranslator::new("t1".into(), "r2".into());
        let r = vals(t.on_event(ev(json!({
            "type": "tool.call.completed", "tool_call_id": "x",
            "name": "get_user_timezone", "result": "America/Los_Angeles",
        }))));
        assert_eq!(kinds(&r), ["TOOL_CALL_RESULT"]);
        assert_eq!(r[0]["toolCallId"], "x");
        assert_eq!(r[0]["content"], "America/Los_Angeles");
    }

    #[test]
    fn terminal_error_closes_open_text_then_run_error() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(delta("c1", "r1", "partial"));
        let e = vals(t.on_event(ev(json!({
            "type": "llm.call.errored", "call_id": "c1", "attempt": 0,
            "error": "boom", "retryable": false,
        }))));
        assert_eq!(kinds(&e), ["TEXT_MESSAGE_END", "RUN_ERROR"]);
        assert_eq!(e[1]["message"], "boom");
        assert!(t.terminated);
        assert!(t.open_text.is_empty());
    }

    #[test]
    fn retryable_error_is_silent() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let e = t.on_event(ev(json!({
            "type": "llm.call.errored", "call_id": "c1", "attempt": 0,
            "error": "temporary", "retryable": true,
        })));
        assert!(e.is_empty());
        assert!(!t.terminated);
    }

    #[test]
    fn nothing_emitted_after_termination() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(ev(json!({"type": "turn.completed", "turn_id": "r1"})));
        assert!(t.terminated);
        assert!(t.on_delta(delta("c1", "r1", "late")).is_empty());
        assert!(t
            .on_event(ev(json!({"type": "turn.completed", "turn_id": "r1"})))
            .is_empty());
    }
}
