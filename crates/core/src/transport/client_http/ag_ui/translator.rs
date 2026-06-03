use std::collections::HashSet;

use axum::response::sse::Event as SseEvent;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::events::AgUiEvent;
use crate::event_store::Event;
use crate::llm::TokenDelta;
use crate::session::events::{EventPayload, ToolHandler};

/// A client "dialect": the extra AG-UI events that tell *this* client to
/// **execute** a frontend (client-handled) tool call, emitted right after its
/// `TOOL_CALL_START/ARGS/END`. Everything else in the stream is identical across
/// clients — only this trigger differs. Implement a dialect as one function.
pub type ClientToolSignal = fn(tool_call_id: &str, name: &str, input: &Value) -> Vec<AgUiEvent>;

/// Spec / assistant-ui / CopilotKit: they run a frontend tool straight off the
/// standard `TOOL_CALL_*` events, so there's nothing extra to emit.
pub fn spec_client_tool_signal(_tool_call_id: &str, _name: &str, _input: &Value) -> Vec<AgUiEvent> {
    vec![]
}

/// TanStack AI gates frontend-tool execution on a `tool-input-available` CUSTOM
/// event (its own server emits one); without it the call never runs. It's a
/// spec-legal AG-UI CUSTOM event, ignored by clients that don't use it.
pub fn tanstack_client_tool_signal(
    tool_call_id: &str,
    name: &str,
    input: &Value,
) -> Vec<AgUiEvent> {
    vec![AgUiEvent::Custom {
        name: "tool-input-available".to_string(),
        value: json!({ "toolCallId": tool_call_id, "toolName": name, "input": input }),
    }]
}

/// Pick a dialect's signal by name (from the client's hint). Defaults to spec.
pub fn client_tool_signal_for(hint: Option<&str>) -> ClientToolSignal {
    match hint {
        Some("tanstack") => tanstack_client_tool_signal,
        _ => spec_client_tool_signal,
    }
}

/// Translates substructure session events + token deltas into the AG-UI
/// protocol event sequence.
///
/// Stateful: tracks open `TEXT_MESSAGE` / `TOOL_CALL` brackets so every
/// `*_START` is matched by a `*_END` before any terminal event
/// (`RUN_FINISHED` / `RUN_ERROR`) — AG-UI clients are strict state machines.
///
/// Each method returns zero or more [`AgUiEvent`]s; the driver loop serializes
/// them to SSE frames.
pub struct AgUiTranslator {
    thread_id: String,
    run_id: String,
    /// The engine turn this run is following. Learned from `turn.started` (a new
    /// user turn) or adopted from the first token delta (a resumed turn, which
    /// emits no `turn.started`). Used to scope session-wide token deltas — the
    /// engine turn id need not equal the client `runId` on a resumed run.
    turn_id: Option<String>,
    /// call_ids with an open `TEXT_MESSAGE` (START emitted, END not yet).
    open_text: HashSet<String>,
    /// call_ids that produced at least one token delta, so we do not also
    /// synthesize text from the final `llm.call.completed` content.
    streamed_calls: HashSet<String>,
    /// The current llm `call_id` (from `llm.call.requested`). Used as a tool
    /// call's `parentMessageId` so the client binds the tool call to a *stable*
    /// assistant message from the moment it appears — same id the text in that
    /// response uses. Without it a tool-only response has no server message id,
    /// so clients keep an optimistic placeholder and reassign it at
    /// `RUN_FINISHED`, which spawns a phantom message branch and breaks
    /// `addToolResult` (the frontend-tool result never lands → no resume).
    current_call_id: Option<String>,
    /// tool_call_ids mid-bracket — tracked for the close-all finalizer.
    open_tools: HashSet<String>,
    /// The client dialect's frontend-tool execute trigger (see `ClientToolSignal`).
    client_tool_signal: ClientToolSignal,
    pub terminated: bool,
}

impl AgUiTranslator {
    pub fn new(thread_id: String, run_id: String, client_tool_signal: ClientToolSignal) -> Self {
        Self {
            thread_id,
            run_id,
            turn_id: None,
            open_text: HashSet::new(),
            streamed_calls: HashSet::new(),
            current_call_id: None,
            open_tools: HashSet::new(),
            client_tool_signal,
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

    /// A transient token delta. Filtered to this run; opens a `TEXT_MESSAGE`
    /// on the first delta for a call, then streams content.
    pub fn on_delta(&mut self, delta: TokenDelta) -> Vec<AgUiEvent> {
        if self.terminated {
            return vec![];
        }
        // The delta transport is session-wide; scope it to this run's turn. On a
        // resumed turn there is no `turn.started`, so adopt the turn id from the
        // first delta we see, then reject deltas from any other turn.
        match (&self.turn_id, &delta.turn_id) {
            (Some(known), Some(d)) if known != d => return vec![],
            (None, Some(d)) => self.turn_id = Some(d.clone()),
            _ => {}
        }
        let mut out = Vec::new();
        if !self.open_text.contains(&delta.call_id) {
            self.open_text.insert(delta.call_id.clone());
            self.streamed_calls.insert(delta.call_id.clone());
            out.push(AgUiEvent::TextMessageStart {
                message_id: delta.call_id.clone(),
                role: "assistant",
            });
        }
        if let Some(text) = delta.text {
            if !text.is_empty() {
                out.push(AgUiEvent::TextMessageContent {
                    message_id: delta.call_id,
                    delta: text,
                });
            }
        }
        out
    }

    /// A persisted session event.
    pub fn on_event(&mut self, event: EventPayload) -> Vec<AgUiEvent> {
        if self.terminated {
            return vec![];
        }
        match event {
            // A new user turn announces its id; a resumed turn does not (it
            // adopts the id from its first delta in `on_delta`).
            EventPayload::TurnStarted(t) => {
                self.turn_id = Some(t.turn_id);
                vec![]
            }
            // Remember the active llm call so tool calls it produces can name it
            // as their `parentMessageId` (see `current_call_id`).
            EventPayload::LlmCallRequested(r) => {
                self.current_call_id = Some(r.call_id);
                vec![]
            }
            EventPayload::LlmCallCompleted(c) => {
                self.on_llm_completed(c.call_id, c.response.content)
            }
            EventPayload::ToolCallRequested(t) => {
                // The call streams the same way whether the worker or the
                // browser will run it. `parentMessageId` binds it to the
                // assistant message of the llm call that produced it, giving the
                // client a stable server message id (matches the text in that
                // same response).
                let mut out = vec![
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
                ];
                // A client tool yields the run: the engine goes idle until the
                // browser executes it and resumes with a new run carrying the
                // result. Close the run now; the result re-enters on the resumed
                // run as TOOL_CALL_RESULT (via `tool.call.completed`).
                if t.handler == ToolHandler::Client {
                    // The standard TOOL_CALL_* events tell a client a tool was
                    // *called*, not that it's the client's to *execute*. Each
                    // client dialect signals that differently (see
                    // `ClientToolSignal`); emit whatever this one needs.
                    let input: Value =
                        serde_json::from_str(&t.arguments).unwrap_or_else(|_| json!({}));
                    out.extend((self.client_tool_signal)(&t.tool_call_id, &t.name, &input));
                    out.extend(self.close_all_open());
                    out.push(AgUiEvent::RunFinished {
                        thread_id: self.thread_id.clone(),
                        run_id: self.run_id.clone(),
                        result: None,
                    });
                    self.terminated = true;
                }
                out
            }
            EventPayload::ToolCallCompleted(t) => vec![tool_result(t.tool_call_id, t.result)],
            EventPayload::ToolCallErrored(t) => vec![tool_result(t.tool_call_id, t.error)],
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
            // Internal/unmapped events (session.created, worker.*, sub_agent.*,
            // message.new, llm.call.requested, interrupts, …) are not surfaced.
            _ => vec![],
        }
    }

    fn on_llm_completed(&mut self, call_id: String, content: Option<String>) -> Vec<AgUiEvent> {
        if self.open_text.remove(&call_id) {
            // Streamed: close the message we opened from deltas.
            return vec![AgUiEvent::TextMessageEnd {
                message_id: call_id,
            }];
        }
        if self.streamed_calls.contains(&call_id) {
            // Streamed but already closed — nothing to do.
            return vec![];
        }
        // Non-streaming: synthesize the whole message from the final content.
        match content {
            Some(text) if !text.is_empty() => vec![
                AgUiEvent::TextMessageStart {
                    message_id: call_id.clone(),
                    role: "assistant",
                },
                AgUiEvent::TextMessageContent {
                    message_id: call_id.clone(),
                    delta: text,
                },
                AgUiEvent::TextMessageEnd {
                    message_id: call_id,
                },
            ],
            _ => vec![],
        }
    }

    /// Close every open bracket in a deterministic order.
    fn close_all_open(&mut self) -> Vec<AgUiEvent> {
        let mut out = Vec::new();
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

/// Spawn the translation task and return the SSE receiver. Mirrors
/// `merge_session_stream` but maps the merged stream to AG-UI events with
/// bracketing state. The task ends on the first terminal event, when the
/// event channel closes, or on shutdown.
pub fn run_ag_ui_translation(
    mut event_rx: mpsc::Receiver<Event>,
    mut delta_rx: mpsc::Receiver<TokenDelta>,
    thread_id: String,
    run_id: String,
    client_tool_signal: ClientToolSignal,
    shutdown: CancellationToken,
) -> mpsc::Receiver<SseEvent> {
    let (out_tx, out_rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let mut t = AgUiTranslator::new(thread_id, run_id, client_tool_signal);
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
                        // Before closing a streamed text message on completion,
                        // drain any deltas already queued for it so a
                        // TEXT_MESSAGE_END never precedes its last CONTENT.
                        if let EventPayload::LlmCallCompleted(ref c) = payload {
                            if t.open_text.contains(&c.call_id) {
                                while let Ok(d) = delta_rx.try_recv() {
                                    for v in t.on_delta(d) {
                                        if out_tx.send(to_sse(&v)).await.is_err() {
                                            return;
                                        }
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
                    // The delta transport outlives any single turn; its closure
                    // is not a turn-end signal.
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
    use serde_json::json;

    fn ev(v: Value) -> EventPayload {
        serde_json::from_value(v).expect("valid EventPayload")
    }

    fn delta(call_id: &str, run_id: &str, text: &str) -> TokenDelta {
        TokenDelta {
            tenant_id: "t".into(),
            root_session_id: "s".into(),
            session_id: "s".into(),
            agent_id: "a".into(),
            turn_id: Some(run_id.into()),
            call_id: call_id.into(),
            attempt: 0,
            seq: 0,
            text: Some(text.into()),
            finish_reason: None,
        }
    }

    /// Serialize emitted events to their wire JSON so the assertions below can
    /// inspect exact field names and values.
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
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
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
    fn run_finished_carries_result() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
        let d = vals(t.on_event(ev(json!({
            "type": "turn.completed", "turn_id": "r1", "data": {"answer": 42},
        }))));
        assert_eq!(d[0]["result"], json!({"answer": 42}));
    }

    #[test]
    fn ignores_deltas_from_other_runs() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
        // turn.started pins this run's turn; a delta from another turn is dropped.
        let _ = t.on_event(ev(json!({"type": "turn.started", "turn_id": "r1"})));
        assert!(t.on_delta(delta("c1", "OTHER", "x")).is_empty());
    }

    #[test]
    fn resumed_turn_adopts_turn_id_from_first_delta() {
        // A continuation run (runId r2) resuming the original turn r1: there is
        // no turn.started, so the first delta's turn id is adopted, then other
        // turns are rejected.
        let mut t = AgUiTranslator::new("t1".into(), "r2".into(), spec_client_tool_signal);
        let a = vals(t.on_delta(delta("c3", "r1", "ok")));
        assert_eq!(kinds(&a), ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT"]);
        assert!(t.on_delta(delta("c9", "OTHER", "x")).is_empty());
    }

    #[test]
    fn non_streaming_synthesizes_text() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
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
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
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
        // The llm call announces itself, then produces a tool call. The tool
        // call's `parentMessageId` must be that call_id so the client binds it
        // to a stable assistant message (no optimistic-id reassignment, no
        // phantom branch, frontend-tool result lands and resumes the run).
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
        t.on_event(ev(json!({
            "type": "llm.call.requested", "call_id": "c9", "attempt": 0,
            "request": {"model": "m", "messages": []}, "stream": true,
            "retry": serde_json::from_str::<Value>(RETRY).unwrap(),
        })));
        let s = vals(t.on_event(tool_requested("x", "get_user_timezone", "{}", "client")));
        assert_eq!(s[0]["type"], "TOOL_CALL_START");
        assert_eq!(s[0]["parentMessageId"], "c9");
        // The same call's text (if any) uses the same id — one assistant message.
        assert_eq!(s[0]["toolCallId"], "x");
    }

    #[test]
    fn tool_call_without_llm_call_omits_parent_message_id() {
        // Defensive: if no llm.call.requested was seen, we simply omit the field
        // rather than emit a bogus parent.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
        let s = vals(t.on_event(tool_requested("x", "f", "{}", "worker")));
        assert!(s[0].get("parentMessageId").is_none());
    }

    #[test]
    fn parallel_tool_calls_bracket_independently() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
        let a = vals(t.on_event(tool_requested("a", "f", "{}", "worker")));
        let b = vals(t.on_event(tool_requested("b", "g", "{}", "worker")));
        assert_eq!(a[0]["toolCallId"], "a");
        assert_eq!(b[0]["toolCallId"], "b");
        // Each emits its own complete START/ARGS/END triple.
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
    fn client_tool_call_spec_dialect_yields_without_custom() {
        // Spec / assistant-ui / CopilotKit: the call streams and the run yields;
        // no extra execute signal — they run the tool off TOOL_CALL_*.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
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

    #[test]
    fn client_tool_call_tanstack_dialect_emits_custom() {
        // TanStack AI: same stream plus a `tool-input-available` CUSTOM event so
        // its runtime executes the tool.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), tanstack_client_tool_signal);
        let s = vals(t.on_event(tool_requested("x", "get_user_timezone", "{}", "client")));
        assert_eq!(
            kinds(&s),
            [
                "TOOL_CALL_START",
                "TOOL_CALL_ARGS",
                "TOOL_CALL_END",
                "CUSTOM",
                "RUN_FINISHED"
            ]
        );
        let custom = &s[3];
        assert_eq!(custom["name"], "tool-input-available");
        assert_eq!(custom["value"]["toolCallId"], "x");
        assert_eq!(custom["value"]["toolName"], "get_user_timezone");
        assert!(t.terminated);
    }

    #[test]
    fn client_tool_result_emitted_on_resume() {
        // On the resumed run, the browser's result re-enters as TOOL_CALL_RESULT
        // (the engine's tool.call.completed), keyed by the same toolCallId.
        let mut t = AgUiTranslator::new("t1".into(), "r2".into(), spec_client_tool_signal);
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
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
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
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
        let e = t.on_event(ev(json!({
            "type": "llm.call.errored", "call_id": "c1", "attempt": 0,
            "error": "temporary", "retryable": true,
        })));
        assert!(e.is_empty());
        assert!(!t.terminated);
    }

    #[test]
    fn nothing_emitted_after_termination() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into(), spec_client_tool_signal);
        let _ = t.on_event(ev(json!({"type": "turn.completed", "turn_id": "r1"})));
        assert!(t.terminated);
        assert!(t.on_delta(delta("c1", "r1", "late")).is_empty());
        assert!(t
            .on_event(ev(json!({"type": "turn.completed", "turn_id": "r1"})))
            .is_empty());
    }
}
