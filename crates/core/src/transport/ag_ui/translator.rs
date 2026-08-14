use std::collections::{HashMap, HashSet};

use axum::response::sse::Event as SseEvent;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::events::{AgUiEvent, AgUiInterrupt, RunOutcome};
use crate::protocol::TokenDelta;
use crate::session::events::EventPayload;
use crate::session::SessionEvent;

/// A tool call's args as an AG-UI delta. Empty args become `"{}"` so clients
/// treat the call as complete (a no-arg call otherwise carries no delta).
fn non_empty_args(arguments: &str) -> String {
    if arguments.is_empty() {
        "{}".to_string()
    } else {
        arguments.to_string()
    }
}

pub struct AgUiTranslator {
    thread_id: String,
    run_id: String,
    turn_id: Option<String>,
    open_text: HashSet<String>,
    streamed_calls: HashSet<String>,
    open_reasoning: HashSet<String>,
    streamed_tool_calls: HashSet<String>,
    streamed_tool_call_args: HashSet<String>,
    closed_tool_calls: HashSet<String>,
    current_call_id: Option<String>,
    open_tools: HashSet<String>,
    sub_agent_calls: HashMap<String, String>,
    terminated: bool,
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
            sub_agent_calls: HashMap::new(),
            terminated: false,
        }
    }

    pub fn terminated(&self) -> bool {
        self.terminated
    }

    pub fn start(&self) -> Vec<AgUiEvent> {
        vec![AgUiEvent::RunStarted {
            thread_id: self.thread_id.clone(),
            run_id: self.run_id.clone(),
        }]
    }

    pub fn on_delta(&mut self, delta: TokenDelta) -> Vec<AgUiEvent> {
        if self.terminated() {
            return vec![];
        }
        match (&self.turn_id, &delta.turn_id) {
            (Some(known), Some(d)) if known != d => return vec![],
            (None, Some(d)) => self.turn_id = Some(d.clone()),
            _ => {}
        }
        let mut out = Vec::new();

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

    /// `ends_run` is [`SessionEvent::ends_run`](crate::session::SessionEvent::ends_run)
    /// for the event this payload came from.
    pub fn on_event(&mut self, event: EventPayload, ends_run: bool) -> Vec<AgUiEvent> {
        if self.terminated() {
            return vec![];
        }
        match event {
            EventPayload::TurnStarted(t) => {
                self.turn_id = Some(t.turn_id);
                vec![]
            }
            EventPayload::LlmCallRequested(r) => {
                self.current_call_id = Some(r.id);
                vec![]
            }
            EventPayload::LlmCallCompleted(c) => self.on_llm_completed(c.id, c.response.content),
            EventPayload::ToolCallRequested(t) => {
                let mut out = if self.streamed_tool_calls.remove(&t.id) {
                    self.open_tools.remove(&t.id);
                    let mut closed = Vec::new();
                    // Emit args even when empty ("{}"): AG-UI clients close the
                    // args stream — and fire a client tool's execute — only on a
                    // non-empty delta.
                    if !self.streamed_tool_call_args.remove(&t.id) {
                        closed.push(AgUiEvent::ToolCallArgs {
                            tool_call_id: t.id.clone(),
                            delta: non_empty_args(&t.arguments),
                        });
                    }
                    closed.push(AgUiEvent::ToolCallEnd {
                        tool_call_id: t.id.clone(),
                    });
                    closed
                } else {
                    vec![
                        AgUiEvent::ToolCallStart {
                            tool_call_id: t.id.clone(),
                            tool_call_name: t.name.clone(),
                            parent_message_id: self.current_call_id.clone(),
                        },
                        AgUiEvent::ToolCallArgs {
                            tool_call_id: t.id.clone(),
                            delta: non_empty_args(&t.arguments),
                        },
                        AgUiEvent::ToolCallEnd {
                            tool_call_id: t.id.clone(),
                        },
                    ]
                };
                self.closed_tool_calls.insert(t.id.clone());
                if ends_run {
                    out.extend(self.finish_client_yield());
                }
                out
            }
            EventPayload::ToolCallCompleted(t) => {
                let mut out = vec![tool_result(t.id.clone(), t.result.rendered())];
                if ends_run {
                    out.extend(self.finish_client_yield());
                }
                out
            }
            EventPayload::ToolCallErrored(t) => {
                let mut out = vec![tool_result(t.id.clone(), t.error.message)];
                if ends_run {
                    out.extend(self.finish_client_yield());
                }
                out
            }
            EventPayload::SubAgentRequested(s) => {
                // A delegation surfaces to the client as a tool call named for the
                // sub-agent; its answer arrives later on `sub_agent.turn_completed`,
                // keyed by child session id, so remember the mapping.
                self.sub_agent_calls.insert(s.id, s.tool_call_id.clone());
                let out = if self.streamed_tool_calls.remove(&s.tool_call_id) {
                    self.open_tools.remove(&s.tool_call_id);
                    self.streamed_tool_call_args.remove(&s.tool_call_id);
                    vec![AgUiEvent::ToolCallEnd {
                        tool_call_id: s.tool_call_id.clone(),
                    }]
                } else {
                    vec![
                        AgUiEvent::ToolCallStart {
                            tool_call_id: s.tool_call_id.clone(),
                            tool_call_name: s.agent_id,
                            parent_message_id: self.current_call_id.clone(),
                        },
                        AgUiEvent::ToolCallEnd {
                            tool_call_id: s.tool_call_id.clone(),
                        },
                    ]
                };
                self.closed_tool_calls.insert(s.tool_call_id.clone());
                out
            }
            EventPayload::SubAgentTurnCompleted(s) => {
                self.settle_sub_agent(&s.id, sub_agent_result(&s.data), ends_run)
            }
            EventPayload::SubAgentErrored(s) => {
                self.settle_sub_agent(&s.id, s.error.message, ends_run)
            }
            EventPayload::LlmCallErrored(e) if !e.retryable => self.run_error(e.error.message),
            EventPayload::SessionCancelled => self.run_error("session cancelled".to_string()),
            EventPayload::SessionInterrupted(p) => {
                let mut out = self.close_all_open();
                out.push(AgUiEvent::RunFinished {
                    thread_id: self.thread_id.clone(),
                    run_id: self.run_id.clone(),
                    result: None,
                    outcome: Some(RunOutcome::Interrupt {
                        interrupts: vec![AgUiInterrupt::from_session(&p)],
                    }),
                });
                self.terminated = true;
                out
            }
            EventPayload::TurnCompleted(t) => {
                // A terminally-failed finalizer completes the turn as a failed run.
                if let Some(err) = t.error {
                    return self.run_error(err.message);
                }
                let mut out = self.close_all_open();
                out.push(AgUiEvent::RunFinished {
                    thread_id: self.thread_id.clone(),
                    run_id: self.run_id.clone(),
                    result: if t.data.is_null() { None } else { Some(t.data) },
                    outcome: None,
                });
                self.terminated = true;
                out
            }
            _ => vec![],
        }
    }

    fn on_llm_completed(&mut self, call_id: String, content: Option<String>) -> Vec<AgUiEvent> {
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

    /// Emit a sub-agent's answer as the result of its delegating tool call,
    /// then finish the run if it was the last unresolved call in a client batch.
    fn settle_sub_agent(
        &mut self,
        session_id: &str,
        content: String,
        ends_run: bool,
    ) -> Vec<AgUiEvent> {
        let Some(tool_call_id) = self.sub_agent_calls.remove(session_id) else {
            return vec![];
        };
        let mut out = vec![tool_result(tool_call_id, content)];
        if ends_run {
            out.extend(self.finish_client_yield());
        }
        out
    }

    fn finish_client_yield(&mut self) -> Vec<AgUiEvent> {
        self.terminated = true;
        let mut out = self.close_all_open();
        out.push(AgUiEvent::RunFinished {
            thread_id: self.thread_id.clone(),
            run_id: self.run_id.clone(),
            result: None,
            outcome: None,
        });
        out
    }

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

    pub fn finalize_error(&mut self, message: String) -> Vec<AgUiEvent> {
        if self.terminated() {
            return vec![];
        }
        self.run_error(message)
    }

    fn run_error(&mut self, message: String) -> Vec<AgUiEvent> {
        self.terminated = true;
        let mut out = self.close_all_open();
        out.push(AgUiEvent::RunError { message });
        out
    }
}

fn reasoning_id(call_id: &str) -> String {
    format!("{call_id}-reasoning")
}

/// A sub-agent turn result: a bare string renders verbatim, anything else as JSON.
fn sub_agent_result(data: &serde_json::Value) -> String {
    match data {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
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

pub fn run_ag_ui_translation(
    mut event_rx: mpsc::Receiver<SessionEvent>,
    mut delta_rx: mpsc::Receiver<TokenDelta>,
    thread_id: String,
    run_id: String,
    shutdown: CancellationToken,
) -> mpsc::Receiver<SseEvent> {
    let (out_tx, out_rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let mut t = AgUiTranslator::new(thread_id, run_id);
        for v in t.start() {
            if out_tx.send(v.to_sse()).await.is_err() {
                return;
            }
        }
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    for v in t.finalize_error("server shutting down".to_string()) {
                        let _ = out_tx.send(v.to_sse()).await;
                    }
                    return;
                }
                _ = out_tx.closed() => return,
                ev = event_rx.recv() => match ev {
                    Some(event) => {
                        let ends_run = event.ends_run();
                        let payload = event.payload;
                        // Drain queued deltas before handling `llm.call.completed`
                        // (and the tool.call.requested events that follow) so no
                        // closing event outruns its last streamed fragment.
                        if matches!(payload, EventPayload::LlmCallCompleted(_)) {
                            while let Ok(d) = delta_rx.try_recv() {
                                for v in t.on_delta(d) {
                                    if out_tx.send(v.to_sse()).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                        for v in t.on_event(payload, ends_run) {
                            if out_tx.send(v.to_sse()).await.is_err() {
                                return;
                            }
                        }
                        if t.terminated() {
                            return;
                        }
                    }
                    None => {
                        if !t.terminated() {
                            let msg = "run stream closed before completion".to_string();
                            for v in t.finalize_error(msg) {
                                let _ = out_tx.send(v.to_sse()).await;
                            }
                        }
                        return;
                    }
                },
                delta = delta_rx.recv() => match delta {
                    Some(d) => {
                        for v in t.on_delta(d) {
                            if out_tx.send(v.to_sse()).await.is_err() {
                                return;
                            }
                        }
                    }
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
    use crate::protocol::ToolCallChunk;
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

    const RETRY: &str = r#"{"attempt_timeout_secs":null,"total_timeout_secs":null,"max_attempts":1,"backoff_base_secs":1,"backoff_max_secs":1}"#;

    fn tool_requested(id: &str, name: &str, args: &str, handler: &str) -> EventPayload {
        ev(json!({
            "type": "tool.call.requested",
            "id": id,
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

        let c = vals(t.on_event(
            ev(json!({
                "type": "llm.call.completed", "id": "c1", "attempt": 0,
                "response": {"model": "m"},
            })),
            false,
        ));
        assert_eq!(kinds(&c), ["TEXT_MESSAGE_END"]);

        let d = vals(t.on_event(
            ev(json!({"type": "turn.completed", "turn_id": "r1"})),
            false,
        ));
        assert_eq!(kinds(&d), ["RUN_FINISHED"]);
        assert_eq!(d[0]["threadId"], "t1");
        assert_eq!(d[0]["runId"], "r1");
        assert!(t.terminated());
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
        assert_eq!(a[0]["messageId"], "c1-reasoning");
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

        let d = vals(t.on_event(
            ev(json!({
                "type": "llm.call.completed", "id": "c1", "attempt": 0,
                "response": {"model": "m"},
            })),
            false,
        ));
        assert_eq!(kinds(&d), ["TEXT_MESSAGE_END"]);
        assert!(t.open_reasoning.is_empty());
    }

    #[test]
    fn leading_empty_text_does_not_preempt_reasoning() {
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
        let done = vals(t.on_event(
            ev(json!({
                "type": "llm.call.completed", "id": "c1", "attempt": 0,
                "response": {"model": "m"},
            })),
            true,
        ));
        assert_eq!(kinds(&done), ["REASONING_MESSAGE_END", "REASONING_END"]);
        assert!(t.open_reasoning.is_empty());
    }

    #[test]
    fn streamed_tool_args_then_requested_only_closes() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        t.on_event(
            ev(json!({
                "type": "llm.call.requested", "id": "c1", "attempt": 0, "llm": "claude",
                "request": {"model": "m", "messages": []}, "stream": true,
                "retry": serde_json::from_str::<Value>(RETRY).unwrap(),
            })),
            false,
        );

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

        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]), false);
        let r = vals(t.on_event(
            tool_requested("call-1", "set_color", r#"{"red":255}"#, "worker"),
            false,
        ));
        assert_eq!(kinds(&r), ["TOOL_CALL_END"]);
        assert!(t.open_tools.is_empty());
        assert!(t.streamed_tool_calls.is_empty());
    }

    #[test]
    fn streamed_tool_with_no_args_emits_complete_on_close() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let a = vals(t.on_delta(tool_args_delta(
            "c1",
            "r1",
            "call-1",
            Some("get_color"),
            None,
        )));
        assert_eq!(kinds(&a), ["TOOL_CALL_START"]);
        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]), false);
        let r = vals(t.on_event(tool_requested("call-1", "get_color", "{}", "worker"), false));
        assert_eq!(kinds(&r), ["TOOL_CALL_ARGS", "TOOL_CALL_END"]);
        assert_eq!(r[0]["delta"], "{}");
    }

    #[test]
    fn streamed_tool_with_empty_args_emits_placeholder() {
        // A no-arg tool call carries empty arguments; emit "{}" so AG-UI clients
        // see a complete args stream and fire client-side execute.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(tool_args_delta(
            "c1",
            "r1",
            "call-1",
            Some("get_timezone"),
            None,
        ));
        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]), false);
        let r = vals(t.on_event(
            tool_requested("call-1", "get_timezone", "", "client"),
            false,
        ));
        let args = r
            .iter()
            .find(|e| e["type"] == "TOOL_CALL_ARGS")
            .expect("emits TOOL_CALL_ARGS");
        assert_eq!(args["delta"], "{}");
    }

    #[test]
    fn late_tool_fragment_after_requested_is_ignored() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(tool_args_delta("c1", "r1", "call-1", Some("f"), Some("{}")));
        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]), false);
        let _ = t.on_event(tool_requested("call-1", "f", "{}", "worker"), false);
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
        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]), false);
        let r = vals(t.on_event(tool_requested("call-1", "get_color", "{}", "client"), true));
        assert_eq!(kinds(&r), ["TOOL_CALL_END", "RUN_FINISHED"]);
        assert!(t.terminated());
    }

    #[test]
    fn run_finished_carries_result() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let d = vals(t.on_event(
            ev(json!({
                "type": "turn.completed", "turn_id": "r1", "data": {"answer": 42},
            })),
            false,
        ));
        assert_eq!(d[0]["result"], json!({"answer": 42}));
    }

    #[test]
    fn ignores_deltas_from_other_runs() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(ev(json!({"type": "turn.started", "turn_id": "r1"})), false);
        assert!(t.on_delta(delta("c1", "OTHER", "x")).is_empty());
    }

    #[test]
    fn resumed_turn_adopts_turn_id_from_first_delta() {
        let mut t = AgUiTranslator::new("t1".into(), "r2".into());
        let a = vals(t.on_delta(delta("c3", "r1", "ok")));
        assert_eq!(kinds(&a), ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT"]);
        assert!(t.on_delta(delta("c9", "OTHER", "x")).is_empty());
    }

    #[test]
    fn non_streaming_synthesizes_text() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let c = vals(t.on_event(
            ev(json!({
                "type": "llm.call.completed", "id": "c2", "attempt": 0,
                "response": {"model": "m", "content": "hi there"},
            })),
            false,
        ));
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
        let s = vals(t.on_event(s, false));
        assert_eq!(
            kinds(&s),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"]
        );
        assert_eq!(s[0]["toolCallId"], "x");
        assert_eq!(s[0]["toolCallName"], "get_weather");
        assert_eq!(s[1]["delta"], r#"{"city":"SF"}"#);

        let r = vals(t.on_event(
            ev(json!({
                "type": "tool.call.completed", "id": "x",
                "name": "get_weather", "result": {"content":[{"type":"text","text":r#"{"temp":62}"#}]},
            })),
            false,
        ));
        assert_eq!(kinds(&r), ["TOOL_CALL_RESULT"]);
        assert_eq!(r[0]["toolCallId"], "x");
        assert_eq!(r[0]["content"], r#"{"temp":62}"#);
        assert_eq!(r[0]["role"], "tool");
    }

    #[test]
    fn tool_call_carries_parent_message_id_from_llm_call() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        t.on_event(
            ev(json!({
                "type": "llm.call.requested", "id": "c9", "attempt": 0, "llm": "claude",
                "request": {"model": "m", "messages": []}, "stream": true,
                "retry": serde_json::from_str::<Value>(RETRY).unwrap(),
            })),
            false,
        );
        let s = vals(t.on_event(
            tool_requested("x", "get_user_timezone", "{}", "client"),
            true,
        ));
        assert_eq!(s[0]["type"], "TOOL_CALL_START");
        assert_eq!(s[0]["parentMessageId"], "c9");
        assert_eq!(s[0]["toolCallId"], "x");
    }

    #[test]
    fn tool_call_without_llm_call_omits_parent_message_id() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let s = vals(t.on_event(tool_requested("x", "f", "{}", "worker"), false));
        assert!(s[0].get("parentMessageId").is_none());
    }

    #[test]
    fn parallel_tool_calls_bracket_independently() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let a = vals(t.on_event(tool_requested("a", "f", "{}", "worker"), false));
        let b = vals(t.on_event(tool_requested("b", "g", "{}", "worker"), false));
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
        let s = vals(t.on_event(
            tool_requested("x", "get_user_timezone", "{}", "client"),
            true,
        ));
        assert_eq!(
            kinds(&s),
            [
                "TOOL_CALL_START",
                "TOOL_CALL_ARGS",
                "TOOL_CALL_END",
                "RUN_FINISHED"
            ]
        );
        assert!(t.terminated());
    }

    #[test]
    fn client_answers_in_a_fresh_run_emit_results_without_yielding() {
        // The bedrock/settle path: a run whose submission carries client tool
        // answers has no batch of its own (the calls were requested in a prior
        // run), so each settle surfaces a TOOL_CALL_RESULT keyed by tool_call_id
        // and the run does not prematurely finish.
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let a = t.on_event(
            ev(json!({
                "type": "tool.call.completed", "id": "tc-1", "name": "f", "result": {"content":[{"type":"text","text":"RA"}]},
            })),
            false,
        );
        assert_eq!(kinds(&vals(a.clone())), ["TOOL_CALL_RESULT"]);
        assert_eq!(vals(a)[0]["messageId"], "tc-1");

        let b = vals(t.on_event(
            ev(json!({
                "type": "tool.call.completed", "id": "tc-2", "name": "g", "result": {"content":[{"type":"text","text":"RB"}]},
            })),
            false,
        ));
        assert_eq!(kinds(&b), ["TOOL_CALL_RESULT"]);
        assert!(!t.terminated(), "settles in a fresh run must not finish it");
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
            "type": "llm.call.completed", "id": call_id, "attempt": 0,
            "response": {"model": "m", "tool_calls": tool_calls},
        }))
    }

    #[test]
    fn parallel_client_tools_yield_once_after_whole_batch() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(llm_completed_with_tools("c1", &["a", "b"]), false);

        let first = vals(t.on_event(tool_requested("a", "set_color", "{}", "client"), false));
        assert_eq!(
            kinds(&first),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"],
            "first parallel client tool must not end the run"
        );
        assert!(!t.terminated());

        let second = vals(t.on_event(tool_requested("b", "get_color", "{}", "client"), true));
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
        assert!(t.terminated());
    }

    #[test]
    fn mixed_batch_waits_for_worker_result_before_yielding() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(llm_completed_with_tools("c1", &["w", "c"]), false);

        assert!(!t
            .on_event(tool_requested("w", "get_weather", "{}", "worker"), false)
            .iter()
            .any(|e| matches!(e, AgUiEvent::RunFinished { .. })));
        let after_client =
            vals(t.on_event(tool_requested("c", "set_color", "{}", "client"), false));
        assert_eq!(
            kinds(&after_client),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"],
            "must not yield while a worker tool in the batch is unresolved"
        );
        assert!(!t.terminated());

        let done = vals(t.on_event(
            ev(json!({
                "type": "tool.call.completed", "id": "w",
                "name": "get_weather", "result": {"content":[{"type":"text","text":r#"{"temp":62}"#}]},
            })),
            true,
        ));
        assert_eq!(kinds(&done), ["TOOL_CALL_RESULT", "RUN_FINISHED"]);
        assert!(t.terminated());
    }

    #[test]
    fn pure_worker_batch_does_not_yield() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(llm_completed_with_tools("c1", &["w1", "w2"]), false);
        let _ = t.on_event(tool_requested("w1", "f", "{}", "worker"), false);
        let _ = t.on_event(tool_requested("w2", "g", "{}", "worker"), false);
        let r1 = vals(t.on_event(
            ev(json!({
                "type": "tool.call.completed", "id": "w1", "name": "f", "result": {"content":[{"type":"text","text":"1"}]},
            })),
            false,
        ));
        let r2 = vals(t.on_event(
            ev(json!({
                "type": "tool.call.completed", "id": "w2", "name": "g", "result": {"content":[{"type":"text","text":"2"}]},
            })),
            false,
        ));
        assert_eq!(kinds(&r1), ["TOOL_CALL_RESULT"]);
        assert_eq!(kinds(&r2), ["TOOL_CALL_RESULT"]);
        assert!(!t.terminated());
        let end = vals(t.on_event(
            ev(json!({"type": "turn.completed", "turn_id": "r1"})),
            false,
        ));
        assert_eq!(kinds(&end), ["RUN_FINISHED"]);
    }

    #[test]
    fn client_tool_result_emitted_on_resume() {
        let mut t = AgUiTranslator::new("t1".into(), "r2".into());
        let r = vals(t.on_event(
            ev(json!({
                "type": "tool.call.completed", "id": "x",
                "name": "get_user_timezone", "result": {"content":[{"type":"text","text":"America/Los_Angeles"}]},
            })),
            false,
        ));
        assert_eq!(kinds(&r), ["TOOL_CALL_RESULT"]);
        assert_eq!(r[0]["toolCallId"], "x");
        assert_eq!(r[0]["content"], "America/Los_Angeles");
    }

    #[test]
    fn terminal_error_closes_open_text_then_run_error() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(delta("c1", "r1", "partial"));
        let e = vals(t.on_event(
            ev(json!({
                "type": "llm.call.errored", "id": "c1", "attempt": 0,
                "error": {"message": "boom", "code": "internal"}, "retryable": false,
            })),
            false,
        ));
        assert_eq!(kinds(&e), ["TEXT_MESSAGE_END", "RUN_ERROR"]);
        assert_eq!(e[1]["message"], "boom");
        assert!(t.terminated());
        assert!(t.open_text.is_empty());
    }

    #[test]
    fn retryable_error_is_silent() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let e = t.on_event(
            ev(json!({
                "type": "llm.call.errored", "id": "c1", "attempt": 0,
                "error": {"message": "temporary", "code": "internal"}, "retryable": true,
            })),
            false,
        );
        assert!(e.is_empty());
        assert!(!t.terminated());
    }

    #[test]
    fn nothing_emitted_after_termination() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(
            ev(json!({"type": "turn.completed", "turn_id": "r1"})),
            false,
        );
        assert!(t.terminated());
        assert!(t.on_delta(delta("c1", "r1", "late")).is_empty());
        assert!(t
            .on_event(
                ev(json!({"type": "turn.completed", "turn_id": "r1"})),
                false
            )
            .is_empty());
    }

    #[test]
    fn session_interrupt_finishes_run_with_interrupt_outcome() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_delta(delta("c1", "r1", "thinking..."));
        let e = vals(t.on_event(
            ev(json!({
                "type": "session.interrupted",
                "interrupt_id": "int-1",
                "origin": "frontend",
                "reason": "confirmation",
                "payload": {
                    "message": "Send the email?",
                    "toolCallId": "tc-1",
                    "responseSchema": {"type": "object"},
                },
            })),
            false,
        ));
        assert_eq!(kinds(&e), ["TEXT_MESSAGE_END", "RUN_FINISHED"]);
        let finished = &e[1];
        assert_eq!(finished["outcome"]["type"], "interrupt");
        let interrupt = &finished["outcome"]["interrupts"][0];
        assert_eq!(interrupt["id"], "int-1");
        assert_eq!(interrupt["reason"], "confirmation");
        assert_eq!(interrupt["message"], "Send the email?");
        assert_eq!(interrupt["toolCallId"], "tc-1");
        assert_eq!(interrupt["responseSchema"]["type"], "object");
        assert!(t.terminated());
    }

    fn sub_agent_requested(session: &str, agent: &str, tool_id: &str) -> EventPayload {
        ev(json!({
            "type": "sub_agent.requested",
            "id": session,
            "agent_id": agent,
            "tool_call_id": tool_id,
            "retry": serde_json::from_str::<Value>(RETRY).unwrap(),
        }))
    }

    fn sub_agent_turn_completed(session: &str, data: Value) -> EventPayload {
        ev(json!({
            "type": "sub_agent.turn_completed",
            "id": session,
            "data": data,
        }))
    }

    #[test]
    fn streamed_sub_agent_call_closes_then_renders_result() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        // The delegating tool call streams first, exactly like a normal tool call.
        let a = vals(t.on_delta(tool_args_delta(
            "c1",
            "r1",
            "call-1",
            Some("weather"),
            Some(r#"{"message":"Paris?"}"#),
        )));
        assert_eq!(kinds(&a), ["TOOL_CALL_START", "TOOL_CALL_ARGS"]);
        let _ = t.on_event(llm_completed_with_tools("c1", &["call-1"]), false);

        // `sub_agent.requested` stands in for `tool.call.requested`: it closes the bracket.
        let req = vals(t.on_event(sub_agent_requested("child-1", "weather", "call-1"), false));
        assert_eq!(kinds(&req), ["TOOL_CALL_END"]);
        assert!(t.open_tools.is_empty());
        assert!(!t.terminated(), "a lone sub-agent must not yield the run");

        // The child's answer surfaces as the tool result, keyed by the tool call id.
        let done = vals(t.on_event(
            sub_agent_turn_completed("child-1", json!("Clear, 68F.")),
            false,
        ));
        assert_eq!(kinds(&done), ["TOOL_CALL_RESULT"]);
        assert_eq!(done[0]["toolCallId"], "call-1");
        assert_eq!(done[0]["content"], "Clear, 68F.");
        assert!(!t.terminated());

        // The turn finishes on the parent's own completion, not the sub-agent's.
        let end = vals(t.on_event(
            ev(json!({"type": "turn.completed", "turn_id": "r1"})),
            false,
        ));
        assert_eq!(kinds(&end), ["RUN_FINISHED"]);
    }

    #[test]
    fn non_string_sub_agent_result_is_serialized_as_json() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(sub_agent_requested("child-1", "lookup", "call-1"), false);
        let done = vals(t.on_event(
            sub_agent_turn_completed("child-1", json!({"tempF": 68})),
            true,
        ));
        assert_eq!(done[0]["content"], r#"{"tempF":68}"#);
    }

    #[test]
    fn unstreamed_sub_agent_call_synthesizes_the_bracket() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let req = vals(t.on_event(sub_agent_requested("child-1", "weather", "call-1"), false));
        assert_eq!(kinds(&req), ["TOOL_CALL_START", "TOOL_CALL_END"]);
        assert_eq!(req[0]["toolCallName"], "weather");
        assert_eq!(req[0]["toolCallId"], "call-1");
    }

    #[test]
    fn sub_agent_error_surfaces_as_a_result() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(sub_agent_requested("child-1", "weather", "call-1"), false);
        let err = vals(t.on_event(
            ev(json!({
                "type": "sub_agent.errored",
                "id": "child-1",
                "error": {"message": "child boom", "code": "internal"},
            })),
            false,
        ));
        assert_eq!(kinds(&err), ["TOOL_CALL_RESULT"]);
        assert_eq!(err[0]["toolCallId"], "call-1");
        assert_eq!(err[0]["content"], "child boom");
    }

    #[test]
    fn mixed_batch_waits_for_sub_agent_before_yielding() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let _ = t.on_event(llm_completed_with_tools("c1", &["sub", "c"]), false);

        let after_sub = vals(t.on_event(sub_agent_requested("child-1", "weather", "sub"), false));
        assert_eq!(kinds(&after_sub), ["TOOL_CALL_START", "TOOL_CALL_END"]);
        assert!(!t.terminated());

        let after_client =
            vals(t.on_event(tool_requested("c", "set_color", "{}", "client"), false));
        assert_eq!(
            kinds(&after_client),
            ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"],
            "must not yield while the sub-agent in the batch is unresolved"
        );
        assert!(!t.terminated());

        let done = vals(t.on_event(sub_agent_turn_completed("child-1", json!("done")), true));
        assert_eq!(kinds(&done), ["TOOL_CALL_RESULT", "RUN_FINISHED"]);
        assert!(t.terminated());
    }

    #[test]
    fn normal_completion_has_no_outcome() {
        let mut t = AgUiTranslator::new("t1".into(), "r1".into());
        let e = vals(t.on_event(
            ev(json!({"type": "turn.completed", "turn_id": "r1"})),
            false,
        ));
        assert_eq!(kinds(&e), ["RUN_FINISHED"]);
        assert!(e[0].get("outcome").is_none());
    }
}
