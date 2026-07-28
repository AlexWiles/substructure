use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::decision::{Action, LlmHandler, ToolHandler, Trigger};
use super::events::*;
use super::reconcile::{landing_leaf, plan_reconcile};
use super::state::{
    json_to_string, new_call_id, ConnectorSyncState, EffectTracking, SessionState, SessionStatus,
};
use super::tool_contract::{declared_tool, output_violation, DeclaredTool};
use crate::connectors::{filter, RemoteTool};
use crate::protocol::{
    AgentConfig, ClientAppend, ClientContext, ClientMessage, ClientMessages, ClientPayload,
    Content, ContentPart, DraftMessage, EffectStatus, ErrorCode, ImageUrl, InterruptOrigin,
    LlmFormat, LlmRequest, LlmResponse, NewMessage, RetryPolicy, Role, SessionOwner, WorkerState,
};
use crate::runtime::Caller;

#[derive(Debug, Clone)]
pub enum CommandPayload {
    CreateSession {
        agent_id: String,
        owner: SessionOwner,
        ancestry: Vec<String>,
        worker_retry: RetryPolicy,
    },
    SubmitClientPayload {
        payload: ClientPayload,
        turn_id: Option<String>,
    },
    SendMessage {
        message: DraftMessage,
        #[allow(dead_code)]
        stream: bool,
        turn_id: Option<String>,
        parent_id: Option<String>,
    },
    RequestLlmCall {
        call_id: String,
        request: LlmRequest,
        stream: bool,
        retry: RetryPolicy,
        handler: LlmHandler,
        format: Option<LlmFormat>,
    },
    CompleteLlmCall {
        call_id: String,
        /// `None` = settle the current attempt; `Some` fences a stale executor.
        attempt: Option<u32>,
        response: LlmResponse,
    },
    FailLlmCall {
        call_id: String,
        attempt: Option<u32>,
        error: String,
        retryable: bool,
        code: Option<ErrorCode>,
        detail: Option<serde_json::Value>,
    },
    RequestToolCall {
        tool_call_id: String,
        name: String,
        arguments: String,
        retry: RetryPolicy,
    },
    CompleteToolCall {
        tool_call_id: String,
        attempt: Option<u32>,
        result: String,
    },
    FailToolCall {
        tool_call_id: String,
        attempt: Option<u32>,
        error: String,
        retryable: bool,
    },
    CompleteConnectorSync {
        connection_id: String,
        attempt: Option<u32>,
        prefix: Option<String>,
        tools: Vec<RemoteTool>,
    },
    FailConnectorSync {
        connection_id: String,
        attempt: Option<u32>,
        error: String,
        retryable: bool,
        needs_reauth: bool,
    },
    RequestSubAgent {
        session_id: String,
        agent_id: String,
        tool_call_id: String,
        retry: RetryPolicy,
    },
    StartSubAgent {
        session_id: String,
    },
    FailSubAgent {
        session_id: String,
        error: String,
        retryable: bool,
    },
    CompleteSubAgentTurn {
        session_id: String,
        agent_id: String,
        turn_id: String,
        data: serde_json::Value,
        cost: Decimal,
        token_usage: BTreeMap<String, u64>,
    },
    Interrupt {
        interrupt_id: String,
        reason: String,
        payload: serde_json::Value,
    },
    ResumeInterrupt {
        interrupt_id: String,
        payload: serde_json::Value,
    },
    SubmitWorkerDecision {
        decision_id: String,
        transcript: Vec<DraftMessage>,
        actions: Vec<Action>,
        /// `None` = no opinion, keep the current state.
        state: Option<WorkerState>,
        /// `None` = no opinion, keep the current agent config.
        agent: Option<AgentConfig>,
    },
    FailWorkerDecision {
        decision_id: String,
        error: String,
        retryable: bool,
    },
    CancelSession,
    /// The agent finished its turn: begin finalization by notifying the worker
    /// (`turn.finished`), deferring completion.
    FinishTurn {
        data: serde_json::Value,
    },
    /// The finalizer settled: complete the turn (emit `TurnCompleted` + `SessionDone`).
    CompleteTurn,
    Wake {
        now: DateTime<Utc>,
    },
    /// Boot-time recovery: fail in-flight work whose executor died with the
    /// process — pending worker decisions (reply rides the severed dispatch) and
    /// pending server-handled LLM calls (the awaiting future is gone) — so their
    /// retry policies re-issue now instead of at the deadline.
    ReconcileDispatch,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum SessionError {
    #[error("session has not been created")]
    SessionNotCreated,
    #[error("session already exists")]
    SessionAlreadyCreated,
    #[error("session is interrupted")]
    SessionInterrupted,
    #[error("session start failed")]
    SessionStartFailed,
    #[error("turn already active: {turn_id}")]
    TurnAlreadyActive { turn_id: String },
    #[error("turn already completed: {turn_id}")]
    TurnAlreadyCompleted { turn_id: String },
    #[error("session has no active turn")]
    NoActiveTurn,
    #[error("client subject is required")]
    MissingSubject,
    #[error("session access denied")]
    SessionAccessDenied,
    #[error("effect not found")]
    EffectNotFound,
    #[error("effect is not pending")]
    EffectNotPending,
    #[error("effect attempt mismatch")]
    EffectAttemptMismatch,
    #[error("caller may not settle this effect")]
    EffectWrongHandler,
}

/// A tool message in a submitted view that answers a still-pending client call.
struct Completion {
    /// Position in the normalized view (used to line up with the reconcile plan).
    index: usize,
    tool_call_id: String,
    name: String,
    result: String,
}

impl SessionState {
    fn ensure_internal(caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            _ => Err(SessionError::SessionAccessDenied),
        }
    }

    fn ensure_machine_or_system(caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::Machine { .. } => Ok(()),
            Caller::Frontend { .. } => Err(SessionError::SessionAccessDenied),
        }
    }

    fn ensure_tenant_matches(caller: &Caller, tenant_id: &str) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            Caller::Machine {
                tenant_id: caller_tenant,
                ..
            }
            | Caller::Frontend {
                tenant_id: caller_tenant,
                ..
            } => {
                if caller_tenant != tenant_id {
                    return Err(SessionError::SessionAccessDenied);
                }
                Ok(())
            }
        }
    }

    fn caller_interrupt_origin(caller: &Caller) -> InterruptOrigin {
        match caller {
            Caller::System { .. } => InterruptOrigin::System,
            Caller::Machine { .. } => InterruptOrigin::Machine,
            Caller::Frontend { .. } => InterruptOrigin::Frontend,
        }
    }

    fn ensure_owns_session(&self, caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::Machine { .. } => Ok(()),
            Caller::Frontend { user_id, .. } => {
                let owner = self
                    .owner
                    .as_ref()
                    .ok_or(SessionError::SessionAccessDenied)?;
                if owner.id.as_deref() != Some(user_id.as_str()) {
                    return Err(SessionError::SessionAccessDenied);
                }
                Ok(())
            }
        }
    }

    fn check_tool_call_caller(
        &self,
        tc: &super::state::ToolCallState,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        self.ensure_owns_session(caller)?;
        if matches!(caller, Caller::Frontend { .. }) && tc.handler != ToolHandler::Client {
            return Err(SessionError::EffectWrongHandler);
        }
        Ok(())
    }

    fn check_llm_call_caller(
        &self,
        call: Option<&super::state::LlmCallState>,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            Caller::Frontend { .. } => Err(SessionError::EffectWrongHandler),
            Caller::Machine { .. } => match call {
                Some(c) if c.handler == LlmHandler::Worker => Ok(()),
                Some(_) => Err(SessionError::EffectWrongHandler),
                None => Err(SessionError::EffectNotFound),
            },
        }
    }

    pub fn handle(
        &self,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Result<Vec<EventPayload>, SessionError> {
        match (&self.agent_id, cmd) {
            (
                None,
                CommandPayload::CreateSession {
                    agent_id,
                    owner,
                    ancestry,
                    worker_retry,
                },
            ) => {
                Self::ensure_tenant_matches(caller, &owner.tenant_id)?;
                if let Caller::Frontend { user_id, .. } = caller {
                    if owner.id.as_deref() != Some(user_id.as_str()) {
                        return Err(SessionError::SessionAccessDenied);
                    }
                }
                let mut events = vec![EventPayload::SessionCreated(Box::new(SessionCreated {
                    agent_id,
                    identity: owner,
                    ancestry,
                    worker_retry,
                }))];
                // The session's first decision: the worker declares its identity
                // before any client input. A brand-new session has no pending
                // decision, so this is requested (not queued) directly.
                let start = self.emit_decision_request(&events, Trigger::SessionStart);
                events.extend(start);
                Ok(events)
            }
            (Some(_), CommandPayload::CreateSession { .. }) => {
                Err(SessionError::SessionAlreadyCreated)
            }
            (None, _) => Err(SessionError::SessionNotCreated),
            (Some(_), cmd) => self.handle_active(cmd, caller),
        }
    }

    // Sole producer of NewMessage; the tree changes only on decision submit.
    // Executes the plan from `plan_reconcile` — the one interpreter of "what
    // recording this list writes" — so submit-time classification and delivery
    // annotation can't drift from what actually lands in the tree.
    //
    // Returns the events and the post-batch head. A non-empty view writing
    // nothing selects its leaf as the branch: the head rebases via `HeadMoved`.
    fn reconcile_transcript(
        &self,
        transcript: Vec<DraftMessage>,
    ) -> (Vec<EventPayload>, Option<String>) {
        // Normalize again at the write seam: a view frozen into a queued trigger
        // can race the decision that records a tool node (the echo predates it),
        // so folding at client-submit time alone would fork the tree here.
        let transcript = self.normalize_client_view(transcript);
        let known: std::collections::HashSet<&str> =
            self.nodes.iter().map(|n| n.message.id.as_str()).collect();
        let plan = plan_reconcile(&known, &transcript);
        let mut events = Vec::with_capacity(plan.len());
        let mut plan_iter = plan.iter().peekable();
        let mut parent: Option<String> = None;
        for (index, msg) in transcript.into_iter().enumerate() {
            match plan_iter.peek() {
                Some(write) if write.index == index => {
                    let rerecord = write.rerecord;
                    plan_iter.next();
                    // Re-record known ids in the news region as fresh nodes so
                    // the branch stays a chain instead of grafting onto the old.
                    let msg = if rerecord {
                        msg.rerecord()
                    } else {
                        msg.record()
                    };
                    let id = msg.id.clone();
                    events.push(EventPayload::NewMessage(NewMessage {
                        message: msg,
                        parent_id: parent.take(),
                    }));
                    parent = Some(id);
                }
                // Known prefix before the news: advance the parent cursor only.
                _ => parent = msg.id,
            }
        }
        if !events.is_empty() {
            return (events, parent);
        }
        match parent {
            Some(leaf) if self.head_id.as_deref() != Some(leaf.as_str()) => {
                let moved = EventPayload::HeadMoved(HeadMoved {
                    head_id: leaf.clone(),
                });
                (vec![moved], Some(leaf))
            }
            _ => (events, self.head_id.clone()),
        }
    }

    /// The `output` schema the settling tool was declared with, resolved by
    /// lineage on the active path. `None` when the tool declared no output
    /// contract or the call has no resolvable spec.
    fn declared_output_schema(&self, tool_call_id: &str, name: &str) -> Option<serde_json::Value> {
        let tree = self.message_tree();
        let path = tree.path_to(tree.head_id.as_deref()?);
        match declared_tool(tool_call_id, name, &path, &self.llm_calls) {
            DeclaredTool::Declared(tool) => tool.output.clone(),
            _ => None,
        }
    }

    fn emit_tool_result(
        &self,
        batch: &[EventPayload],
        tool_call_id: String,
        name: String,
        output: String,
        is_error: bool,
    ) -> Vec<EventPayload> {
        let (result, error) = if is_error {
            (None, Some(output))
        } else {
            (Some(output), None)
        };
        self.emit_decision_request(
            batch,
            Trigger::ToolFinished {
                id: tool_call_id,
                ok: !is_error,
                name,
                result,
                error,
            },
        )
    }

    fn emit_sub_agent_result(
        &self,
        batch: &[EventPayload],
        session_id: String,
        tool_call_id: String,
        agent_id: String,
        output: String,
        is_error: bool,
    ) -> Vec<EventPayload> {
        let (result, error) = if is_error {
            (None, Some(output))
        } else {
            (Some(output), None)
        };
        self.emit_decision_request(
            batch,
            Trigger::SubAgentFinished {
                id: tool_call_id,
                ok: !is_error,
                session_id,
                agent_id,
                result,
                error,
            },
        )
    }

    /// Whether the decision's effect is anchored off the path to `leaf`.
    fn stale_decision(&self, leaf: Option<&str>, trigger: &Trigger) -> bool {
        !self.anchor_on_path(leaf, self.trigger_anchor(trigger))
    }

    /// Void events for every effect matching `stranded`.
    fn void_effects(
        &self,
        stranded: impl Fn(&EffectTracking, Option<&str>) -> bool,
    ) -> Vec<EventPayload> {
        let mut events = Vec::new();
        for tc in self.tool_calls.values() {
            if stranded(&tc.tracking, tc.anchor.as_deref()) {
                events.push(EventPayload::CallVoided(CallVoided {
                    kind: EffectKind::ToolCall,
                    id: tc.tool_call_id.clone(),
                    session_id: None,
                }));
            }
        }
        for call in self.llm_calls.values() {
            if stranded(&call.tracking, call.anchor.as_deref()) {
                events.push(EventPayload::CallVoided(CallVoided {
                    kind: EffectKind::LlmCall,
                    id: call.call_id.clone(),
                    session_id: None,
                }));
            }
        }
        for sa in self.sub_agent_calls.values() {
            if stranded(&sa.tracking, sa.anchor.as_deref()) {
                events.push(EventPayload::CallVoided(CallVoided {
                    kind: EffectKind::SubAgent,
                    id: sa.tool_call_id.clone(),
                    session_id: Some(sa.session_id.clone()),
                }));
            }
        }
        events
    }

    /// Voids pending LLM calls on the path to `leaf` (in-batch ones
    /// unconditionally), sparing tools and sub-agents.
    fn void_llm_calls_for_interrupt(
        &self,
        batch: &[EventPayload],
        leaf: Option<&str>,
    ) -> Vec<EventPayload> {
        let mut ids: Vec<String> = self
            .llm_calls
            .values()
            .filter(|c| c.tracking.status == EffectStatus::Pending)
            .filter(|c| self.anchor_on_path(leaf, c.anchor.as_deref()))
            .map(|c| c.call_id.clone())
            .collect();
        ids.extend(batch.iter().filter_map(|e| match e {
            EventPayload::LlmCallRequested(r) => Some(r.call_id.clone()),
            _ => None,
        }));
        ids.into_iter()
            .filter(|id| {
                !batch
                    .iter()
                    .any(|e| matches!(e, EventPayload::CallVoided(v) if v.id == *id))
            })
            .map(|id| {
                EventPayload::CallVoided(CallVoided {
                    kind: EffectKind::LlmCall,
                    id,
                    session_id: None,
                })
            })
            .collect()
    }

    /// Whether `batch` already voided this call.
    fn batch_voided(batch: &[EventPayload], kind: EffectKind, id: &str) -> bool {
        batch
            .iter()
            .any(|e| matches!(e, EventPayload::CallVoided(v) if v.kind == kind && v.id == id))
    }

    /// Void work and drop undelivered decisions anchored off the retained path.
    fn void_stranded_work(&self, leaf: Option<&str>) -> Vec<EventPayload> {
        let mut events = self.void_effects(|tracking, anchor| {
            matches!(
                tracking.status,
                EffectStatus::Pending | EffectStatus::RetryScheduled
            ) && !self.anchor_on_path(leaf, anchor)
        });
        for wd in self.worker_decisions.values() {
            if matches!(
                wd.tracking.status,
                EffectStatus::Queued | EffectStatus::RetryScheduled
            ) && self.stale_decision(leaf, &wd.trigger)
            {
                events.push(EventPayload::DecisionRequestDropped(
                    DecisionRequestDropped {
                        decision_id: wd.decision_id.clone(),
                    },
                ));
            }
        }
        events
    }

    /// `(tool_call_id, name, content)` for a transcript tool message answering a pending client tool call.
    fn pending_client_result(&self, m: &DraftMessage) -> Option<(String, String, String)> {
        if m.role != Role::Tool {
            return None;
        }
        let id = m.tool_call_id.as_deref()?;
        let tc = self.tool_calls.get(id)?;
        if tc.handler != ToolHandler::Client || tc.tracking.status != EffectStatus::Pending {
            return None;
        }
        let content = m
            .content
            .as_ref()
            .map(Content::text_owned)
            .unwrap_or_default();
        Some((id.to_string(), tc.name.clone(), content))
    }

    /// The id of a recorded tool-result node answering `tool_call_id`, if any —
    /// preferring one on the active path, then the most recently recorded.
    fn recorded_result_node(
        &self,
        tool_call_id: &str,
        on_path: &std::collections::HashSet<&str>,
    ) -> Option<String> {
        let mut best: Option<&str> = None;
        let mut best_on_path = false;
        for m in self.nodes.iter().map(|n| &n.message) {
            if m.role != Role::Tool || m.tool_call_id.as_deref() != Some(tool_call_id) {
                continue;
            }
            let on = on_path.contains(m.id.as_str());
            if on {
                best = Some(&m.id);
                best_on_path = true;
            } else if !best_on_path {
                best = Some(&m.id);
            }
        }
        best.map(str::to_string)
    }

    /// Fold client echoes of already-recorded tool results back onto their
    /// nodes: an unknown-id tool message whose call already has a recorded node
    /// IS that node's echo, so it adopts the node's id and the tree sees a
    /// resend rather than a fork. Identification, not deletion — nothing is
    /// dropped, so the reconcile plan stays coherent.
    pub fn normalize_client_view(&self, messages: Vec<DraftMessage>) -> Vec<DraftMessage> {
        let known: std::collections::HashSet<&str> =
            self.nodes.iter().map(|n| n.message.id.as_str()).collect();
        let on_path: std::collections::HashSet<&str> = self
            .head_id
            .as_deref()
            .map(|h| self.path_ids(h))
            .unwrap_or_default();
        messages
            .into_iter()
            .map(|m| {
                let is_known = m.id.as_deref().is_some_and(|id| known.contains(id));
                if m.role == Role::Tool && !is_known {
                    if let Some(node_id) = m
                        .tool_call_id
                        .as_deref()
                        .and_then(|tcid| self.recorded_result_node(tcid, &on_path))
                    {
                        return DraftMessage {
                            id: Some(node_id),
                            ..m
                        };
                    }
                }
                m
            })
            .collect()
    }

    fn emit_decision_request(&self, batch: &[EventPayload], trigger: Trigger) -> Vec<EventPayload> {
        self.emit_decision_request_at(batch, trigger, self.head_id.as_deref())
    }

    /// Every decision is created by a queued event carrying its trigger — the
    /// single stored copy. A promotion marker follows immediately unless an
    /// interrupt on `gate_leaf`'s path (or a live decision) parks it.
    fn emit_decision_request_at(
        &self,
        batch: &[EventPayload],
        trigger: Trigger,
        gate_leaf: Option<&str>,
    ) -> Vec<EventPayload> {
        let parked = self.has_pending_worker_decision()
            || self.has_unsettled_session_start()
            || self.connectors_unready(batch, gate_leaf)
            || self.active_interrupt_for(gate_leaf).is_some()
            || batch
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_)));
        let decision_id = new_call_id();
        let mut events = vec![EventPayload::DecisionRequestQueued(DecisionRequestQueued {
            decision_id: decision_id.clone(),
            trigger,
        })];
        if !parked {
            events.push(EventPayload::WorkerDecisionRequested(
                WorkerDecisionRequested { decision_id },
            ));
        }
        events
    }

    /// Whether a connection the config in force names has not settled. A
    /// decision that would prompt the model parks behind this: the config names
    /// tools the engine has not fetched, so the turn cannot be authored against
    /// it yet. The same sentence as `has_unsettled_session_start`, one level down.
    ///
    /// Also checks the batch, because the config write that introduces a
    /// connector emits its fetch in the same batch — invisible to `self`, which
    /// is the pre-batch state.
    ///
    /// **Every site that emits `WorkerDecisionRequested` must consult this.**
    /// There are five, and they do not share a guard: `emit_decision_request_at`,
    /// the promotion pass in `SubmitWorkerDecision`, `release_decisions`, the
    /// resume in `ResumeInterrupt`, and both loops in `handle_wake`. Waiting is
    /// always bounded — a fetch carries a deadline and always settles, failed if
    /// nothing else — so blocking on this can delay a decision but never strand
    /// one.
    fn connectors_unready(&self, batch: &[EventPayload], leaf: Option<&str>) -> bool {
        self.has_pending_connector_sync(leaf)
            || batch
                .iter()
                .any(|e| matches!(e, EventPayload::ConnectorSyncRequested(_)))
    }

    /// Append the connector tools in force to a request's tool list.
    ///
    /// Done here rather than where the request is authored, because both authors
    /// need it and neither can do it: the engine's proposal builds tools from
    /// `tools_as_llm`, and a worker hand-authoring a call cannot name a connector
    /// tool it has never seen. The agent declared the connector, so its tools are
    /// part of every call that config authorizes.
    ///
    /// Resolved against the pre-batch state, so a connector takes effect from the
    /// decision after the one declaring it — by then its fetch has settled, which
    /// is what the decision gate is for.
    fn with_connector_tools(&self, request: LlmRequest) -> LlmRequest {
        let connector_tools = self.connector_tools(self.head_id.as_deref()).tools;
        if connector_tools.is_empty() {
            return request;
        }
        let mut tools = request.tools.unwrap_or_default();
        // Skip what the request already offers. A re-prompt is built from the
        // previous call's stored spec, which already carries these, and every
        // provider rejects a duplicate tool name.
        for tool in &connector_tools {
            if !tools.iter().any(|t| t.name == tool.name) {
                tools.push(tool.to_llm_tool());
            }
        }
        LlmRequest {
            tools: Some(tools),
            ..request
        }
    }

    /// The in-flight fetch `connection_id` names, fenced against a stale
    /// executor the same way a tool or LLM settle is.
    fn check_connector_sync(
        &self,
        connection_id: &str,
        attempt: Option<u32>,
    ) -> Result<&ConnectorSyncState, SessionError> {
        let sync = self
            .connector_syncs
            .get(connection_id)
            .ok_or(SessionError::EffectNotFound)?;
        if sync.tracking.status != EffectStatus::Pending {
            return Err(SessionError::EffectNotPending);
        }
        if attempt.is_some_and(|a| a != sync.tracking.retry.attempts) {
            return Err(SessionError::EffectAttemptMismatch);
        }
        Ok(sync)
    }

    /// Report what the filter did to a fresh offer.
    ///
    /// Logged once per fetch rather than carried on every decision: these are
    /// facts about the config, identical on every turn, and a worker cannot act
    /// on them anyway — an `include` that matches nothing is a typo for whoever
    /// wrote the config, not a runtime condition for the agent to handle.
    fn report_filter(&self, connection_id: &str, offered: &[RemoteTool], prefix: Option<&str>) {
        let Some(connector) = self
            .resolve_agent_for(self.head_id.as_deref())
            .and_then(|c| c.mcp.into_iter().find(|c| c.id == connection_id))
        else {
            return;
        };
        let r = filter::resolve(&connector, offered, prefix);
        tracing::info!(
            connection = %connection_id,
            offered = r.offered,
            resolved = r.tools.len(),
            "fetched connector tools"
        );
        if !r.unmatched_include.is_empty() {
            tracing::warn!(
                connection = %connection_id,
                patterns = ?r.unmatched_include,
                "connector include patterns matched no tool"
            );
        }
        if !r.oversized.is_empty() {
            tracing::warn!(
                connection = %connection_id,
                tools = ?r.oversized,
                "connector tool names too long to offer; shorten the connection id or turn off prefixing"
            );
        }
        if r.unannotated > 0 {
            tracing::warn!(
                connection = %connection_id,
                count = r.unannotated,
                "connector tools dropped for carrying no annotation to test"
            );
        }
    }

    /// Promote the next queued decision now that a fetch settled, if nothing
    /// else parks it. `batch` carries the settling events, which `self` — the
    /// pre-batch state — cannot see.
    fn release_decisions(&self, batch: &[EventPayload]) -> Vec<EventPayload> {
        let settled: std::collections::HashSet<&str> = batch
            .iter()
            .filter_map(|e| match e {
                EventPayload::ConnectorSyncCompleted(p) => Some(p.connection_id.as_str()),
                EventPayload::ConnectorSyncErrored(p) => Some(p.connection_id.as_str()),
                _ => None,
            })
            .collect();
        if self.has_pending_worker_decision() || self.has_unsettled_session_start() {
            return Vec::new();
        }
        // Still waiting on a sibling connection: the config needs all of them.
        let waiting = self
            .connector_syncs
            .values()
            .any(|s| s.is_in_flight() && !settled.contains(s.connection_id.as_str()));
        if waiting {
            return Vec::new();
        }
        self.queued_decisions()
            .into_iter()
            .find(|d| !self.decision_parked(&d.trigger))
            .map(|d| {
                EventPayload::WorkerDecisionRequested(WorkerDecisionRequested {
                    decision_id: d.decision_id.clone(),
                })
            })
            .into_iter()
            .collect()
    }

    /// Fetch every connection `config` names that this session has never fetched.
    /// Keyed on the connection, so a config rewritten for unrelated reasons —
    /// client tools appearing, a branch switch — costs nothing.
    fn sync_connectors(&self, config: &AgentConfig) -> Vec<EventPayload> {
        config
            .mcp
            .iter()
            .filter(|c| !self.connector_syncs.contains_key(&c.id))
            .map(|c| {
                EventPayload::ConnectorSyncRequested(ConnectorSyncRequested {
                    connection_id: c.id.clone(),
                    attempt: 0,
                    retry: RetryPolicy::connector_default(),
                })
            })
            .collect()
    }

    /// A failed `session.start` left the session with no config, so the client's
    /// decision can't run against it. Re-queue the start ahead of that decision:
    /// `queued_decisions` orders by arrival and `has_unsettled_session_start`
    /// parks the rest, so the retry runs first and the message follows it once a
    /// config lands. Recovery is the next message, not a reset — fix the worker
    /// and say something.
    fn restart_session(&self, batch: &[EventPayload]) -> Vec<EventPayload> {
        self.emit_decision_request(batch, Trigger::SessionStart)
    }

    /// The run is over, so nothing queued can still run: drop them all rather
    /// than promote a decision whose turn already ended. After a terminally-failed
    /// `session.start` this is also the only correct move — no config was ever
    /// resolved, so a promoted turn would settle as a silent no-op.
    fn drop_queued_decisions(&self) -> Vec<EventPayload> {
        self.queued_decisions()
            .into_iter()
            .map(|d| {
                EventPayload::DecisionRequestDropped(DecisionRequestDropped {
                    decision_id: d.decision_id.clone(),
                })
            })
            .collect()
    }

    /// Emit the run terminal from the in-flight `finalizing` output: `TurnCompleted`
    /// (carrying `error` when the finalizer failed terminally) + `SessionDone`. Falls
    /// back to a bare `SessionDone` if nothing is finalizing.
    fn finalize_run(&self, error: Option<String>) -> Vec<EventPayload> {
        match self.finalizing.clone() {
            Some(f) => vec![
                EventPayload::TurnCompleted(TurnCompleted {
                    turn_id: f.turn_id,
                    data: f.data,
                    turn_cost: f.cost,
                    turn_token_usage: f.usage,
                    error,
                }),
                EventPayload::SessionDone(SessionDone {}),
            ],
            None => vec![EventPayload::SessionDone(SessionDone {})],
        }
    }

    /// The run terminal for a terminally-failed decision. `TurnCompleted` is the
    /// only terminal every consumer watches, so a failure that never reaches it
    /// strands the turn open forever: no reply, no error, no timeout. Void what
    /// was in flight, drop what was queued, then end the turn as a failed run.
    fn fail_run(&self, error: String) -> Vec<EventPayload> {
        // The finalizer itself failed: the frozen output is already held, and
        // `finalize_run` reports it against the turn that produced it.
        if self.finalizing.is_some() {
            return self.finalize_run(Some(error));
        }
        let mut events = self.void_effects(|tracking, _| {
            matches!(
                tracking.status,
                EffectStatus::Pending | EffectStatus::RetryScheduled
            )
        });
        events.extend(self.drop_queued_decisions());
        // No turn to end — a `session.start` that failed before any client input.
        // The decision's error event is the whole record.
        let Some(turn_id) = self.turn_id.clone() else {
            return events;
        };
        if self.completed_turn_ids.contains(&turn_id) {
            return events;
        }
        events.push(EventPayload::TurnCompleted(TurnCompleted {
            turn_id,
            data: serde_json::Value::Null,
            turn_cost: self.turn_cost,
            turn_token_usage: self.turn_token_usage.clone(),
            error: Some(error),
        }));
        events.push(EventPayload::SessionDone(SessionDone {}));
        events
    }

    fn handle_active(
        &self,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Result<Vec<EventPayload>, SessionError> {
        if let Some(owner) = self.owner.as_ref() {
            Self::ensure_tenant_matches(caller, &owner.tenant_id)?;
        }
        match cmd {
            CommandPayload::CreateSession { .. } => Err(SessionError::SessionAlreadyCreated),

            CommandPayload::SubmitClientPayload { payload, turn_id } => {
                self.ensure_owns_session(caller)?;

                if let Some(ref tid) = turn_id {
                    if self.completed_turn_ids.contains(tid) {
                        return Err(SessionError::TurnAlreadyCompleted {
                            turn_id: tid.clone(),
                        });
                    }
                    if self.turn_id.as_ref() == Some(tid) {
                        return Err(SessionError::TurnAlreadyActive {
                            turn_id: tid.clone(),
                        });
                    }
                }

                let mut events: Vec<EventPayload> = Vec::new();
                if let Some(turn_id) = turn_id {
                    events.push(EventPayload::TurnStarted(TurnStarted { turn_id }));
                }

                match payload {
                    ClientPayload::Message(ClientMessage { message, stream: _ }) => {
                        if self.head_parked() && message.role == Role::User {
                            return Err(SessionError::SessionInterrupted);
                        }
                        if self.session_start_failed && message.role == Role::User {
                            let restart = self.restart_session(&events);
                            events.extend(restart);
                        }
                        if message.role == Role::User {
                            let request = self.emit_decision_request(
                                &events,
                                Trigger::ClientMessage {
                                    messages: vec![message],
                                    client: ClientContext::default(),
                                },
                            );
                            events.extend(request);
                        }
                    }
                    ClientPayload::Append(ClientAppend {
                        messages,
                        stream: _,
                        client,
                    }) => {
                        if self.head_parked() && messages.iter().any(|m| m.role == Role::User) {
                            return Err(SessionError::SessionInterrupted);
                        }
                        if self.session_start_failed
                            && messages.iter().any(|m| m.role == Role::User)
                        {
                            let restart = self.restart_session(&events);
                            events.extend(restart);
                        }
                        let request = self.emit_decision_request(
                            &events,
                            Trigger::ClientMessage { messages, client },
                        );
                        events.extend(request);
                    }
                    ClientPayload::Messages(ClientMessages {
                        messages,
                        stream: _,
                        client,
                    }) => {
                        // Fold client echoes of already-recorded results onto their
                        // nodes so the tree sees a resend, not a fork.
                        let messages = self.normalize_client_view(messages);

                        // Answers to still-pending client calls, first-wins per
                        // call id (a repeat within one view has no recorded node
                        // to fold onto, so the later copy is dropped).
                        let mut seen = std::collections::HashSet::new();
                        let completions: Vec<Completion> = messages
                            .iter()
                            .enumerate()
                            .filter_map(|(index, m)| {
                                self.pending_client_result(m)
                                    .map(|(tc, name, result)| Completion {
                                        index,
                                        tool_call_id: tc,
                                        name,
                                        result,
                                    })
                            })
                            .filter(|c| seen.insert(c.tool_call_id.clone()))
                            .collect();

                        // What recording this view would write, by the one
                        // reconcile interpreter; classification reads this plan
                        // rather than re-walking the tree.
                        let known: std::collections::HashSet<&str> =
                            self.nodes.iter().map(|n| n.message.id.as_str()).collect();
                        let plan = plan_reconcile(&known, &messages);
                        // Sharing no prefix with a non-empty tree is almost
                        // always a lost or mis-built client view, not an
                        // intentional fork.
                        if plan.first().map(|w| w.index) == Some(0) && !self.nodes.is_empty() {
                            tracing::warn!(
                                "client view shares no prefix with the session; recording will fork at the root"
                            );
                        }

                        // Gate where the view lands: a view escaping the parked
                        // head dispatches; answers to pending work still queue.
                        let landing = landing_leaf(&messages, &plan);
                        if completions.is_empty()
                            && self.active_interrupt_for(landing.as_deref()).is_some()
                        {
                            return Err(SessionError::SessionInterrupted);
                        }

                        // Fast path iff the view's only change is one answer to one
                        // pending call — provable from the plan: exactly one write,
                        // and it is that answer.
                        let single_answer = plan.len() == 1
                            && completions.len() == 1
                            && plan.first().map(|w| w.index)
                                == completions.first().map(|c| c.index);

                        if single_answer {
                            // Mirror the settle endpoint: settle + tool.finished,
                            // the worker appends the node, the view is discarded.
                            if let Some(c) = completions.into_iter().next() {
                                events.push(EventPayload::ToolCallCompleted(ToolCallCompleted {
                                    tool_call_id: c.tool_call_id.clone(),
                                    name: c.name.clone(),
                                    result: c.result.clone(),
                                }));
                                let request = self.emit_tool_result(
                                    &events,
                                    c.tool_call_id,
                                    c.name,
                                    c.result,
                                    false,
                                );
                                events.extend(request);
                            }
                        } else {
                            // Bedrock: settle every answer silently (no
                            // tool.finished), then hand the worker the whole
                            // normalized view as one frozen transcript. The
                            // worker's echo records the client's messages. Covers
                            // plan-empty views (no-op resend / regenerate): still
                            // delivered, the worker decides.
                            for c in completions {
                                events.push(EventPayload::ToolCallCompleted(ToolCallCompleted {
                                    tool_call_id: c.tool_call_id,
                                    name: c.name,
                                    result: c.result,
                                }));
                            }
                            let request = self.emit_decision_request_at(
                                &events,
                                Trigger::ClientTranscript {
                                    messages,
                                    new_from: 0,
                                    client,
                                },
                                landing.as_deref(),
                            );
                            events.extend(request);
                        }
                    }
                    ClientPayload::Action(action) => {
                        if self.head_parked() {
                            return Err(SessionError::SessionInterrupted);
                        }
                        let request = self.emit_decision_request(
                            &events,
                            Trigger::ClientAction {
                                name: action.name,
                                args: action.args,
                            },
                        );
                        events.extend(request);
                    }
                }

                Ok(events)
            }

            CommandPayload::SendMessage {
                message,
                stream: _,
                turn_id,
                parent_id: _,
            } => {
                Self::ensure_internal(caller)?;
                if let Some(ref tid) = turn_id {
                    if self.completed_turn_ids.contains(tid) {
                        return Err(SessionError::TurnAlreadyCompleted {
                            turn_id: tid.clone(),
                        });
                    }
                    if self.turn_id.as_ref() == Some(tid) {
                        return Err(SessionError::TurnAlreadyActive {
                            turn_id: tid.clone(),
                        });
                    }
                }

                if self.head_parked() && message.role == Role::User {
                    return Err(SessionError::SessionInterrupted);
                }
                let mut events = Vec::new();
                if let Some(turn_id) = turn_id {
                    events.push(EventPayload::TurnStarted(TurnStarted { turn_id }));
                }
                if message.role == Role::User {
                    let request = self.emit_decision_request(
                        &events,
                        Trigger::ClientMessage {
                            messages: vec![message],
                            client: ClientContext::default(),
                        },
                    );
                    events.extend(request);
                }
                Ok(events)
            }

            CommandPayload::RequestLlmCall {
                call_id,
                request,
                stream,
                retry,
                handler,
                format,
            } => {
                Self::ensure_internal(caller)?;

                // Idempotent by id: (re-)issue only for a new/Failed/RetryScheduled call.
                let issue = matches!(
                    self.llm_calls.get(&call_id).map(|c| &c.tracking.status),
                    None | Some(&EffectStatus::Failed) | Some(&EffectStatus::RetryScheduled)
                );

                if issue {
                    // Mint ids now so the stored prompt (and its retries) is
                    // deterministic across replay; keep the wire form.
                    let request = LlmRequest {
                        messages: request
                            .messages
                            .into_iter()
                            .map(|m| DraftMessage::from(m.record()))
                            .collect(),
                        ..request
                    };
                    let request = self.with_connector_tools(request);
                    let mut events = vec![EventPayload::LlmCallRequested(LlmCallRequested {
                        call_id: call_id.clone(),
                        attempt: 0,
                        request: request.clone(),
                        stream,
                        retry: retry.clone(),
                        handler,
                        format,
                    })];

                    if handler == LlmHandler::Worker {
                        let execute = self.emit_decision_request(
                            &events,
                            Trigger::LlmExecute {
                                id: call_id,
                                request,
                                format,
                                stream,
                                attempt: 0,
                                deadline: retry.deadline(chrono::Utc::now()),
                            },
                        );
                        events.extend(execute);
                    }

                    Ok(events)
                } else {
                    tracing::debug!(
                        %call_id,
                        "llm call id already issued; request no-ops (idempotent)"
                    );
                    Ok(vec![])
                }
            }

            CommandPayload::CompleteLlmCall {
                call_id,
                attempt,
                response,
            } => {
                let is_system = matches!(caller, Caller::System { .. });
                let call = self.llm_calls.get(&call_id);
                self.check_llm_call_caller(call, caller)?;

                match call {
                    Some(c)
                        if c.tracking.status == EffectStatus::Pending
                            && attempt.is_none_or(|a| a == c.tracking.retry.attempts) =>
                    {
                        let truncated = response.finish_reason.as_deref() == Some("length");
                        let usage = response.usage.clone();
                        let cost = response.cost;
                        let tool_calls = if response.tool_calls.is_empty() {
                            None
                        } else {
                            Some(response.tool_calls.clone())
                        };
                        let content = if response.images.is_empty() {
                            response.content.clone().map(Content::Text)
                        } else {
                            let mut parts: Vec<ContentPart> = Vec::new();
                            if let Some(text) = &response.content {
                                parts.push(ContentPart::Text { text: text.clone() });
                            }
                            for img in &response.images {
                                parts.push(ContentPart::ImageUrl {
                                    image_url: ImageUrl {
                                        url: img.url.clone(),
                                    },
                                });
                            }
                            Some(Content::Parts(parts))
                        };
                        // Record the assistant under the call id. It's globally
                        // unique (so reconcile still records a new node) and it's
                        // the id the client was already streamed — AG-UI keys the
                        // assistant message on the call id — so a client's
                        // full-view echo matches this node instead of forking.
                        let message = DraftMessage {
                            id: Some(call_id.clone()),
                            role: Role::Assistant,
                            content,
                            tool_calls,
                            tool_call_id: None,
                            name: None,
                        };
                        let mut events = vec![EventPayload::LlmCallCompleted(LlmCallCompleted {
                            call_id: call_id.clone(),
                            attempt: c.tracking.retry.attempts,
                            response,
                        })];
                        let settle = self.emit_decision_request(
                            &events,
                            Trigger::llm_ok(call_id, message, truncated, usage, cost),
                        );
                        events.extend(settle);
                        Ok(events)
                    }
                    // Late/stale settle: silent for the executor (idempotent), an error for an out-of-band caller.
                    _ if is_system => Ok(vec![]),
                    None => Err(SessionError::EffectNotFound),
                    Some(c) if c.tracking.status != EffectStatus::Pending => {
                        Err(SessionError::EffectNotPending)
                    }
                    Some(_) => Err(SessionError::EffectAttemptMismatch),
                }
            }

            CommandPayload::FailLlmCall {
                call_id,
                attempt,
                error,
                retryable,
                code,
                detail,
            } => {
                let is_system = matches!(caller, Caller::System { .. });
                let call = self.llm_calls.get(&call_id);
                self.check_llm_call_caller(call, caller)?;
                let call = match call {
                    Some(c)
                        if c.tracking.status == EffectStatus::Pending
                            && attempt.is_none_or(|a| a == c.tracking.retry.attempts) =>
                    {
                        c
                    }
                    _ if is_system => return Ok(vec![]),
                    None => return Err(SessionError::EffectNotFound),
                    Some(c) if c.tracking.status != EffectStatus::Pending => {
                        return Err(SessionError::EffectNotPending)
                    }
                    Some(_) => return Err(SessionError::EffectAttemptMismatch),
                };
                let mut events = vec![EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: call_id.clone(),
                    attempt: call.tracking.retry.attempts,
                    error: error.clone(),
                    retryable,
                    code: code.clone(),
                    detail: detail.clone(),
                })];
                if call
                    .tracking
                    .retry_policy
                    .exhausted(&call.tracking.retry, retryable)
                {
                    let settle = self.emit_decision_request(
                        &events,
                        Trigger::llm_err(call_id, error, code, detail),
                    );
                    events.extend(settle);
                }
                Ok(events)
            }

            // Where a tool call runs follows from its name against the config in
            // force, resolved once here and frozen onto the call — a later config
            // change must not reroute a call already in flight.
            CommandPayload::RequestToolCall {
                tool_call_id,
                name,
                arguments,
                retry,
            } => {
                Self::ensure_internal(caller)?;
                let handler = self.tool_handler_for(&name);
                // Resolved once, here, and frozen onto the call with its
                // handler: a later config or fetch must not reroute a call
                // already in flight.
                let target = self.connector_tool_for(&name).map(|t| ConnectorTarget {
                    connector: t.connector,
                    remote_name: t.remote_name,
                });
                match self.tool_calls.get(&tool_call_id) {
                    Some(_) => Ok(vec![]),
                    None => {
                        let mut events = vec![EventPayload::ToolCallRequested(ToolCallRequested {
                            tool_call_id: tool_call_id.clone(),
                            attempt: 0,
                            name: name.clone(),
                            arguments: arguments.clone(),
                            handler,
                            target,
                            retry: retry.clone(),
                        })];
                        if handler == ToolHandler::Worker {
                            let execute = self.emit_decision_request(
                                &events,
                                Trigger::ToolExecute {
                                    id: tool_call_id,
                                    name,
                                    arguments,
                                    attempt: 0,
                                    deadline: retry.deadline(chrono::Utc::now()),
                                },
                            );
                            events.extend(execute);
                        }
                        Ok(events)
                    }
                }
            }

            CommandPayload::CompleteToolCall {
                tool_call_id,
                attempt,
                result,
            } => {
                let tc = self
                    .tool_calls
                    .get(&tool_call_id)
                    .ok_or(SessionError::EffectNotFound)?;

                if tc.tracking.status != EffectStatus::Pending {
                    return Err(SessionError::EffectNotPending);
                }

                if attempt.is_some_and(|a| a != tc.tracking.retry.attempts) {
                    return Err(SessionError::EffectAttemptMismatch);
                }

                self.check_tool_call_caller(tc, caller)?;

                let name = tc.name.clone();

                // Enforce the tool's own declared output contract: a violating
                // result settles as a terminal tool error instead.
                if let Some(schema) = self.declared_output_schema(&tool_call_id, &name) {
                    if let Some(violation) = output_violation(&schema, &result) {
                        return self.handle(
                            CommandPayload::FailToolCall {
                                tool_call_id,
                                attempt,
                                error: format!(
                                    "tool output violated its declared schema: {violation}"
                                ),
                                retryable: false,
                            },
                            caller,
                        );
                    }
                }

                let mut events = vec![EventPayload::ToolCallCompleted(ToolCallCompleted {
                    tool_call_id: tool_call_id.clone(),
                    name: name.clone(),
                    result: result.clone(),
                })];
                let settle = self.emit_tool_result(&events, tool_call_id, name, result, false);
                events.extend(settle);
                Ok(events)
            }

            CommandPayload::FailToolCall {
                tool_call_id,
                attempt,
                error,
                retryable,
            } => {
                let tc = self
                    .tool_calls
                    .get(&tool_call_id)
                    .ok_or(SessionError::EffectNotFound)?;
                if tc.tracking.status != EffectStatus::Pending {
                    return Err(SessionError::EffectNotPending);
                }
                if attempt.is_some_and(|a| a != tc.tracking.retry.attempts) {
                    return Err(SessionError::EffectAttemptMismatch);
                }
                self.check_tool_call_caller(tc, caller)?;
                let name = tc.name.clone();
                let exhausted = tc
                    .tracking
                    .retry_policy
                    .exhausted(&tc.tracking.retry, retryable);
                let mut events = vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id: tool_call_id.clone(),
                    name: name.clone(),
                    error: error.clone(),
                    retryable,
                })];
                if exhausted {
                    let settle = self.emit_tool_result(&events, tool_call_id, name, error, true);
                    events.extend(settle);
                }
                Ok(events)
            }

            CommandPayload::CompleteConnectorSync {
                connection_id,
                attempt,
                prefix,
                tools,
            } => {
                Self::ensure_internal(caller)?;
                self.check_connector_sync(&connection_id, attempt)?;
                self.report_filter(&connection_id, &tools, prefix.as_deref());
                let mut events = vec![EventPayload::ConnectorSyncCompleted(Box::new(
                    ConnectorSyncCompleted {
                        connection_id,
                        prefix,
                        tools,
                    },
                ))];
                events.extend(self.release_decisions(&events));
                Ok(events)
            }

            CommandPayload::FailConnectorSync {
                connection_id,
                attempt,
                error,
                retryable,
                needs_reauth,
            } => {
                Self::ensure_internal(caller)?;
                let sync = self.check_connector_sync(&connection_id, attempt)?;
                let terminal = sync.tracking.is_terminal_failure(retryable);
                // At ERROR because the turn goes ahead without these tools, and
                // a model answering confidently with a connector missing looks
                // exactly like one answering with it. `subs run` shows ERROR by
                // default, so this is the only notice a human gets.
                if terminal {
                    tracing::error!(
                        connection = %connection_id,
                        error = %error,
                        needs_reauth,
                        "connector unreachable; its tools are not offered to the model"
                    );
                } else {
                    tracing::warn!(
                        connection = %connection_id,
                        error = %error,
                        attempt = sync.tracking.retry.attempts,
                        "connector fetch failed; retrying"
                    );
                }
                let mut events = vec![EventPayload::ConnectorSyncErrored(ConnectorSyncErrored {
                    connection_id,
                    error,
                    retryable,
                    needs_reauth,
                })];
                // A terminal failure is settled, so it stops parking anything.
                // The engine unblocks and the worker decides whether a connector
                // it cannot reach is fatal; parking forever is the one outcome
                // that leaves nobody able to act.
                if terminal {
                    events.extend(self.release_decisions(&events));
                }
                Ok(events)
            }

            CommandPayload::RequestSubAgent {
                session_id,
                agent_id,
                tool_call_id,
                retry,
            } => {
                Self::ensure_internal(caller)?;
                match self.sub_agent_calls.get(&session_id) {
                    Some(_) => Ok(vec![]),
                    None => Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
                        session_id,
                        agent_id,
                        tool_call_id,
                        retry,
                    })]),
                }
            }

            CommandPayload::StartSubAgent { session_id } => {
                Self::ensure_internal(caller)?;
                match self
                    .sub_agent_calls
                    .get(&session_id)
                    .map(|c| &c.tracking.status)
                {
                    Some(&EffectStatus::Pending) => {
                        Ok(vec![EventPayload::SubAgentStarted(SubAgentStarted {
                            session_id,
                        })])
                    }
                    _ => Ok(vec![]),
                }
            }

            CommandPayload::FailSubAgent {
                session_id,
                error,
                retryable,
            } => {
                Self::ensure_internal(caller)?;
                let Some(sa) = self.sub_agent_calls.get(&session_id) else {
                    return Ok(vec![]);
                };
                if sa.tracking.status != EffectStatus::Pending {
                    return Ok(vec![]);
                }
                let tool_call_id = sa.tool_call_id.clone();
                let agent_id = sa.agent_id.clone();
                let exhausted = sa
                    .tracking
                    .retry_policy
                    .exhausted(&sa.tracking.retry, retryable);
                let mut events = vec![EventPayload::SubAgentErrored(SubAgentErrored {
                    session_id: session_id.clone(),
                    error: error.clone(),
                    retryable,
                })];
                if exhausted {
                    let settle = self.emit_sub_agent_result(
                        &events,
                        session_id,
                        tool_call_id,
                        agent_id,
                        error,
                        true,
                    );
                    events.extend(settle);
                }
                Ok(events)
            }

            CommandPayload::CompleteSubAgentTurn {
                session_id,
                data,
                cost,
                token_usage,
                ..
            } => {
                Self::ensure_internal(caller)?;
                let Some(sa) = self.sub_agent_calls.get(&session_id) else {
                    return Ok(vec![]);
                };
                let tool_call_id = sa.tool_call_id.clone();
                let agent_id = sa.agent_id.clone();
                let result = json_to_string(&data);
                let mut events = vec![EventPayload::SubAgentTurnCompleted(SubAgentTurnCompleted {
                    session_id: session_id.clone(),
                    cost,
                    token_usage,
                    data,
                })];
                let settle = self.emit_sub_agent_result(
                    &events,
                    session_id,
                    tool_call_id,
                    agent_id,
                    result,
                    false,
                );
                events.extend(settle);
                Ok(events)
            }

            CommandPayload::Interrupt {
                interrupt_id,
                reason,
                payload,
            } => {
                self.ensure_owns_session(caller)?;
                if self.head_parked() {
                    return Ok(vec![]);
                }
                let mut events = vec![EventPayload::SessionInterrupted(SessionInterrupted {
                    interrupt_id,
                    origin: Self::caller_interrupt_origin(caller),
                    reason,
                    payload,
                    anchor: self.head_id.clone(),
                })];
                events.extend(self.void_llm_calls_for_interrupt(&[], self.head_id.as_deref()));
                Ok(events)
            }

            CommandPayload::ResumeInterrupt {
                interrupt_id,
                payload,
            } => {
                self.ensure_owns_session(caller)?;
                let Some(open) = self.open_interrupt(&interrupt_id) else {
                    return Ok(vec![]);
                };
                if Self::caller_interrupt_origin(caller).privilege() < open.origin.privilege() {
                    return Err(SessionError::SessionAccessDenied);
                }
                let parked_head =
                    self.anchor_on_path(self.head_id.as_deref(), open.anchor.as_deref());
                let mut events = vec![EventPayload::InterruptResumed(InterruptResumed {
                    interrupt_id: interrupt_id.clone(),
                    payload: payload.clone(),
                })];
                // Trigger only when the interrupt parked the head path. Promoted
                // directly: emit_decision_request would park on the interrupt
                // this batch just resumed.
                if parked_head {
                    let trigger = Trigger::InterruptResumed {
                        interrupt_id,
                        payload,
                    };
                    let decision_id = new_call_id();
                    events.push(EventPayload::DecisionRequestQueued(DecisionRequestQueued {
                        decision_id: decision_id.clone(),
                        trigger,
                    }));
                    if !self.has_pending_worker_decision()
                        && !self.has_unsettled_session_start()
                        && !self.connectors_unready(&events, self.head_id.as_deref())
                    {
                        events.push(EventPayload::WorkerDecisionRequested(
                            WorkerDecisionRequested { decision_id },
                        ));
                    }
                }
                Ok(events)
            }

            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript,
                actions,
                state,
                agent,
            } => {
                Self::ensure_machine_or_system(caller)?;
                match self
                    .worker_decisions
                    .get(&decision_id)
                    .map(|d| &d.tracking.status)
                {
                    Some(&EffectStatus::Pending) => {}
                    _ => return Ok(vec![]),
                }
                let mut events: Vec<EventPayload> = vec![EventPayload::WorkerDecisionCompleted(
                    WorkerDecisionCompleted { decision_id },
                )];
                // The head once this batch applies — `self.head_id` is stale
                // until then, so in-batch anchors must use this instead.
                let (reconcile, head_after) = self.reconcile_transcript(transcript);

                // New nodes carry no anchor; walk to the deepest pre-existing ancestor.
                let batch_parents: std::collections::HashMap<&str, Option<&str>> = reconcile
                    .iter()
                    .filter_map(|e| match e {
                        EventPayload::NewMessage(m) => {
                            Some((m.message.id.as_str(), m.parent_id.as_deref()))
                        }
                        _ => None,
                    })
                    .collect();
                let mut prefix_cursor = head_after.as_deref();
                let mut seen = std::collections::HashSet::new();
                while let Some(id) = prefix_cursor {
                    if !seen.insert(id) {
                        break; // malformed parent cycle guard, mirrors path_ids
                    }
                    match batch_parents.get(id) {
                        Some(parent) => prefix_cursor = *parent,
                        None => break,
                    }
                }
                let prefix_leaf = prefix_cursor.map(str::to_string);

                events.extend(reconcile);

                let reap = self.void_stranded_work(prefix_leaf.as_deref());
                events.extend(reap);

                if let Some(state) = state {
                    if state != self.resolve_state_for(prefix_leaf.as_deref()) {
                        events.push(EventPayload::WorkerStateUpdated(WorkerStateUpdated {
                            state,
                            anchor: head_after.clone(),
                        }));
                    }
                }

                if let Some(config) = agent {
                    if Some(&config) != self.resolve_agent_for(prefix_leaf.as_deref()).as_ref() {
                        // A config write is what triggers a fetch; the promotion
                        // pass below waits on it.
                        events.extend(self.sync_connectors(&config));
                        events.push(EventPayload::AgentConfigUpdated(AgentConfigUpdated {
                            config,
                            anchor: head_after.clone(),
                        }));
                    }
                }

                let system = Caller::System {
                    tenant_id: self
                        .owner
                        .as_ref()
                        .map(|o| o.tenant_id.clone())
                        .unwrap_or_default(),
                };
                for action in actions {
                    let sub_events = match action {
                        // Settle for work this batch voided: drop it.
                        Action::ToolResult { ref id, .. } | Action::ToolError { ref id, .. }
                            if Self::batch_voided(&events, EffectKind::ToolCall, id) =>
                        {
                            Ok(vec![])
                        }
                        Action::LlmResult { ref id, .. } | Action::LlmError { ref id, .. }
                            if Self::batch_voided(&events, EffectKind::LlmCall, id) =>
                        {
                            Ok(vec![])
                        }
                        Action::CallLlm {
                            id,
                            request,
                            stream,
                            retry,
                            handler,
                            format,
                        } => self.handle(
                            CommandPayload::RequestLlmCall {
                                call_id: id,
                                request,
                                stream,
                                retry,
                                handler,
                                format,
                            },
                            &system,
                        ),
                        Action::CallTool {
                            id,
                            name,
                            arguments,
                            retry,
                        } => self.handle(
                            CommandPayload::RequestToolCall {
                                tool_call_id: id,
                                name,
                                arguments,
                                retry,
                            },
                            &system,
                        ),
                        Action::SpawnSubAgent {
                            session_id,
                            agent_id,
                            tool_call_id,
                            retry,
                        } => self.handle(
                            CommandPayload::RequestSubAgent {
                                session_id,
                                agent_id,
                                tool_call_id,
                                retry,
                            },
                            &system,
                        ),
                        Action::SendMessage {
                            session_id,
                            message,
                        } => Ok(vec![EventPayload::SessionMessageRequested(
                            SessionMessageRequested {
                                target_session_id: session_id,
                                message,
                            },
                        )]),
                        // Idempotence checks prefix_leaf: head_after may be
                        // in-batch, unwalkable.
                        Action::Interrupt {
                            interrupt_id,
                            reason,
                            payload,
                        } => {
                            if self.active_interrupt_for(prefix_leaf.as_deref()).is_some() {
                                Ok(vec![])
                            } else {
                                let mut sub =
                                    vec![EventPayload::SessionInterrupted(SessionInterrupted {
                                        interrupt_id,
                                        origin: InterruptOrigin::Frontend,
                                        reason,
                                        payload,
                                        anchor: head_after.clone(),
                                    })];
                                sub.extend(
                                    self.void_llm_calls_for_interrupt(
                                        &events,
                                        prefix_leaf.as_deref(),
                                    ),
                                );
                                Ok(sub)
                            }
                        }
                        Action::ToolResult {
                            id,
                            attempt,
                            result,
                        } => self.handle(
                            CommandPayload::CompleteToolCall {
                                tool_call_id: id,
                                attempt,
                                result,
                            },
                            &system,
                        ),
                        Action::LlmResult {
                            id,
                            attempt,
                            response,
                        } => self.handle(
                            CommandPayload::CompleteLlmCall {
                                call_id: id,
                                attempt,
                                response,
                            },
                            &system,
                        ),
                        Action::ToolError {
                            id,
                            attempt,
                            error,
                            retryable,
                            ..
                        } => self.handle(
                            CommandPayload::FailToolCall {
                                tool_call_id: id,
                                attempt,
                                error,
                                retryable,
                            },
                            &system,
                        ),
                        Action::LlmError {
                            id,
                            attempt,
                            error,
                            retryable,
                            code,
                            detail,
                        } => self.handle(
                            CommandPayload::FailLlmCall {
                                call_id: id,
                                attempt,
                                error,
                                retryable,
                                code,
                                detail,
                            },
                            &system,
                        ),
                        // A `done` while finalizing completes the turn; otherwise it
                        // ends the agent's turn and starts finalization.
                        Action::Done { data } => {
                            let cmd = if self.finalizing.is_some() {
                                CommandPayload::CompleteTurn
                            } else {
                                CommandPayload::FinishTurn { data }
                            };
                            self.handle(cmd, &system)
                        }
                    };
                    if let Ok(sub) = sub_events {
                        events.extend(sub);
                    }
                }

                // Promote the next unparked queued decision, unless this batch
                // interrupted (its anchor is in-batch, invisible here) or a
                // connector this batch declared is still being fetched.
                let interrupted_in_batch = events
                    .iter()
                    .any(|e| matches!(e, EventPayload::SessionInterrupted(_)));
                if !interrupted_in_batch
                    && !self.connectors_unready(&events, prefix_leaf.as_deref())
                {
                    let promoted = {
                        let dropped: std::collections::HashSet<&str> = events
                            .iter()
                            .filter_map(|e| match e {
                                EventPayload::DecisionRequestDropped(d) => {
                                    Some(d.decision_id.as_str())
                                }
                                _ => None,
                            })
                            .collect();
                        let mut candidates = self
                            .queued_decisions()
                            .into_iter()
                            .filter(|d| !dropped.contains(d.decision_id.as_str()))
                            .filter(|d| !self.decision_parked(&d.trigger))
                            .map(|d| d.decision_id.clone());
                        candidates.next().or_else(|| {
                            events.iter().find_map(|e| match e {
                                EventPayload::DecisionRequestQueued(q)
                                    if !self.decision_parked(&q.trigger) =>
                                {
                                    Some(q.decision_id.clone())
                                }
                                _ => None,
                            })
                        })
                    };
                    if let Some(decision_id) = promoted {
                        events.push(EventPayload::WorkerDecisionRequested(
                            WorkerDecisionRequested { decision_id },
                        ));
                    }
                }

                Ok(events)
            }

            CommandPayload::FailWorkerDecision {
                decision_id,
                error,
                retryable,
            } => {
                Self::ensure_machine_or_system(caller)?;
                let decision = match self.worker_decisions.get(&decision_id) {
                    Some(d) if d.tracking.status == EffectStatus::Pending => d,
                    _ => return Ok(vec![]),
                };
                let terminal = decision.tracking.is_terminal_failure(retryable);
                let mut events = vec![EventPayload::WorkerDecisionErrored(WorkerDecisionErrored {
                    decision_id,
                    error: error.clone(),
                    retryable,
                })];
                // Any terminally-failed decision ends the run — a turn whose driving
                // decision is dead can never settle on its own. A failed turn.finished
                // still completes its turn: the output is durable, the finalizer isn't.
                // Retryable failures wait for redelivery via Wake.
                if terminal {
                    events.extend(self.fail_run(error));
                }
                Ok(events)
            }

            CommandPayload::CancelSession => {
                Self::ensure_machine_or_system(caller)?;
                if matches!(self.status, SessionStatus::Done) {
                    return Ok(vec![]);
                }
                let mut events = vec![EventPayload::SessionCancelled];
                events.extend(self.void_effects(|tracking, _| {
                    matches!(
                        tracking.status,
                        EffectStatus::Pending | EffectStatus::RetryScheduled
                    )
                }));
                Ok(events)
            }

            CommandPayload::FinishTurn { data } => {
                Self::ensure_internal(caller)?;
                // The agent's turn is done. For an active, not-yet-finished turn,
                // begin finalization: queue turn.finished with the frozen output and
                // defer completion (the queued trigger sets `finalizing`, see the
                // DecisionRequestQueued apply). Otherwise settle immediately.
                if let Some(turn_id) = self.turn_id.clone() {
                    if !self.completed_turn_ids.contains(&turn_id) {
                        return Ok(self.emit_decision_request(
                            &[],
                            Trigger::turn_finished(
                                turn_id,
                                data,
                                self.turn_cost,
                                self.turn_token_usage.clone(),
                            ),
                        ));
                    }
                }
                Ok(vec![EventPayload::SessionDone(SessionDone {})])
            }

            // The finalizer settled: emit the run terminal from the frozen output.
            CommandPayload::CompleteTurn => {
                Self::ensure_internal(caller)?;
                Ok(self.finalize_run(None))
            }

            CommandPayload::Wake { now } => {
                Self::ensure_internal(caller)?;
                self.handle_wake(now)
            }

            CommandPayload::ReconcileDispatch => {
                Self::ensure_internal(caller)?;
                self.handle_reconcile_dispatch()
            }
        }
    }

    /// Fail every in-flight effect orphaned by a process death. Pull-dispatched
    /// decisions are not provably orphaned (the worker may still submit); failing
    /// them is safe — a late submit no-ops — at worst re-running the handler.
    /// Worker tool calls are never touched: their async settle may still arrive.
    fn handle_reconcile_dispatch(&self) -> Result<Vec<EventPayload>, SessionError> {
        const LOST: &str = "dispatch lost on engine restart";
        let mut events = vec![];
        for wd in self.worker_decisions.values() {
            if wd.tracking.status == EffectStatus::Pending {
                let terminal = wd.tracking.is_terminal_failure(true);
                events.push(EventPayload::WorkerDecisionErrored(WorkerDecisionErrored {
                    decision_id: wd.decision_id.clone(),
                    error: LOST.to_string(),
                    retryable: true,
                }));
                if terminal {
                    events.extend(self.fail_run(LOST.to_string()));
                    return Ok(events);
                }
            }
        }
        for call in self.llm_calls.values() {
            if call.handler == LlmHandler::Server && call.tracking.status == EffectStatus::Pending {
                events.push(EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: call.call_id.clone(),
                    attempt: call.tracking.retry.attempts,
                    error: LOST.to_string(),
                    retryable: true,
                    code: None,
                    detail: None,
                }));
                if call
                    .tracking
                    .retry_policy
                    .exhausted(&call.tracking.retry, true)
                {
                    let settle = self.emit_decision_request(
                        &events,
                        Trigger::llm_err(call.call_id.clone(), LOST.to_string(), None, None),
                    );
                    events.extend(settle);
                }
            }
        }
        Ok(events)
    }

    fn handle_wake(&self, now: DateTime<Utc>) -> Result<Vec<EventPayload>, SessionError> {
        // Timed-out pending LLM calls → fail
        for call in self.llm_calls.values() {
            if call.tracking.status == EffectStatus::Pending
                && call.tracking.deadline.is_some_and(|d| d <= now)
            {
                return Ok(vec![EventPayload::LlmCallErrored(LlmCallErrored {
                    call_id: call.call_id.clone(),
                    attempt: call.tracking.retry.attempts,
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                    code: Some(ErrorCode::DeadlineExceeded),
                    detail: None,
                })]);
            }
        }

        // Timed-out pending tool calls → fail
        for tc in self.tool_calls.values() {
            if tc.tracking.status == EffectStatus::Pending
                && tc.tracking.deadline.is_some_and(|d| d <= now)
            {
                let mut events = vec![EventPayload::ToolCallErrored(ToolCallErrored {
                    tool_call_id: tc.tool_call_id.clone(),
                    name: tc.name.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                })];
                if tc.tracking.retry_policy.exhausted(&tc.tracking.retry, true) {
                    let settle = self.emit_tool_result(
                        &events,
                        tc.tool_call_id.clone(),
                        tc.name.clone(),
                        "deadline exceeded".to_string(),
                        true,
                    );
                    events.extend(settle);
                }
                return Ok(events);
            }
        }

        // RetryScheduled LLM calls ready to re-issue
        for call in self.llm_calls.values() {
            if call.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = call.tracking.retry.next_at {
                    if next_at <= now {
                        let request = call.spec.to_request(call.prompt.clone());
                        let mut events = vec![EventPayload::LlmCallRequested(LlmCallRequested {
                            call_id: call.call_id.clone(),
                            attempt: call.tracking.retry.attempts,
                            request: request.clone(),
                            stream: call.stream,
                            retry: call.tracking.retry_policy.clone(),
                            handler: call.handler,
                            format: call.format,
                        })];
                        if call.handler == LlmHandler::Worker {
                            let execute = self.emit_decision_request(
                                &events,
                                Trigger::LlmExecute {
                                    id: call.call_id.clone(),
                                    request,
                                    format: call.format,
                                    stream: call.stream,
                                    attempt: call.tracking.retry.attempts,
                                    deadline: call.tracking.retry_policy.deadline(now),
                                },
                            );
                            events.extend(execute);
                        }
                        return Ok(events);
                    }
                }
            }
        }

        // RetryScheduled tool calls ready to re-issue
        for tc in self.tool_calls.values() {
            if tc.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = tc.tracking.retry.next_at {
                    if next_at <= now {
                        let mut events = vec![EventPayload::ToolCallRequested(ToolCallRequested {
                            tool_call_id: tc.tool_call_id.clone(),
                            attempt: tc.tracking.retry.attempts,
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                            handler: tc.handler,
                            // A retry keeps the target it was routed to.
                            target: tc.target.clone(),
                            retry: tc.tracking.retry_policy.clone(),
                        })];
                        if tc.handler == ToolHandler::Worker {
                            let execute = self.emit_decision_request(
                                &events,
                                Trigger::ToolExecute {
                                    id: tc.tool_call_id.clone(),
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.clone(),
                                    attempt: tc.tracking.retry.attempts,
                                    deadline: tc.tracking.retry_policy.deadline(now),
                                },
                            );
                            events.extend(execute);
                        }
                        return Ok(events);
                    }
                }
            }
        }

        // Timed-out pending connector syncs → fail. Decisions park behind these,
        // so a hung connection that never times out parks the session forever.
        for sync in self.connector_syncs.values() {
            if sync.tracking.status == EffectStatus::Pending
                && sync.tracking.deadline.is_some_and(|d| d <= now)
            {
                let mut events = vec![EventPayload::ConnectorSyncErrored(ConnectorSyncErrored {
                    connection_id: sync.connection_id.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                    needs_reauth: false,
                })];
                if sync.tracking.is_terminal_failure(true) {
                    tracing::error!(
                        connection = %sync.connection_id,
                        "connector did not answer in time; its tools are not offered to the model"
                    );
                    events.extend(self.release_decisions(&events));
                }
                return Ok(events);
            }
        }

        // RetryScheduled connector syncs ready to re-fetch
        for sync in self.connector_syncs.values() {
            if sync.tracking.status == EffectStatus::RetryScheduled
                && sync.tracking.retry.next_at.is_some_and(|at| at <= now)
            {
                return Ok(vec![EventPayload::ConnectorSyncRequested(
                    ConnectorSyncRequested {
                        connection_id: sync.connection_id.clone(),
                        attempt: sync.tracking.retry.attempts,
                        retry: sync.tracking.retry_policy.clone(),
                    },
                )]);
            }
        }

        // Timed-out pending sub-agent calls → fail
        for sa in self.sub_agent_calls.values() {
            if sa.tracking.status == EffectStatus::Pending
                && sa.tracking.deadline.is_some_and(|d| d <= now)
            {
                let mut events = vec![EventPayload::SubAgentErrored(SubAgentErrored {
                    session_id: sa.session_id.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                })];
                if sa.tracking.retry_policy.exhausted(&sa.tracking.retry, true) {
                    let settle = self.emit_sub_agent_result(
                        &events,
                        sa.session_id.clone(),
                        sa.tool_call_id.clone(),
                        sa.agent_id.clone(),
                        "deadline exceeded".to_string(),
                        true,
                    );
                    events.extend(settle);
                }
                return Ok(events);
            }
        }

        // RetryScheduled sub-agent calls ready to re-issue
        for sa in self.sub_agent_calls.values() {
            if sa.tracking.status == EffectStatus::RetryScheduled {
                if let Some(next_at) = sa.tracking.retry.next_at {
                    if next_at <= now {
                        return Ok(vec![EventPayload::SubAgentRequested(SubAgentRequested {
                            session_id: sa.session_id.clone(),
                            agent_id: sa.agent_id.clone(),
                            tool_call_id: sa.tool_call_id.clone(),
                            retry: sa.tracking.retry_policy.clone(),
                        })]);
                    }
                }
            }
        }

        // Timed-out pending worker decisions → fail
        for wd in self.worker_decisions.values() {
            if wd.tracking.status == EffectStatus::Pending
                && wd.tracking.deadline.is_some_and(|d| d <= now)
            {
                let mut events = vec![EventPayload::WorkerDecisionErrored(WorkerDecisionErrored {
                    decision_id: wd.decision_id.clone(),
                    error: "deadline exceeded".to_string(),
                    retryable: true,
                })];
                // A terminal timeout ends the run, same as a terminal error.
                if wd.tracking.is_terminal_failure(true) {
                    events.extend(self.fail_run("deadline exceeded".to_string()));
                }
                return Ok(events);
            }
        }

        // Re-issue a due decision (a RetryScheduled retry, or a queued one) only
        // when nothing is live — a second live decision lets two transcript
        // writers run against the same head and fork the tree. This mirrors the
        // condition `wake_at` schedules on; state can change between scheduling
        // and firing, so re-check here. Lost decisions are otherwise recovered by
        // work-queue redelivery, not re-fired.
        if !self.has_pending_worker_decision() {
            for wd in self.worker_decisions.values() {
                if wd.tracking.status == EffectStatus::RetryScheduled
                    && !self.decision_parked(&wd.trigger)
                    && !self.connectors_unready(&[], self.head_id.as_deref())
                {
                    if let Some(next_at) = wd.tracking.retry.next_at {
                        if next_at <= now {
                            return Ok(vec![EventPayload::WorkerDecisionRequested(
                                WorkerDecisionRequested {
                                    decision_id: wd.decision_id.clone(),
                                },
                            )]);
                        }
                    }
                }
            }

            // An unsettled session.start owns the slot until it settles:
            // queued decisions wait for its retry even before it is due. An
            // unsettled connector fetch holds it the same way.
            if !self.has_unsettled_session_start()
                && !self.connectors_unready(&[], self.head_id.as_deref())
            {
                if let Some(wd) = self
                    .queued_decisions()
                    .into_iter()
                    .find(|d| !self.decision_parked(&d.trigger))
                {
                    return Ok(vec![EventPayload::WorkerDecisionRequested(
                        WorkerDecisionRequested {
                            decision_id: wd.decision_id.clone(),
                        },
                    )]);
                }
            }
        }

        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;

    use super::super::aggregate::{CommitContext, SessionAggregate};
    use super::super::state::{AgentVersion, Logged};
    use super::*;
    use crate::protocol::{AgentTool, Handler, LlmTool, McpServer, McpTools, Message};
    use crate::runtime::session::events::EventPayload;
    use crate::runtime::span::SpanContext;
    use crate::runtime::Caller;

    /// Run a command through the handler and commit the resulting events, like production `execute`.
    fn dispatch(
        agg: &mut SessionAggregate,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Vec<EventPayload> {
        let events = agg.state.handle(cmd, caller).expect("setup command failed");
        let ctx = CommitContext {
            span: SpanContext::root(),
            occurred_at: Utc::now(),
        };
        agg.commit(events.clone(), &ctx);
        events
    }

    /// Build an empty aggregate and run `CreateSession`, then drain the
    /// `session.start` decision with an empty (no-config) response so tests
    /// resume from a clean "no pending decision" state. Use
    /// [`create_session_with_config`] when a test needs an agent config set.
    fn create_session(session_id: &str, tenant_id: &str, user_id: &str) -> SessionAggregate {
        create_session_with_config(session_id, tenant_id, user_id, None)
    }

    fn create_session_with_config(
        session_id: &str,
        tenant_id: &str,
        user_id: &str,
        agent: Option<AgentConfig>,
    ) -> SessionAggregate {
        let mut agg = SessionAggregate::new(
            session_id.to_string(),
            tenant_id.to_string(),
            SessionState::new(session_id.to_string()),
        );
        let events = dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: tenant_id.to_string(),
                    id: Some(user_id.to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let start = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(w) => Some(w.decision_id.clone()),
                _ => None,
            })
            .expect("CreateSession opens a session.start decision");
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: start,
                transcript: vec![],
                actions: vec![],
                state: None,
                agent,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        agg
    }

    /// Complete the pending `session.start` decision with an empty response, for
    /// tests that build the aggregate directly (e.g. a custom `worker_retry`)
    /// instead of through [`create_session`].
    fn drain_session_start(agg: &mut SessionAggregate) {
        let start = agg
            .state
            .worker_decisions
            .values()
            .find(|d| matches!(d.trigger, Trigger::SessionStart))
            .map(|d| d.decision_id.clone())
            .expect("a pending session.start decision");
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: start,
                transcript: vec![],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine(),
        );
    }

    #[test]
    fn frontend_can_complete_own_client_handled_tool_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        declare_client_tool(&mut agg, "my_tool");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                result: "ok".to_string(),
            },
            &Caller::Frontend {
                tenant_id: "tenant-a".to_string(),
                user_id: "user-1".to_string(),
                attrs: HashMap::new(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallCompleted(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [ToolCallCompleted, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
        assert_eq!(tc.result.as_deref(), Some("ok"));
        assert!(!tc.is_error);
    }

    #[test]
    fn frontend_with_mismatched_user_id_is_denied() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let caller = Caller::Frontend {
            tenant_id: "tenant-a".to_string(),
            user_id: "other-user".to_string(),
            attrs: HashMap::new(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: Some(0),
                    result: "ok".to_string(),
                },
                &caller,
            )
            .expect_err("mismatched user_id should be rejected");

        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn frontend_cannot_complete_worker_handled_tool_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let caller = Caller::Frontend {
            tenant_id: "tenant-a".to_string(),
            user_id: "user-1".to_string(),
            attrs: HashMap::new(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: Some(0),
                    result: "ok".to_string(),
                },
                &caller,
            )
            .expect_err("frontend should not complete worker-handled tool calls");

        assert!(
            matches!(err, SessionError::EffectWrongHandler),
            "expected EffectWrongHandler; got {err:?}"
        );
    }

    /// Declare a `get_weather` tool with an output contract, run its llm.call
    /// to the point where the tool call is in flight, and settle it with
    /// `result`. Returns the settle's events.
    fn settle_with_output_contract(result: &str) -> (SessionAggregate, Vec<EventPayload>) {
        use crate::protocol::{LlmTool, ToolCall, ToolCallFunction};

        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "call-1".to_string(),
                request: LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: Some(vec![LlmTool {
                        name: "get_weather".to_string(),
                        description: "d".to_string(),
                        input: None,
                        output: Some(serde_json::json!({
                            "type": "object",
                            "properties": { "temp_c": { "type": "number" } },
                            "required": ["temp_c"],
                        })),
                    }]),
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &system(),
        );
        let tool_call = ToolCall {
            id: "tc-1".to_string(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: "get_weather".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let finished = dispatch(
            &mut agg,
            CommandPayload::CompleteLlmCall {
                call_id: "call-1".to_string(),
                attempt: Some(0),
                response: LlmResponse {
                    model: "test-model".to_string(),
                    content: None,
                    tool_calls: vec![tool_call.clone()],
                    finish_reason: None,
                    usage: None,
                    cost: None,
                    images: vec![],
                },
            },
            &system(),
        );
        let decision_id = finished
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("llm.finished opens a decision");
        // Echo the default: record the assistant under the call id, dispatch the call.
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![DraftMessage {
                    id: Some("call-1".to_string()),
                    role: Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![tool_call]),
                    tool_call_id: None,
                    name: None,
                }],
                actions: vec![Action::CallTool {
                    id: "tc-1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                    retry: RetryPolicy::no_retry(),
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );
        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                result: result.to_string(),
            },
            &machine(),
        );
        (agg, events)
    }

    #[test]
    fn a_result_violating_the_declared_output_schema_settles_as_an_error() {
        let (agg, events) = settle_with_output_contract(r#"{"temp_c": "warm"}"#);

        assert!(
            matches!(events.as_slice(), [EventPayload::ToolCallErrored(_), ..]),
            "the completion becomes a terminal failure; got {events:?}"
        );
        assert!(
            decision_with(&events, |t| matches!(
                t,
                Trigger::ToolFinished { id, ok: false, error: Some(e), .. }
                    if id == "tc-1" && e.contains("tool output violated its declared schema")
            ))
            .is_some(),
            "the violation reaches the model as the tool's error; got {events:?}"
        );
        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert!(tc.is_error);
    }

    #[test]
    fn a_result_satisfying_the_declared_output_schema_settles_normally() {
        let (agg, events) = settle_with_output_contract(r#"{"temp_c": 21}"#);

        assert!(
            matches!(events.as_slice(), [EventPayload::ToolCallCompleted(_), ..]),
            "a conforming result completes; got {events:?}"
        );
        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert!(!tc.is_error);
        assert_eq!(tc.result.as_deref(), Some(r#"{"temp_c": 21}"#));
    }

    #[test]
    fn request_tool_call_with_client_handler_does_not_queue_worker_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        declare_client_tool(&mut agg, "my_tool");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::ToolCallRequested(_)]),
            "client-handled tool call should emit ToolCallRequested only; got {events:?}"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Pending);
        assert_eq!(tc.handler, ToolHandler::Client);
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn request_tool_call_with_worker_handler_emits_decision_to_execute() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallRequested(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "worker-handled tool call should also queue a worker decision; got {events:?}"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.handler, ToolHandler::Worker);
    }

    #[test]
    fn machine_completes_worker_handled_tool_call_after_worker_releases_decision() {
        // Async-tool flow: worker acks the tool.execute with no actions, then the tool settles out-of-band.
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let request_events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let d1 = request_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("worker-handled tool call emits a tool.execute decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine,
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                result: "ok".to_string(),
            },
            &machine,
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallCompleted(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [ToolCallCompleted, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn machine_completes_worker_handled_tool_call_before_worker_releases_decision() {
        // Tool result arrives before the worker acks its tool.execute decision.
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                result: "ok".to_string(),
            },
            &Caller::Machine {
                tenant_id: "tenant-a".to_string(),
                key_id: "prod-key-1".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallCompleted(_),
                    EventPayload::DecisionRequestQueued(_),
                ]
            ),
            "expected [ToolCallCompleted, DecisionRequestQueued]; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn complete_tool_call_with_wrong_attempt_fails() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let caller = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: Some(7),
                    result: "ok".to_string(),
                },
                &caller,
            )
            .expect_err("wrong attempt should be rejected");

        assert!(
            matches!(err, SessionError::EffectAttemptMismatch),
            "expected EffectAttemptMismatch; got {err:?}"
        );
    }

    #[test]
    fn submit_client_payload_with_active_turn_id_is_rejected() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let payload = ClientPayload::Message(ClientMessage {
            message: DraftMessage {
                id: None,
                role: Role::User,
                content: Some(Content::Text("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            stream: false,
        });

        dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: payload.clone(),
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload,
                    turn_id: Some("turn-1".to_string()),
                },
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect_err("re-submitting an active turn_id should be rejected");

        match err {
            SessionError::TurnAlreadyActive { turn_id } => assert_eq!(turn_id, "turn-1"),
            other => panic!("expected TurnAlreadyActive; got {other:?}"),
        }
    }

    #[test]
    fn submit_worker_decision_dispatches_action_and_completes_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("hi".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let events = agg
            .state
            .handle(
                CommandPayload::SubmitWorkerDecision {
                    decision_id,
                    transcript: vec![],
                    actions: vec![Action::CallTool {
                        id: "tc-1".to_string(),
                        name: "my_tool".to_string(),
                        arguments: "{}".to_string(),
                        retry: RetryPolicy::no_retry(),
                    }],
                    state: None,
                    agent: None,
                },
                &machine,
            )
            .expect("submit worker decision should succeed");

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionCompleted(_))),
            "expected WorkerDecisionCompleted; got {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallRequested(_))),
            "CallTool action should expand into a ToolCallRequested event; got {events:?}"
        );
    }

    #[test]
    fn duplicate_submit_worker_decision_is_no_op() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("hi".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: decision_id.clone(),
                transcript: vec![],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine,
        );

        // Second submission with the same decision_id should be a no-op.
        let events = agg
            .state
            .handle(
                CommandPayload::SubmitWorkerDecision {
                    decision_id,
                    transcript: vec![],
                    actions: vec![Action::CallTool {
                        id: "tc-1".to_string(),
                        name: "my_tool".to_string(),
                        arguments: "{}".to_string(),
                        retry: RetryPolicy::no_retry(),
                    }],
                    state: None,
                    agent: None,
                },
                &machine,
            )
            .expect("duplicate submission should not error");

        assert!(
            events.is_empty(),
            "duplicate worker decision submission should emit no events; got {events:?}"
        );
    }

    #[test]
    fn user_message_rejected_while_session_interrupted() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let user_message = ClientPayload::Message(ClientMessage {
            message: DraftMessage {
                id: None,
                role: Role::User,
                content: Some(Content::Text("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            stream: false,
        });

        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload: user_message,
                    turn_id: Some("turn-1".to_string()),
                },
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect_err("user messages should be rejected while interrupted");

        assert!(
            matches!(err, SessionError::SessionInterrupted),
            "expected SessionInterrupted; got {err:?}"
        );
    }

    #[test]
    fn complete_unknown_tool_call_fails() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let caller = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-unknown".to_string(),
                    attempt: Some(0),
                    result: "ok".to_string(),
                },
                &caller,
            )
            .expect_err("unknown tool call should be rejected");

        assert!(
            matches!(err, SessionError::EffectNotFound),
            "expected EffectNotFound; got {err:?}"
        );
    }

    #[test]
    fn send_message_wakes_a_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::SendMessage {
                message: DraftMessage {
                    id: None,
                    role: Role::User,
                    content: Some(Content::Text("hi".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                stream: false,
                turn_id: None,
                parent_id: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_)
                ]
            ),
            "expected [WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn cancel_session_emits_cancelled() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::CancelSession,
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::SessionCancelled]),
            "expected [SessionCancelled]; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Done);
    }

    #[test]
    fn mark_done_emits_session_done() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::FinishTurn {
                data: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // No active turn, no ancestry → just SessionDone, status returns to Idle.
        assert!(
            matches!(events.as_slice(), [EventPayload::SessionDone(_)]),
            "expected [SessionDone]; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn wake_with_no_pending_effects_is_noop() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let events = agg
            .state
            .handle(
                CommandPayload::Wake { now: Utc::now() },
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect("wake should succeed");

        assert!(
            events.is_empty(),
            "wake on idle session should be a no-op; got {events:?}"
        );
    }

    #[test]
    fn reconcile_dispatch_with_nothing_pending_is_noop() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let events = agg
            .state
            .handle(
                CommandPayload::ReconcileDispatch,
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect("reconcile should succeed");

        assert!(
            events.is_empty(),
            "reconcile on idle session should be a no-op; got {events:?}"
        );
    }

    #[test]
    fn reconcile_dispatch_schedules_a_retry_for_a_pending_decision() {
        let mut agg = create_session_with_retry(RetryPolicy::worker_default());
        let setup = dispatch(
            &mut agg,
            CommandPayload::SendMessage {
                message: node_msg("", Role::User, "hi"),
                stream: false,
                turn_id: None,
                parent_id: None,
            },
            &system(),
        );
        let decision_id = setup
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("message requests a decision");

        let events = dispatch(&mut agg, CommandPayload::ReconcileDispatch, &system());

        assert!(
            matches!(events.as_slice(), [EventPayload::WorkerDecisionErrored(_)]),
            "expected [WorkerDecisionErrored]; got {events:?}"
        );
        let wd = agg.state.worker_decisions.get(&decision_id).expect("kept");
        assert_eq!(wd.tracking.status, EffectStatus::RetryScheduled);
        assert!(wd.tracking.retry.next_at.is_some(), "a retry is scheduled");
    }

    #[test]
    fn reconcile_dispatch_without_retries_fails_the_run() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &system(),
        );

        let events = dispatch(&mut agg, CommandPayload::ReconcileDispatch, &system());

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::WorkerDecisionErrored(_),
                    EventPayload::TurnCompleted(_),
                    EventPayload::SessionDone(_)
                ]
            ),
            "a no-retry policy makes reconcile terminal; got {events:?}"
        );
    }

    #[test]
    fn reconcile_dispatch_retries_a_pending_server_llm_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: request_with(vec![]),
                stream: false,
                retry: RetryPolicy::llm_default(),
                handler: LlmHandler::Server,
            },
            &system(),
        );

        let events = dispatch(&mut agg, CommandPayload::ReconcileDispatch, &system());

        assert!(
            matches!(events.as_slice(), [EventPayload::LlmCallErrored(_)]),
            "expected [LlmCallErrored]; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("kept");
        assert_eq!(call.tracking.status, EffectStatus::RetryScheduled);
        assert!(
            call.tracking.retry.next_at.is_some(),
            "a retry is scheduled"
        );
    }

    #[test]
    fn request_llm_call_emits_requested() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::LlmCallRequested(_)]),
            "expected [LlmCallRequested]; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.tracking.status, EffectStatus::Pending);
    }

    fn node_msg(id: &str, role: Role, content: &str) -> DraftMessage {
        DraftMessage {
            id: (!id.is_empty()).then(|| id.to_string()),
            role,
            content: Some(Content::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn request_with(messages: Vec<DraftMessage>) -> LlmRequest {
        LlmRequest {
            model: "test-model".to_string(),
            messages,
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        }
    }

    /// Drive a worker `Append` action onto the tree via a user-message decision.
    fn append_via_worker(agg: &mut SessionAggregate, transcript: Vec<DraftMessage>) {
        let setup = dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("seed", Role::User, "seed"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        let decision_id = setup
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message requests a worker decision");
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript,
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine(),
        );
    }

    #[test]
    fn request_llm_call_stores_prompt_without_minting_nodes() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: request_with(vec![
                    node_msg("sys", Role::System, "be helpful"),
                    node_msg("u1", Role::User, "hi"),
                ]),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // The call records the worker's prompt but mints no tree nodes.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::NewMessage(_))),
            "request mints no tree nodes; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(
            call.prompt
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["sys", "u1"]
        );
    }

    #[test]
    fn submit_messages_forwards_a_replace_submission() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Messages(ClientMessages {
                    messages: vec![
                        node_msg("c1", Role::User, "hi"),
                        node_msg("a1", Role::Assistant, "hello"),
                        node_msg("c2", Role::User, "more"),
                    ],
                    stream: false,
                    client: Default::default(),
                }),
                turn_id: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(!events
            .iter()
            .any(|e| matches!(e, EventPayload::NewMessage(_))));
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(q) => Some(&q.trigger),
                _ => None,
            })
            .expect("a decision request");
        match trigger {
            Trigger::ClientTranscript { messages, .. } => {
                assert_eq!(
                    messages.iter().map(|m| m.id.as_deref()).collect::<Vec<_>>(),
                    vec![Some("c1"), Some("a1"), Some("c2")]
                );
            }
            t => panic!("expected a UserTranscript trigger; got {t:?}"),
        }
    }

    #[test]
    fn submit_append_queues_the_batch_as_a_client_message_trigger() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Append(ClientAppend {
                    messages: vec![
                        node_msg("c1", Role::User, "hi"),
                        node_msg("c2", Role::User, "more"),
                    ],
                    stream: false,
                    client: Default::default(),
                }),
                turn_id: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // Nothing records at submit; the batch rides the trigger and
        // materializes against the path at delivery.
        assert!(!events
            .iter()
            .any(|e| matches!(e, EventPayload::NewMessage(_))));
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(q) => Some(&q.trigger),
                _ => None,
            })
            .expect("a decision request");
        match trigger {
            Trigger::ClientMessage { messages, .. } => {
                assert_eq!(
                    messages.iter().map(|m| m.id.as_deref()).collect::<Vec<_>>(),
                    vec![Some("c1"), Some("c2")]
                );
            }
            t => panic!("expected a ClientMessage trigger; got {t:?}"),
        }
    }

    fn tool_msg(tool_call_id: &str, content: &str) -> DraftMessage {
        DraftMessage {
            id: None,
            role: Role::Tool,
            content: Some(Content::Text(content.into())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }

    fn tool_node(id: &str, tool_call_id: &str, content: &str) -> DraftMessage {
        DraftMessage {
            id: Some(id.into()),
            ..tool_msg(tool_call_id, content)
        }
    }

    /// Record `transcript` into the tree via one worker decision, giving tests
    /// exact control over recorded ids (reconcile keeps explicit unknown ids).
    fn seed_tree(agg: &mut SessionAggregate, transcript: Vec<DraftMessage>) {
        let events = dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "seed"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        let decision_id = decision_with(&events, |_| true).expect("a decision");
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript,
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine(),
        );
    }

    /// The messages carried by the (fired or queued) `client.messages` decision.
    fn transcript_messages(events: &[EventPayload]) -> Option<Vec<DraftMessage>> {
        events.iter().find_map(|e| {
            let trigger = match e {
                EventPayload::DecisionRequestQueued(p) => &p.trigger,
                _ => return None,
            };
            match trigger {
                Trigger::ClientTranscript { messages, .. } => Some(messages.clone()),
                _ => None,
            }
        })
    }

    fn submit_messages(
        agg: &mut SessionAggregate,
        messages: Vec<DraftMessage>,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Messages(ClientMessages {
                    messages,
                    stream: false,
                    client: Default::default(),
                }),
                turn_id: None,
            },
            &system(),
        )
    }

    #[test]
    fn submit_messages_single_answer_takes_the_fast_path() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");

        let events = submit_messages(&mut agg, vec![tool_msg("tc-1", "the answer")]);

        // The view's only change is one answer to one pending call: settle plus
        // a tool.finished decision, mirroring the settle endpoint. No transcript.
        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))));
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);
        assert!(
            decision_with(&events, |t| matches!(t, Trigger::ClientTranscript { .. })).is_none(),
            "the fast path fires tool.finished, not a transcript; got {events:?}"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
        assert_eq!(tc.result.as_deref(), Some("the answer"));
    }

    #[test]
    fn submit_messages_settles_client_tools_across_submissions() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "a");
        request_client_tool(&mut agg, "b");

        // Each lone answer is its own fast path (tool.finished), settling one
        // call at a time.
        let first = submit_messages(&mut agg, vec![tool_msg("a", "RA")]);
        assert_eq!(fired_tool_result(&first), vec!["a".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("a").unwrap().result.as_deref(),
            Some("RA")
        );
        assert_eq!(
            agg.state.tool_calls.get("b").unwrap().tracking.status,
            EffectStatus::Pending
        );

        let second = submit_messages(&mut agg, vec![tool_msg("b", "RB")]);
        assert_eq!(fired_tool_result(&second), vec!["b".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("b").unwrap().result.as_deref(),
            Some("RB")
        );
    }

    #[test]
    fn submit_messages_settles_all_client_tools_with_one_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "a");
        request_client_tool(&mut agg, "b");

        let events = submit_messages(&mut agg, vec![tool_msg("a", "RA"), tool_msg("b", "RB")]);
        // Both settle up front, then one live transcript decision carries the proposal.
        let sequence: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                EventPayload::ToolCallCompleted(_) => Some("complete"),
                EventPayload::WorkerDecisionRequested(_) => Some("live"),
                EventPayload::DecisionRequestQueued(_) => Some("queued"),
                _ => None,
            })
            .collect();
        assert_eq!(sequence, vec!["complete", "complete", "queued", "live"]);
        assert_eq!(
            agg.state.tool_calls.get("a").unwrap().result.as_deref(),
            Some("RA")
        );
        assert_eq!(
            agg.state.tool_calls.get("b").unwrap().result.as_deref(),
            Some("RB")
        );
    }

    #[test]
    fn submit_messages_echoing_a_resolved_tool_result_settles_nothing() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");
        submit_messages(&mut agg, vec![tool_msg("tc-1", "done")]);

        // A re-sent, already-resolved tool result matches nothing pending and must
        // not fabricate a completion; it proceeds as a plain transcript submission.
        let echo = submit_messages(&mut agg, vec![tool_msg("tc-1", "done")]);
        assert!(
            !echo
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "nothing to complete; got {echo:?}"
        );
        assert!(
            decision_with(&echo, |t| matches!(t, Trigger::ClientTranscript { .. })).is_some(),
            "the submission still delivers as a transcript decision; got {echo:?}"
        );
    }

    #[test]
    fn submit_messages_with_tool_results_and_a_user_message_settles_and_delivers_once() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");

        let events = submit_messages(
            &mut agg,
            vec![tool_msg("tc-1", "R"), node_msg("", Role::User, "and also")],
        );

        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))));
        assert_eq!(fired_tool_result(&events), Vec::<String>::new());
        let transcript_decisions = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    EventPayload::DecisionRequestQueued(p)
                        if matches!(p.trigger, Trigger::ClientTranscript { .. })
                )
            })
            .count();
        assert_eq!(
            transcript_decisions, 1,
            "one decision carries the whole submission"
        );

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
        assert_eq!(tc.result.as_deref(), Some("R"));
    }

    #[test]
    fn transcript_with_completions_passes_the_interrupt_gate_and_queues() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &system(),
        );

        let events = submit_messages(&mut agg, vec![tool_msg("tc-1", "R")]);

        // A lone answer still takes the fast path while interrupted; the
        // tool.finished decision queues until resume rather than firing.
        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))));
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::DecisionRequestQueued(p)
                    if matches!(p.trigger, Trigger::ToolFinished { .. })
            )),
            "the decision queues until resume; got {events:?}"
        );
        assert!(!events
            .iter()
            .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))));
    }

    #[test]
    fn plain_transcript_rejected_while_interrupted() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &system(),
        );

        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload: ClientPayload::Messages(ClientMessages {
                        messages: vec![node_msg("", Role::User, "hello")],
                        stream: false,
                        client: Default::default(),
                    }),
                    turn_id: None,
                },
                &system(),
            )
            .expect_err("a transcript that settles nothing is rejected while interrupted");
        assert!(matches!(err, SessionError::SessionInterrupted));
    }

    #[test]
    fn normalize_folds_a_client_tool_echo_onto_its_recorded_node() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        // Recorded tree: u1, a1, and a tool-result node w1 answering tc-1.
        seed_tree(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "calling"),
                tool_node("w1", "tc-1", "result"),
            ],
        );

        // A local-memory client resends the whole view with ITS OWN id for the
        // already-recorded tool message, plus a new user turn.
        let events = submit_messages(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "calling"),
                tool_node("client-tm", "tc-1", "result"),
                node_msg("", Role::User, "next"),
            ],
        );

        let messages = transcript_messages(&events).expect("a transcript decision");
        // The echo adopts w1's id, so the whole prefix is known and only the new
        // user turn is news — no fork.
        assert_eq!(
            messages[2].id.as_deref(),
            Some("w1"),
            "tool echo adopts the recorded node id"
        );
        let known: std::collections::HashSet<&str> = agg
            .state
            .nodes
            .iter()
            .map(|n| n.message.id.as_str())
            .collect();
        let plan = plan_reconcile(&known, &messages);
        assert_eq!(
            plan.len(),
            1,
            "only the new user turn is news; got {plan:?}"
        );
        assert_eq!(plan.first().map(|w| w.index), Some(3));
    }

    #[test]
    fn tool_echo_frozen_before_recording_folds_at_the_write_seam() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        seed_tree(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "calling"),
                tool_node("w1", "tc-1", "result"),
            ],
        );

        // The client's view froze before w1 recorded (queued behind the
        // tool.finished decision), so its echo carries a client id. The worker
        // echoes that stale view; the write seam still folds it onto w1.
        let d = open_decision(&mut agg, "resubmit");
        let events = submit_state(
            &mut agg,
            d,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "calling"),
                tool_node("client-tm", "tc-1", "result"),
                node_msg("u2", Role::User, "next"),
            ],
            None,
        );

        let new_ids: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                EventPayload::NewMessage(m) => Some(m.message.id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(new_ids, ["u2"], "only the new turn is news; got {events:?}");
        let tree = agg.state.message_tree();
        let u2 = tree.nodes.iter().find(|n| n.message.id == "u2").unwrap();
        assert_eq!(
            u2.parent_id.as_deref(),
            Some("w1"),
            "no fork at the tool node"
        );
    }

    #[test]
    fn edit_with_tail_replay_keeps_the_tool_result() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        // Recorded: u1, a1, tool node w1 (tc-1), a2.
        seed_tree(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "calling"),
                tool_node("w1", "tc-1", "result"),
                node_msg("a2", Role::Assistant, "done"),
            ],
        );

        // Edit u1 (fresh id), then replay the tail including the client's own
        // copy of the tool result. The echo folds onto w1, so the re-recorded
        // branch keeps the tool result rather than dangling a tool_use.
        let events = submit_messages(
            &mut agg,
            vec![
                node_msg("u1b", Role::User, "hi (edited)"),
                node_msg("a1", Role::Assistant, "calling"),
                tool_node("client-tm", "tc-1", "result"),
                node_msg("a2", Role::Assistant, "done"),
            ],
        );

        let messages = transcript_messages(&events).expect("a transcript decision");
        assert_eq!(
            messages[2].id.as_deref(),
            Some("w1"),
            "the replayed tool result folds onto w1"
        );
        let known: std::collections::HashSet<&str> = agg
            .state
            .nodes
            .iter()
            .map(|n| n.message.id.as_str())
            .collect();
        let plan = plan_reconcile(&known, &messages);
        // News starts at the edit and doesn't stop: all four re-record onto the
        // new branch, tool result included.
        assert_eq!(
            plan.len(),
            4,
            "the whole edited branch is news; got {plan:?}"
        );
        assert!(
            plan.iter().any(|w| w.index == 2),
            "the tool result is part of the re-recorded branch"
        );
    }

    #[test]
    fn scrambled_answer_takes_the_bedrock_path() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");

        // A view whose recording changes more than the one answer (an extra new
        // assistant turn) can't take the fast path — it's recorded as-is.
        let events = submit_messages(
            &mut agg,
            vec![
                tool_msg("tc-1", "R"),
                node_msg("", Role::Assistant, "scrambled"),
            ],
        );

        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))));
        assert_eq!(fired_tool_result(&events), Vec::<String>::new());
        assert!(
            transcript_messages(&events).is_some(),
            "a scrambled view delivers as a transcript; got {events:?}"
        );
        assert_eq!(
            agg.state.tool_calls.get("tc-1").unwrap().tracking.status,
            EffectStatus::Completed
        );
    }

    #[test]
    fn duplicate_answers_in_one_view_settle_once() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");

        // Two tool messages answer the same pending call; only the first counts.
        let events = submit_messages(
            &mut agg,
            vec![tool_msg("tc-1", "first"), tool_msg("tc-1", "second")],
        );

        let completions = events
            .iter()
            .filter(|e| matches!(e, EventPayload::ToolCallCompleted(_)))
            .count();
        assert_eq!(
            completions, 1,
            "one settle for the one call; got {events:?}"
        );
        assert_eq!(
            agg.state.tool_calls.get("tc-1").unwrap().result.as_deref(),
            Some("first")
        );
    }

    #[test]
    fn recorded_echo_beside_a_new_answer_fast_paths_the_new_one() {
        // The across-runs case: run 1's answer to tc-1 has been recorded (worker
        // echoed w1) by the time run 2 arrives. Run 2 resends its stale copy of
        // tc-1 beside a genuinely new answer to tc-2.
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        seed_tree(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "calling"),
                tool_node("w1", "tc-1", "RA"),
            ],
        );
        request_client_tool(&mut agg, "tc-2");

        let events = submit_messages(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "calling"),
                tool_node("client-tm", "tc-1", "RA"),
                tool_msg("tc-2", "RB"),
            ],
        );

        // The stale echo folds onto w1, so only the new answer is news: fast path.
        assert_eq!(fired_tool_result(&events), vec!["tc-2".to_string()]);
        assert!(
            transcript_messages(&events).is_none(),
            "run 2 reduces to a single live answer; got {events:?}"
        );
        assert_eq!(
            agg.state.tool_calls.get("tc-2").unwrap().result.as_deref(),
            Some("RB")
        );
    }

    #[test]
    fn mixed_answer_and_message_while_interrupted_queues_bedrock() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &system(),
        );

        let events = submit_messages(
            &mut agg,
            vec![tool_msg("tc-1", "R"), node_msg("", Role::User, "and also")],
        );

        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))));
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::DecisionRequestQueued(p)
                    if matches!(p.trigger, Trigger::ClientTranscript { .. })
            )),
            "the bedrock transcript queues until resume; got {events:?}"
        );
        assert!(!events
            .iter()
            .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))));
    }

    #[test]
    fn queued_client_message_stays_a_delta_until_delivery() {
        // Materialization happens at delivery, so a queued message composes with
        // whatever the decision ahead of it writes.
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let first = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "A"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        assert!(first
            .iter()
            .any(|e| matches!(e, EventPayload::DecisionRequestQueued(p)
                if matches!(p.trigger, Trigger::ClientMessage { .. }))));

        let second = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "B"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        assert!(
            second
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(p)
                    if matches!(p.trigger, Trigger::ClientMessage { .. }))),
            "the stored trigger keeps the bare message; got {second:?}"
        );
    }

    #[test]
    fn reconcile_re_records_known_ids_past_the_first_new_node() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        let d1 = decision_with(&events, |_| true).expect("a decision");
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![
                    node_msg("u1", Role::User, "hi"),
                    node_msg("a1", Role::Assistant, "yo"),
                    node_msg("u2", Role::User, "more"),
                ],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine(),
        );
        assert_eq!(agg.state.head_id.as_deref(), Some("u2"));

        // Edit a1 while keeping u2's known id after the fork point: the known id
        // must be re-recorded as a fresh node, not grafted back onto the old branch.
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "again"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        let d2 = decision_with(&events, |_| true).expect("a decision");
        let submit = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d2,
                transcript: vec![
                    node_msg("u1", Role::User, "hi"),
                    node_msg("e1", Role::Assistant, "edited"),
                    node_msg("u2", Role::User, "more"),
                ],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine(),
        );

        let new_nodes: Vec<(&str, Option<&str>, &Message)> = submit
            .iter()
            .filter_map(|e| match e {
                EventPayload::NewMessage(m) => {
                    Some((m.message.id.as_str(), m.parent_id.as_deref(), &m.message))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            new_nodes.len(),
            2,
            "e1 and the u2 re-record; got {submit:?}"
        );
        assert_eq!((new_nodes[0].0, new_nodes[0].1), ("e1", Some("u1")));
        let (copy_id, copy_parent, copy) = new_nodes[1];
        assert_ne!(
            copy_id, "u2",
            "the known id past the fork gets a fresh node id"
        );
        assert_eq!(copy_parent, Some("e1"));
        assert_eq!(
            copy.content.as_ref().map(Content::text_owned).as_deref(),
            Some("more")
        );
        assert_eq!(agg.state.head_id.as_deref(), Some(copy_id));
    }

    #[test]
    fn complete_llm_call_emits_completed() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteLlmCall {
                call_id: "llm-1".to_string(),
                attempt: Some(0),
                response: LlmResponse {
                    model: "test-model".to_string(),
                    content: Some("hello".to_string()),
                    tool_calls: vec![],
                    finish_reason: Some("stop".to_string()),
                    usage: None,
                    cost: None,
                    images: vec![],
                },
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::LlmCallCompleted(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [LlmCallCompleted, WorkerDecisionRequested]; got {events:?}"
        );
        assert!(
            decision_with(&events, |t| matches!(
                t,
                Trigger::LlmFinished { id, ok: true, .. } if id == "llm-1"
            ))
            .is_some(),
            "completion fires an llm.finished trigger; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn llm_completion_records_the_assistant_under_the_call_id() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "call-1", LlmHandler::Server);
        let events = complete_llm(&mut agg, "call-1", 0, &system());

        // The assistant node's id is the call id, matching what AG-UI streamed,
        // so a client's echo of it reconciles instead of forking.
        let msg_id = events
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(p) => match &p.trigger {
                    Trigger::LlmFinished {
                        message: Some(m), ..
                    } => m.id.clone(),
                    _ => None,
                },
                _ => None,
            })
            .expect("an llm.finished trigger carrying the assistant");
        assert_eq!(msg_id, "call-1");
    }

    #[test]
    fn agui_resend_of_a_prior_assistant_turn_appends_without_forking() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        // Turn 1: a user message, then an LLM turn whose assistant is recorded
        // under the call id, then the worker echoes it into the tree.
        let d1 = decision_with(
            &submit_messages(&mut agg, vec![node_msg("u1", Role::User, "hi")]),
            |_| true,
        )
        .expect("a client decision");
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![node_msg("u1", Role::User, "hi")],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine(),
        );
        request_llm(&mut agg, "call-1", LlmHandler::Server);
        let done = complete_llm(&mut agg, "call-1", 0, &system());
        // The worker echoes the assistant message straight from the trigger; its
        // id is whatever the engine recorded it under (the fix: the call id).
        let (d2, assistant) = done
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(p) => match &p.trigger {
                    Trigger::LlmFinished {
                        message: Some(m), ..
                    } => Some((p.decision_id.clone(), m.clone())),
                    _ => None,
                },
                _ => None,
            })
            .expect("an llm.finished decision carrying the assistant");
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d2,
                transcript: vec![node_msg("u1", Role::User, "hi"), assistant],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine(),
        );
        // The client was streamed the assistant under the call id, so its
        // full-view echo uses "call-1". The recorded node must carry the same id.
        assert_eq!(agg.state.head_id.as_deref(), Some("call-1"));

        let events = submit_messages(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("call-1", Role::Assistant, "hello"),
                node_msg("", Role::User, "again"),
            ],
        );
        let messages = transcript_messages(&events).expect("a transcript decision");
        let known: std::collections::HashSet<&str> = agg
            .state
            .nodes
            .iter()
            .map(|n| n.message.id.as_str())
            .collect();
        let plan = plan_reconcile(&known, &messages);
        assert_eq!(
            plan.len(),
            1,
            "only the new user turn is news; got {plan:?}"
        );
        assert_eq!(plan.first().map(|w| w.index), Some(2));
    }

    #[test]
    fn fail_llm_call_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::FailLlmCall {
                call_id: "llm-1".to_string(),
                attempt: Some(0),
                error: "provider down".to_string(),
                retryable: false,
                code: None,
                detail: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // Retry exhausted, so the handler fires a follow-up worker decision with the error.
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::LlmCallErrored(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [LlmCallErrored, WorkerDecisionRequested]; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.tracking.status, EffectStatus::Failed);
    }

    #[test]
    fn llm_retry_reuses_the_stored_prompt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let retry = RetryPolicy {
            timeout_secs: None,
            max_retries: 3,
            backoff_base_secs: 1,
            backoff_max_secs: 1,
        };
        // The retry re-issues the stored prompt.
        append_via_worker(
            &mut agg,
            vec![
                node_msg("sys", Role::System, "sys prompt"),
                node_msg("u1", Role::User, "hi"),
            ],
        );
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: request_with(vec![
                    node_msg("sys", Role::System, "sys prompt"),
                    node_msg("u1", Role::User, "hi"),
                ]),
                stream: false,
                retry,
                handler: LlmHandler::Server,
            },
            &system(),
        );

        let call = agg.state.llm_calls.get("llm-1").expect("call present");
        assert_eq!(
            call.prompt
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["sys", "u1"]
        );

        dispatch(
            &mut agg,
            CommandPayload::FailLlmCall {
                call_id: "llm-1".to_string(),
                attempt: Some(0),
                error: "provider hiccup".to_string(),
                retryable: true,
                code: None,
                detail: None,
            },
            &system(),
        );
        assert_eq!(
            agg.state.llm_calls.get("llm-1").map(|c| &c.tracking.status),
            Some(&EffectStatus::RetryScheduled)
        );

        let later = Utc::now() + chrono::Duration::seconds(120);
        let events = dispatch(&mut agg, CommandPayload::Wake { now: later }, &system());
        let reissued = events
            .iter()
            .find_map(|e| match e {
                EventPayload::LlmCallRequested(p) => Some(p),
                _ => None,
            })
            .expect("retry re-issues the llm call");

        assert_eq!(reissued.request.model, "test-model");
        assert_eq!(reissued.request.messages.len(), 2);
        assert_eq!(reissued.request.messages[0].id.as_deref(), Some("sys"));
        assert_eq!(reissued.request.messages[1].id.as_deref(), Some("u1"));
        assert!(matches!(
            &reissued.request.messages[1].content,
            Some(Content::Text(t)) if t == "hi"
        ));
    }

    fn test_llm_request() -> LlmRequest {
        LlmRequest {
            model: "test-model".to_string(),
            messages: vec![],
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        }
    }

    #[test]
    fn worker_handled_llm_call_emits_request_trigger() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: test_llm_request(),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Worker,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::LlmCallRequested(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [LlmCallRequested, WorkerDecisionRequested]; got {events:?}"
        );
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(p) => Some(&p.trigger),
                _ => None,
            })
            .expect("worker decision present");
        assert!(
            matches!(
                trigger,
                Trigger::LlmExecute { id, .. } if id == "llm-1"
            ),
            "expected an llm.execute trigger for the llm call; got {trigger:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.handler, LlmHandler::Worker);
        assert_eq!(call.tracking.status, EffectStatus::Pending);
    }

    #[test]
    fn server_handled_llm_call_does_not_emit_request_trigger() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: test_llm_request(),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::LlmCallRequested(_)]),
            "expected just [LlmCallRequested]; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.handler, LlmHandler::Server);
    }

    #[test]
    fn return_llm_result_completes_worker_handled_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let request_events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: "llm-1".to_string(),
                request: test_llm_request(),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Worker,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = request_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("worker-handled llm call emits an llm.execute decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![Action::LlmResult {
                    id: "llm-1".to_string(),
                    attempt: Some(0),
                    response: LlmResponse {
                        model: "test-model".to_string(),
                        content: Some("hello from the worker".to_string()),
                        tool_calls: vec![],
                        finish_reason: Some("stop".to_string()),
                        usage: None,
                        cost: None,
                        images: vec![],
                    },
                }],
                state: None,
                agent: None,
            },
            &machine,
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::LlmCallCompleted(_))),
            "expected an LlmCallCompleted event; got {events:?}"
        );
        assert!(
            decision_with(&events, |t| matches!(
                t,
                Trigger::LlmFinished { id, ok: true, .. } if id == "llm-1"
            ))
            .is_some(),
            "completion fires an llm.finished trigger; got {events:?}"
        );
        let call = agg.state.llm_calls.get("llm-1").expect("llm call present");
        assert_eq!(call.tracking.status, EffectStatus::Completed);
    }

    #[test]
    fn fail_tool_call_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let request_events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let d1 = request_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("worker-handled tool call emits a tool.execute decision");

        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };

        // Worker releases its decision so the failure emits a fresh follow-up.
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine,
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::FailToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                error: "boom".to_string(),
                retryable: false,
            },
            &machine,
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::ToolCallErrored(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [ToolCallErrored, WorkerDecisionRequested]; got {events:?}"
        );
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(p) => Some(&p.trigger),
                _ => None,
            })
            .expect("a tool.finished decision");
        assert!(
            matches!(
                trigger,
                Trigger::ToolFinished { id, ok: false, .. } if id == "tc-1"
            ),
            "expected an errored tool.finished for tc-1; got {trigger:?}"
        );
        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Failed);
        assert!(tc.is_error);
        assert_eq!(tc.result.as_deref(), Some("boom"));
    }

    #[test]
    fn request_sub_agent_emits_requested() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                tool_call_id: "call-sa".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::SubAgentRequested(_)]),
            "expected [SubAgentRequested]; got {events:?}"
        );
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent recorded");
        assert_eq!(sa.tracking.status, EffectStatus::Pending);
    }

    #[test]
    fn start_sub_agent_emits_started() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                tool_call_id: "call-sa".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::StartSubAgent {
                session_id: "child-1".to_string(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(events.as_slice(), [EventPayload::SubAgentStarted(_)]),
            "expected [SubAgentStarted]; got {events:?}"
        );
    }

    #[test]
    fn fail_sub_agent_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                tool_call_id: "call-sa".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::FailSubAgent {
                session_id: "child-1".to_string(),
                error: "child crashed".to_string(),
                retryable: false,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::SubAgentErrored(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [SubAgentErrored, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["call-sa".to_string()]);
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent present");
        assert_eq!(sa.tracking.status, EffectStatus::Failed);
        assert_eq!(sa.result.as_deref(), Some("child crashed"));
        assert!(sa.is_error);
    }

    #[test]
    fn complete_sub_agent_turn_emits_completed() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                tool_call_id: "call-sa".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteSubAgentTurn {
                session_id: "child-1".to_string(),
                agent_id: "agent-2".to_string(),
                turn_id: "turn-x".to_string(),
                data: serde_json::json!("done"),
                cost: rust_decimal::Decimal::ZERO,
                token_usage: std::collections::BTreeMap::new(),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::SubAgentTurnCompleted(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [SubAgentTurnCompleted, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["call-sa".to_string()]);
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent present");
        assert_eq!(sa.result.as_deref(), Some("done"));
        assert!(!sa.is_error);
    }

    // ── Batched effect completion ────────────────────────────────────────

    fn machine() -> Caller {
        Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        }
    }

    fn system() -> Caller {
        Caller::System {
            tenant_id: "tenant-a".to_string(),
        }
    }

    /// Declare `name` as client-handled, unanchored so it covers every path.
    ///
    /// Where a call runs is derived from the config rather than passed in, so a
    /// test that exercises the client path has to say so. Pushed straight onto
    /// the log rather than through a decision, to leave event assertions alone.
    fn declare_client_tool(agg: &mut SessionAggregate, name: &str) {
        let mut agent = agent_config("m1");
        agent.tools.push(AgentTool {
            name: name.to_string(),
            description: String::new(),
            input: None,
            output: None,
            handler: Some(Handler::Client),
        });
        agg.state.agent_versions.push(Logged {
            seq: agg.state.agent_versions.last().map_or(0, |v| v.seq),
            entry: AgentVersion {
                agent,
                anchor: None,
            },
        });
    }

    fn request_client_tool(agg: &mut SessionAggregate, id: &str) {
        let name = format!("tool_{id}");
        declare_client_tool(agg, &name);
        dispatch(
            agg,
            CommandPayload::RequestToolCall {
                tool_call_id: id.to_string(),
                name,
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &system(),
        );
    }

    fn complete_tool(agg: &mut SessionAggregate, id: &str, result: &str) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: id.to_string(),
                attempt: Some(0),
                result: result.to_string(),
            },
            &machine(),
        )
    }

    fn wake(agg: &mut SessionAggregate) -> Vec<EventPayload> {
        dispatch(agg, CommandPayload::Wake { now: Utc::now() }, &system())
    }

    /// The `tool_call_id`s of every `tool.finished`/`sub_agent.finished` trigger, in order (deduped by decision id).
    fn fired_tool_result(events: &[EventPayload]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        events
            .iter()
            .filter_map(|e| {
                let (decision_id, trigger) = match e {
                    EventPayload::DecisionRequestQueued(p) => (&p.decision_id, &p.trigger),
                    _ => return None,
                };
                let tool_call_id = match trigger {
                    Trigger::ToolFinished { id, .. } | Trigger::SubAgentFinished { id, .. } => id,
                    _ => return None,
                };
                seen.insert(decision_id.clone())
                    .then(|| tool_call_id.clone())
            })
            .collect()
    }

    fn decision_with(events: &[EventPayload], pred: impl Fn(&Trigger) -> bool) -> Option<String> {
        events.iter().find_map(|e| {
            let (id, trigger) = match e {
                EventPayload::DecisionRequestQueued(p) => (&p.decision_id, &p.trigger),
                _ => return None,
            };
            pred(trigger).then(|| id.clone())
        })
    }

    #[test]
    fn each_completion_fires_a_tool_result() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "a");
        request_client_tool(&mut agg, "b");

        let first = complete_tool(&mut agg, "a", "RA");
        assert_eq!(fired_tool_result(&first), vec!["a".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("a").unwrap().result.as_deref(),
            Some("RA")
        );

        let second = complete_tool(&mut agg, "b", "RB");
        assert_eq!(fired_tool_result(&second), vec!["b".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("b").unwrap().result.as_deref(),
            Some("RB")
        );
    }

    #[test]
    fn worker_tool_fires_tool_result_in_the_completion_commit() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("go".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &system(),
        );
        let decision_id = decision_with(&setup, |t| matches!(t, Trigger::ClientMessage { .. }))
            .expect("user message decision");

        let dispatched = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![Action::CallTool {
                    id: "t1".to_string(),
                    name: "getWeather".to_string(),
                    arguments: "{}".to_string(),
                    retry: RetryPolicy::no_retry(),
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );
        let exec = decision_with(&dispatched, |t| matches!(t, Trigger::ToolExecute { .. }))
            .expect("tool.execute decision");

        let completed = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: exec,
                transcript: vec![],
                actions: vec![Action::ToolResult {
                    id: "t1".to_string(),
                    attempt: Some(0),
                    result: "RA".to_string(),
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );
        assert_eq!(
            fired_tool_result(&completed),
            vec!["t1".to_string()],
            "the finished trigger fires in the completion commit; got {completed:?}"
        );
        assert_eq!(
            agg.state.tool_calls.get("t1").unwrap().result.as_deref(),
            Some("RA")
        );
        assert!(
            fired_tool_result(&wake(&mut agg)).is_empty(),
            "no wake needed"
        );
    }

    #[test]
    fn batch_mixes_tool_and_sub_agent() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "t1");
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "researcher".to_string(),
                tool_call_id: "s1".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &system(),
        );
        dispatch(
            &mut agg,
            CommandPayload::StartSubAgent {
                session_id: "child-1".to_string(),
            },
            &system(),
        );

        let tool_done = complete_tool(&mut agg, "t1", "TOOL");
        assert_eq!(fired_tool_result(&tool_done), vec!["t1".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("t1").unwrap().result.as_deref(),
            Some("TOOL")
        );

        let sub_done = dispatch(
            &mut agg,
            CommandPayload::CompleteSubAgentTurn {
                session_id: "child-1".to_string(),
                agent_id: "researcher".to_string(),
                turn_id: "turn-1".to_string(),
                data: serde_json::json!("FINDINGS"),
                cost: rust_decimal::Decimal::ZERO,
                token_usage: std::collections::BTreeMap::new(),
            },
            &system(),
        );

        assert_eq!(fired_tool_result(&sub_done), vec!["s1".to_string()]);
        let sa = agg
            .state
            .sub_agent_calls
            .get("child-1")
            .expect("sub-agent present");
        assert_eq!(sa.tool_call_id, "s1");
        assert_eq!(sa.agent_id, "researcher");
        assert_eq!(sa.result.as_deref(), Some("FINDINGS"));
    }

    #[test]
    fn tool_and_sub_agent_from_one_turn_dispatch_concurrently() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("go".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &system(),
        );
        let decision_id = setup
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message requests a worker decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![
                    Action::CallTool {
                        id: "t1".to_string(),
                        name: "getWeather".to_string(),
                        arguments: "{}".to_string(),
                        retry: RetryPolicy::no_retry(),
                    },
                    Action::SpawnSubAgent {
                        session_id: "child-1".to_string(),
                        agent_id: "researcher".to_string(),
                        tool_call_id: "s1".to_string(),
                        retry: RetryPolicy::no_retry(),
                    },
                ],
                state: None,
                agent: None,
            },
            &machine(),
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallRequested(_))),
            "tool dispatched; got {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::SubAgentRequested(_))),
            "sub-agent dispatched; got {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::DecisionRequestQueued(p)
                    if matches!(p.trigger, Trigger::ToolExecute { .. })
            )),
            "tool's tool.execute decision dispatched; got {events:?}"
        );

        assert_eq!(
            agg.state.tool_calls.get("t1").map(|t| &t.tracking.status),
            Some(&EffectStatus::Pending)
        );
        assert_eq!(
            agg.state
                .sub_agent_calls
                .get("child-1")
                .map(|s| &s.tracking.status),
            Some(&EffectStatus::Pending)
        );
        assert!(
            fired_tool_result(&events).is_empty(),
            "nothing has completed yet"
        );
    }

    #[test]
    fn timed_out_effect_fires_tool_result_via_wake() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "t1".to_string(),
                name: "tool_t1".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy {
                    timeout_secs: Some(60),
                    max_retries: 0,
                    backoff_base_secs: 0,
                    backoff_max_secs: 0,
                },
            },
            &system(),
        );

        let past = Utc::now() + chrono::Duration::seconds(120);
        let errored = dispatch(&mut agg, CommandPayload::Wake { now: past }, &system());
        assert!(
            errored
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallErrored(_))),
            "deadline exceeded errors the tool; got {errored:?}"
        );
        assert_eq!(
            fired_tool_result(&errored),
            vec!["t1".to_string()],
            "the timed-out call fires a tool.finished"
        );
        assert!(
            fired_tool_result(&wake(&mut agg)).is_empty(),
            "the next wake does not re-fire"
        );
    }

    #[test]
    fn completion_fires_tool_result_once_and_wake_does_not_re_fire() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "a");

        assert!(
            !fired_tool_result(&complete_tool(&mut agg, "a", "RA")).is_empty(),
            "fires on completion"
        );
        assert!(
            fired_tool_result(&wake(&mut agg)).is_empty(),
            "wake does not re-fire"
        );
    }

    #[test]
    fn resume_interrupt_emits_resumed() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected [InterruptResumed, WorkerDecisionRequested]; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn machine_cannot_resume_system_interrupt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "budget_exhausted".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let err = agg
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".to_string(),
                    payload: serde_json::Value::Null,
                },
                &Caller::Machine {
                    tenant_id: "tenant-a".to_string(),
                    key_id: "prod-key-1".to_string(),
                },
            )
            .expect_err("machine caller should not resume a system interrupt");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "system caller should resume a system interrupt; got {events:?}"
        );
    }

    #[test]
    fn machine_resumes_machine_interrupt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let machine = Caller::Machine {
            tenant_id: "tenant-a".to_string(),
            key_id: "prod-key-1".to_string(),
        };
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "awaiting approval".to_string(),
                payload: serde_json::Value::Null,
            },
            &machine,
        );
        assert!(matches!(
            agg.state.projected_status(),
            SessionStatus::Interrupted {
                origin: InterruptOrigin::Machine,
                ..
            }
        ));

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &machine,
        );
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "machine caller should resume its own interrupt; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    fn frontend_caller(tenant_id: &str, user_id: &str) -> Caller {
        Caller::Frontend {
            tenant_id: tenant_id.to_string(),
            user_id: user_id.to_string(),
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn frontend_interrupts_and_resumes_own_session() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let frontend = frontend_caller("tenant-a", "user-1");

        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "user paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &frontend,
        );
        assert!(matches!(
            agg.state.projected_status(),
            SessionStatus::Interrupted {
                origin: InterruptOrigin::Frontend,
                ..
            }
        ));

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &frontend,
        );
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "frontend owner should resume its own interrupt; got {events:?}"
        );
        assert_eq!(agg.state.status, SessionStatus::Idle);
    }

    #[test]
    fn non_owner_frontend_cannot_interrupt() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let err = agg
            .state
            .handle(
                CommandPayload::Interrupt {
                    interrupt_id: "int-1".to_string(),
                    reason: "user paused".to_string(),
                    payload: serde_json::Value::Null,
                },
                &frontend_caller("tenant-a", "user-2"),
            )
            .expect_err("frontend caller should not interrupt another user's session");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn non_owner_frontend_cannot_resume() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "user paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &frontend_caller("tenant-a", "user-1"),
        );

        let err = agg
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".to_string(),
                    payload: serde_json::Value::Null,
                },
                &frontend_caller("tenant-a", "user-2"),
            )
            .expect_err("frontend caller should not resume another user's session");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );

        let err = agg
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".to_string(),
                    payload: serde_json::Value::Null,
                },
                &frontend_caller("tenant-b", "user-1"),
            )
            .expect_err("frontend caller from another tenant should be denied");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn frontend_cannot_resume_machine_interrupt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "awaiting approval".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::Machine {
                tenant_id: "tenant-a".to_string(),
                key_id: "prod-key-1".to_string(),
            },
        );

        let err = agg
            .state
            .handle(
                CommandPayload::ResumeInterrupt {
                    interrupt_id: "int-1".to_string(),
                    payload: serde_json::Value::Null,
                },
                &frontend_caller("tenant-a", "user-1"),
            )
            .expect_err("frontend caller should not resume a machine interrupt");
        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn machine_resumes_frontend_interrupt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "user paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &frontend_caller("tenant-a", "user-1"),
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::Machine {
                tenant_id: "tenant-a".to_string(),
                key_id: "prod-key-1".to_string(),
            },
        );
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "machine caller should resume a frontend interrupt; got {events:?}"
        );
    }

    #[test]
    fn client_action_rejected_while_interrupted() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload: ClientPayload::Action(crate::protocol::ClientAction {
                        name: "refresh".to_string(),
                        args: None,
                    }),
                    turn_id: None,
                },
                &Caller::System {
                    tenant_id: "tenant-a".to_string(),
                },
            )
            .expect_err("client actions should be rejected while interrupted");
        assert!(
            matches!(err, SessionError::SessionInterrupted),
            "expected SessionInterrupted; got {err:?}"
        );
    }

    #[test]
    fn cancel_voids_pending_effects() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("hi".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");
        request_client_tool(&mut agg, "tc-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-1".to_string(),
                agent_id: "helper".to_string(),
                tool_call_id: "call-1".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &system(),
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::CancelSession,
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let mut voided = voided_ids(&events);
        voided.sort_unstable();
        assert_eq!(voided, vec!["call-1", "llm-1", "tc-1"], "got {events:?}");
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::CallVoided(v)
                    if v.kind == EffectKind::SubAgent
                        && v.session_id.as_deref() == Some("child-1")
            )),
            "the sub-agent void names the child session for the cascade; got {events:?}"
        );
        assert!(!agg.state.has_pending_worker_decision());

        let again = dispatch(
            &mut agg,
            CommandPayload::CancelSession,
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(again.is_empty(), "got {again:?}");

        let stale = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![Action::Done {
                    data: serde_json::Value::Null,
                }],
                state: None,
                agent: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(
            stale.is_empty(),
            "stale submission after cancel should no-op; got {stale:?}"
        );
    }

    #[test]
    fn interrupt_voids_llm_calls_but_spares_tools() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);

        let events = dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: String::new(),
                payload: serde_json::Value::Null,
            },
            &system(),
        );
        assert_eq!(voided_ids(&events), vec!["llm-1"], "got {events:?}");
        assert_eq!(
            agg.state.tool_calls.get("tc-1").unwrap().tracking.status,
            EffectStatus::Pending,
            "tools settle during an interrupt and queue"
        );
    }

    #[test]
    fn interrupt_action_voids_llm_calls_requested_in_the_same_submit() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![node_msg("u1", Role::User, "hi")],
                actions: vec![
                    Action::CallLlm {
                        format: None,
                        id: "llm-1".to_string(),
                        request: request_with(vec![]),
                        stream: false,
                        retry: RetryPolicy::no_retry(),
                        handler: LlmHandler::Server,
                    },
                    Action::Interrupt {
                        interrupt_id: "int-1".to_string(),
                        reason: "hold".to_string(),
                        payload: serde_json::Value::Null,
                    },
                ],
                state: None,
                agent: None,
            },
            &machine(),
        );
        assert_eq!(voided_ids(&events), vec!["llm-1"], "got {events:?}");
        assert_eq!(
            agg.state.llm_calls.get("llm-1").unwrap().tracking.status,
            EffectStatus::Failed,
        );
    }

    #[test]
    fn interrupt_voids_pending_worker_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("hi".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "quota_exhausted".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(!agg.state.has_pending_worker_decision());

        // A late submission from the worker is a no-op, not an error.
        let stale = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![Action::Done {
                    data: serde_json::Value::Null,
                }],
                state: None,
                agent: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(
            stale.is_empty(),
            "stale submission should no-op; got {stale:?}"
        );

        // Resume requests a fresh decision immediately.
        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "expected immediate WorkerDecisionRequested; got {events:?}"
        );
    }

    #[test]
    fn tool_result_during_interrupt_queues_until_resume() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let system = Caller::System {
            tenant_id: "tenant-a".to_string(),
        };
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("crawl the site".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &system,
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![Action::CallTool {
                    id: "tc-1".to_string(),
                    name: "crawl".to_string(),
                    arguments: "{}".to_string(),
                    retry: RetryPolicy::no_retry(),
                }],
                state: None,
                agent: None,
            },
            &system,
        );

        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "quota_exhausted".to_string(),
                payload: serde_json::Value::Null,
            },
            &system,
        );

        // The late tool result is recorded but its decision is queued, not delivered.
        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                result: "done".to_string(),
            },
            &system,
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "tool result should be recorded; got {events:?}"
        );
        assert_eq!(
            agg.state.tool_calls.get("tc-1").unwrap().result.as_deref(),
            Some("done")
        );
        // The tool.finished trigger is queued while interrupted; only its delivery is deferred.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(_))),
            "decision should be queued while interrupted; got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))),
            "no decision should be delivered while interrupted; got {events:?}"
        );

        // Resume delivers interrupt.resumed first...
        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &system,
        );
        let resumed_decision_id = events
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(p)
                    if matches!(p.trigger, Trigger::InterruptResumed { .. }) =>
                {
                    Some(p.decision_id.clone())
                }
                _ => None,
            })
            .expect("resume should request an interrupt.resumed decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: resumed_decision_id,
                transcript: vec![],
                actions: vec![],
                state: None,
                agent: None,
            },
            &system,
        );
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => agg
                    .state
                    .worker_decisions
                    .get(&p.decision_id)
                    .map(|d| d.trigger.clone()),
                _ => None,
            })
            .expect("queued decision should promote after the resumed decision completes");
        assert!(
            matches!(trigger, Trigger::ToolFinished { .. }),
            "expected a tool.finished trigger; got {trigger:?}"
        );
    }

    #[test]
    fn worker_interrupt_action_pauses_session_and_resume_carries_payload() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("send the email".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions: vec![Action::Interrupt {
                    interrupt_id: "int-1".to_string(),
                    reason: "confirmation".to_string(),
                    payload: serde_json::json!({"message": "Send the email?"}),
                }],
                state: None,
                agent: None,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::SessionInterrupted(_))),
            "interrupt action should emit SessionInterrupted; got {events:?}"
        );
        assert!(matches!(
            agg.state.projected_status(),
            SessionStatus::Interrupted {
                origin: InterruptOrigin::Frontend,
                ..
            }
        ));

        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::json!({"approved": true}),
            },
            &frontend_caller("tenant-a", "user-1"),
        );
        let trigger = events
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(p) => Some(p.trigger.clone()),
                _ => None,
            })
            .expect("resume should request a worker decision");
        match trigger {
            Trigger::InterruptResumed {
                interrupt_id,
                payload,
            } => {
                assert_eq!(interrupt_id, "int-1");
                assert_eq!(payload, serde_json::json!({"approved": true}));
            }
            other => panic!("expected InterruptResumed trigger; got {other:?}"),
        }
    }

    #[test]
    fn fail_worker_decision_emits_errored() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup_events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: DraftMessage {
                        id: None,
                        role: Role::User,
                        content: Some(Content::Text("hi".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );
        let decision_id = setup_events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message should request a worker decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::FailWorkerDecision {
                decision_id: decision_id.clone(),
                error: "worker offline".to_string(),
                retryable: false,
            },
            &Caller::System {
                tenant_id: "tenant-a".to_string(),
            },
        );

        // A terminal failure also ends the run: the turn it was driving can never
        // settle on its own, and TurnCompleted is the only terminal consumers watch.
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::WorkerDecisionErrored(_),
                    EventPayload::TurnCompleted(_),
                    EventPayload::SessionDone(_)
                ]
            ),
            "expected [WorkerDecisionErrored, TurnCompleted, SessionDone]; got {events:?}"
        );
        let completed = turn_completed(&events).expect("the turn ends");
        assert_eq!(completed.turn_id, "turn-1");
        assert_eq!(
            completed.error.as_deref(),
            Some("worker offline"),
            "the turn carries the failure; got {completed:?}"
        );
        assert!(
            !agg.state.worker_decisions.contains_key(&decision_id),
            "a settled decision leaves the map"
        );
    }

    #[test]
    fn machine_caller_from_wrong_tenant_is_denied() {
        let agg = create_session("sess-1", "tenant-a", "user-1");

        let cross_tenant_machine = Caller::Machine {
            tenant_id: "tenant-b".to_string(),
            key_id: "key-from-tenant-b".to_string(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: Some(0),
                    result: "ok".to_string(),
                },
                &cross_tenant_machine,
            )
            .expect_err("machine from a different tenant should be rejected");

        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn frontend_caller_with_mismatched_tenant_on_create_session_is_denied() {
        let session_id = "sess-1".to_string();
        let agg = SessionAggregate::new(
            session_id.clone(),
            "tenant-a".to_string(),
            SessionState::new(session_id),
        );

        let caller = Caller::Frontend {
            tenant_id: "tenant-a".to_string(),
            user_id: "user-1".to_string(),
            attrs: HashMap::new(),
        };

        let err = agg
            .state
            .handle(
                CommandPayload::CreateSession {
                    agent_id: "agent-1".to_string(),
                    owner: SessionOwner {
                        tenant_id: "tenant-b".to_string(),
                        id: Some("user-1".to_string()),
                        metadata: HashMap::new(),
                    },
                    ancestry: vec![],
                    worker_retry: RetryPolicy::no_retry(),
                },
                &caller,
            )
            .expect_err("creating a session in a different tenant should be rejected");

        assert!(
            matches!(err, SessionError::SessionAccessDenied),
            "expected SessionAccessDenied; got {err:?}"
        );
    }

    #[test]
    fn parallel_tool_results_record_in_completion_order() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let sys = Caller::System {
            tenant_id: "tenant-a".to_string(),
        };

        let request = |agg: &mut SessionAggregate, id: &str| {
            dispatch(
                agg,
                CommandPayload::RequestToolCall {
                    tool_call_id: id.to_string(),
                    name: "t".to_string(),
                    arguments: "{}".to_string(),
                    retry: RetryPolicy::no_retry(),
                },
                &sys,
            );
        };
        request(&mut agg, "tc-a");
        request(&mut agg, "tc-b");

        let complete = |agg: &mut SessionAggregate, id: &str| {
            dispatch(
                agg,
                CommandPayload::CompleteToolCall {
                    tool_call_id: id.to_string(),
                    attempt: Some(0),
                    result: format!("result-{id}"),
                },
                &sys,
            )
        };

        let first = complete(&mut agg, "tc-b");
        assert_eq!(fired_tool_result(&first), vec!["tc-b".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("tc-b").unwrap().result.as_deref(),
            Some("result-tc-b")
        );

        let second = complete(&mut agg, "tc-a");
        assert_eq!(fired_tool_result(&second), vec!["tc-a".to_string()]);
        assert_eq!(
            agg.state.tool_calls.get("tc-a").unwrap().result.as_deref(),
            Some("result-tc-a")
        );
    }

    #[test]
    fn worker_append_action_writes_a_tree_node() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let setup = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                }),
                turn_id: Some("turn-1".to_string()),
            },
            &system(),
        );
        let decision_id = decision_with(&setup, |t| matches!(t, Trigger::ClientMessage { .. }))
            .expect("user message decision");

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![node_msg("u1", Role::User, "hi")],
                actions: vec![],
                state: None,
                agent: None,
            },
            &machine(),
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::NewMessage(m) if m.message.id == "u1")),
            "append writes a NewMessage; got {events:?}"
        );
        assert_eq!(agg.state.head_id.as_deref(), Some("u1"));
    }

    #[test]
    fn complete_tool_call_fires_tool_result_without_appending() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");

        let events = complete_tool(&mut agg, "tc-1", "ok");

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "expected a ToolCallCompleted event; got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::NewMessage(_))),
            "the engine appends no node on completion; got {events:?}"
        );
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);

        let tc = agg.state.tool_calls.get("tc-1").expect("tool call present");
        assert_eq!(tc.tracking.status, EffectStatus::Completed);
        assert_eq!(tc.result.as_deref(), Some("ok"));
        assert!(!tc.is_error);
    }

    // ── Parallel worker-tool results must stay on one linear path ─────────

    fn submit_decision(
        agg: &mut SessionAggregate,
        decision_id: String,
        transcript: Vec<DraftMessage>,
        actions: Vec<Action>,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript,
                actions,
                state: None,
                agent: None,
            },
            &machine(),
        )
    }

    fn call_tool_action(id: &str) -> Action {
        Action::CallTool {
            id: id.to_string(),
            name: "find".to_string(),
            arguments: "{}".to_string(),
            retry: RetryPolicy::no_retry(),
        }
    }

    fn tool_result_action(id: &str) -> Action {
        Action::ToolResult {
            id: id.to_string(),
            attempt: Some(0),
            result: format!("result-{id}"),
        }
    }

    /// The transcript the runtime would hand a decision requested right now:
    /// the root→head path, exactly as `try_extract` materializes it.
    fn delivered_transcript(agg: &SessionAggregate) -> Vec<DraftMessage> {
        agg.state
            .head_id
            .as_deref()
            .map(|h| agg.state.message_tree().path_to(h))
            .unwrap_or_default()
            .into_iter()
            .map(DraftMessage::from)
            .collect()
    }

    fn pending_worker_decisions(agg: &SessionAggregate) -> usize {
        agg.state
            .worker_decisions
            .values()
            .filter(|d| d.tracking.status == EffectStatus::Pending)
            .count()
    }

    /// The single live (pending) decision — its id and trigger.
    fn live_decision(agg: &SessionAggregate) -> (String, Trigger) {
        agg.state
            .worker_decisions
            .values()
            .filter(|d| d.tracking.status == EffectStatus::Pending)
            .min_by_key(|d| d.source_event_sequence)
            .map(|d| (d.decision_id.clone(), d.trigger.clone()))
            .expect("a live decision")
    }

    /// Freeze the transcript each newly-requested decision is handed — the
    /// root→head path at request time, exactly as `try_extract` materializes it
    /// into the delivered decision. A decision keeps this base even if the head
    /// later moves; that is why two writers promoted against the same head fork.
    fn record_bases(
        events: &[EventPayload],
        agg: &SessionAggregate,
        bases: &mut HashMap<String, Vec<DraftMessage>>,
    ) {
        let frozen = delivered_transcript(agg);
        for e in events {
            if let EventPayload::WorkerDecisionRequested(p) = e {
                bases.insert(p.decision_id.clone(), frozen.clone());
            }
        }
    }

    /// Drive the worker to quiescence the way the runtime does: a single thread
    /// answers each live decision with the transcript it was frozen, and a wake
    /// fires after every step to surface queued work. Executes reply with a
    /// result; finishes append their result node to the frozen base.
    fn drive_worker(agg: &mut SessionAggregate, bases: &mut HashMap<String, Vec<DraftMessage>>) {
        for _ in 0..128 {
            let mut live: Vec<(String, Trigger)> = agg
                .state
                .worker_decisions
                .values()
                .filter(|d| d.tracking.status == EffectStatus::Pending)
                .map(|d| (d.decision_id.clone(), d.trigger.clone()))
                .collect();
            live.sort_by(|a, b| a.0.cmp(&b.0));

            let Some((id, trigger)) = live.into_iter().next() else {
                if !agg.state.has_queued_worker_decision() {
                    return;
                }
                let woken = wake(agg);
                record_bases(&woken, agg, bases);
                continue;
            };

            let base = bases.get(&id).cloned().unwrap_or_default();
            let events = match trigger {
                Trigger::ToolExecute { id: tid, .. } => {
                    submit_decision(agg, id, vec![], vec![tool_result_action(&tid)])
                }
                Trigger::ToolFinished { id: tid, name, .. } => {
                    let mut answer = base;
                    answer.push(tool_msg(&tid, &format!("done-{name}")));
                    submit_decision(agg, id, answer, vec![])
                }
                _ => submit_decision(agg, id, base, vec![]),
            };
            record_bases(&events, agg, bases);

            let woken = wake(agg);
            record_bases(&woken, agg, bases);
        }
        panic!("worker did not settle");
    }

    // A wake exists to promote a queued worker decision when the session is
    // otherwise idle. It must not promote one while a decision is already live —
    // a second live decision lets two transcript writers run against the same
    // head and fork the tree. This is the exact hole that split a real session's
    // parallel tool results (a wake fired between two tool.execute submits).
    #[test]
    fn a_wake_does_not_promote_a_second_decision_while_one_is_pending() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        // One decision fans out into two worker-handled tool calls: one execute
        // is promoted inline, the other queues behind it.
        let d0 = decision_with(
            &submit_messages(&mut agg, vec![node_msg("u1", Role::User, "hi")]),
            |_| true,
        )
        .expect("a client decision");
        submit_decision(
            &mut agg,
            d0,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("asst", Role::Assistant, "calling"),
            ],
            vec![call_tool_action("tc-a"), call_tool_action("tc-b")],
        );
        assert_eq!(pending_worker_decisions(&agg), 1, "one execute live");
        assert!(agg.state.has_queued_worker_decision(), "one execute queued");

        wake(&mut agg);

        assert_eq!(
            pending_worker_decisions(&agg),
            1,
            "the wake promoted a second decision while one was pending — the \
             serialization hole that forks parallel tool results"
        );
    }

    // The downstream symptom: a model that fans out into parallel worker-handled
    // tool calls must land every result on one linear path, so the next LLM call
    // is sent all of them. A forked tree hides some results from the model.
    #[test]
    fn parallel_tool_finishes_keep_results_on_one_path() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let mut bases: HashMap<String, Vec<DraftMessage>> = HashMap::new();

        // Record [user, assistant] and fan out into two worker-handled tool
        // calls in one batch — the shape a parallel tool call produces.
        let d0 = decision_with(
            &submit_messages(&mut agg, vec![node_msg("u1", Role::User, "hi")]),
            |_| true,
        )
        .expect("a client decision");
        let fanout = submit_decision(
            &mut agg,
            d0,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("asst", Role::Assistant, "calling"),
            ],
            vec![call_tool_action("tc-a"), call_tool_action("tc-b")],
        );
        record_bases(&fanout, &agg, &mut bases);
        let woken = wake(&mut agg);
        record_bases(&woken, &agg, &mut bases);

        drive_worker(&mut agg, &mut bases);

        // Both results must lie on the single path an LLM call would be sent.
        let head = agg.state.head_id.clone().expect("a head");
        let path = agg.state.message_tree().path_to(&head);
        let seen: Vec<&str> = path
            .iter()
            .filter_map(|m| m.tool_call_id.as_deref())
            .collect();
        assert!(
            seen.contains(&"tc-a") && seen.contains(&"tc-b"),
            "parallel results forked: path holds {seen:?}, not both tc-a and tc-b"
        );
    }

    // Parallel worker tools all complete before any `tool.finished` is processed
    // (executes queue ahead of finishes), so counting only in-flight calls leaves
    // every finish seeing zero pending work — and each prompts, the earlier ones
    // against a transcript missing the still-unrecorded sibling results. Only the
    // last finish should prompt; `pending_work` also counts sibling finishes
    // still awaiting recording.
    #[test]
    fn only_the_last_parallel_tool_finish_reports_no_pending_work() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d0 = decision_with(
            &submit_messages(&mut agg, vec![node_msg("u1", Role::User, "hi")]),
            |_| true,
        )
        .expect("a client decision");
        submit_decision(
            &mut agg,
            d0,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("asst", Role::Assistant, "calling"),
            ],
            vec![call_tool_action("tc-a"), call_tool_action("tc-b")],
        );

        // Run both executes; both tool calls complete and their finishes queue.
        let (exec_a, _) = live_decision(&agg);
        submit_decision(&mut agg, exec_a, vec![], vec![tool_result_action("tc-a")]);
        let (exec_b, _) = live_decision(&agg);
        submit_decision(&mut agg, exec_b, vec![], vec![tool_result_action("tc-b")]);

        // Both calls are done — in-flight work is zero. The first finish is live
        // with the second still queued, so it must report pending work, or it
        // would prompt now against a transcript missing the second result.
        let (finish_first, trigger) = live_decision(&agg);
        assert!(matches!(trigger, Trigger::ToolFinished { .. }));
        assert_eq!(
            agg.state.event_meta().pending_work(&finish_first),
            1,
            "the first finish must wait: a sibling result is still unrecorded"
        );

        // Record the first result; the last finish is now live, nothing pending.
        let mut answer = delivered_transcript(&agg);
        answer.push(tool_msg("tc-a", "A"));
        submit_decision(&mut agg, finish_first, answer, vec![]);

        let (finish_last, trigger) = live_decision(&agg);
        assert!(matches!(trigger, Trigger::ToolFinished { .. }));
        assert_eq!(
            agg.state.event_meta().pending_work(&finish_last),
            0,
            "the last finish prompts: every result is recorded"
        );
    }

    // Deferred and client-handled tools (and sub-agents) leave their call
    // Pending after `tool.execute` returns — the result arrives later, out of
    // band. A fast sibling's finish must not prompt while that call is still in
    // flight, or the model is prompted without the deferred result. An in-flight
    // call keeps `pending_work` non-zero exactly as an unrecorded finish does, so
    // the two patterns compose. (The old in-flight-only count already handled
    // this half; the point here is the fix did not regress it.)
    #[test]
    fn an_in_flight_deferred_sibling_keeps_a_fast_tool_from_prompting() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d0 = decision_with(
            &submit_messages(&mut agg, vec![node_msg("u1", Role::User, "hi")]),
            |_| true,
        )
        .expect("a client decision");
        submit_decision(
            &mut agg,
            d0,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("asst", Role::Assistant, "calling"),
            ],
            vec![call_tool_action("tc-fast"), call_tool_action("tc-deferred")],
        );

        // Answer the fast tool's execute with a result; "start" the deferred one
        // by answering its execute with nothing — the engine leaves its call
        // Pending, just as a deferred tool or a client-handled tool would.
        let (exec_fast, t) = live_decision(&agg);
        assert!(matches!(&t, Trigger::ToolExecute { id, .. } if id == "tc-fast"));
        submit_decision(
            &mut agg,
            exec_fast,
            vec![],
            vec![tool_result_action("tc-fast")],
        );

        let (exec_deferred, t) = live_decision(&agg);
        assert!(matches!(&t, Trigger::ToolExecute { id, .. } if id == "tc-deferred"));
        submit_decision(&mut agg, exec_deferred, vec![], vec![]);

        // The fast tool's finish is live while the deferred call is still in
        // flight — it must report pending work and record only, not prompt.
        let (finish_fast, trigger) = live_decision(&agg);
        assert!(matches!(trigger, Trigger::ToolFinished { .. }));
        assert_eq!(
            agg.state
                .tool_calls
                .get("tc-deferred")
                .unwrap()
                .tracking
                .status,
            EffectStatus::Pending,
            "the deferred call stays in flight after its execute"
        );
        assert_eq!(
            agg.state.event_meta().pending_work(&finish_fast),
            1,
            "the fast tool must wait: a deferred sibling is still in flight"
        );
    }

    // ── Concurrent LLM calls (fan-out) ───────────────────────────────────

    fn call_llm_action(id: &str, handler: LlmHandler) -> Action {
        Action::CallLlm {
            format: None,
            id: id.to_string(),
            request: request_with(vec![]),
            stream: false,
            retry: RetryPolicy::no_retry(),
            handler,
        }
    }

    fn llm_response(content: &str) -> LlmResponse {
        LlmResponse {
            model: "test-model".to_string(),
            content: Some(content.to_string()),
            tool_calls: vec![],
            finish_reason: Some("stop".to_string()),
            usage: None,
            cost: None,
            images: vec![],
        }
    }

    fn request_llm(agg: &mut SessionAggregate, id: &str, handler: LlmHandler) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::RequestLlmCall {
                format: None,
                call_id: id.to_string(),
                request: request_with(vec![]),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler,
            },
            &system(),
        )
    }

    fn complete_llm(
        agg: &mut SessionAggregate,
        id: &str,
        attempt: u32,
        caller: &Caller,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::CompleteLlmCall {
                call_id: id.to_string(),
                attempt: Some(attempt),
                response: llm_response("ok"),
            },
            caller,
        )
    }

    /// Open a worker decision (via a user message) and answer it with `actions`.
    fn submit_decision_with(agg: &mut SessionAggregate, actions: Vec<Action>) -> Vec<EventPayload> {
        let setup = dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("seed", Role::User, "seed"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        let decision_id = setup
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("user message opens a decision");
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript: vec![],
                actions,
                state: None,
                agent: None,
            },
            &machine(),
        )
    }

    /// The call ids of every `llm.execute` trigger, deduped by decision id.
    fn fired_llm_execute(events: &[EventPayload]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        events
            .iter()
            .filter_map(|e| {
                let (decision_id, trigger) = match e {
                    EventPayload::DecisionRequestQueued(p) => (&p.decision_id, &p.trigger),
                    _ => return None,
                };
                match trigger {
                    Trigger::LlmExecute { id, .. } if seen.insert(decision_id.clone()) => {
                        Some(id.clone())
                    }
                    _ => None,
                }
            })
            .collect()
    }

    /// The call ids of every `llm.finished` trigger, deduped by decision id.
    fn settled_llm_ids(events: &[EventPayload]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        events
            .iter()
            .filter_map(|e| {
                let (decision_id, trigger) = match e {
                    EventPayload::DecisionRequestQueued(p) => (&p.decision_id, &p.trigger),
                    _ => return None,
                };
                match trigger {
                    Trigger::LlmFinished { id, .. } if seen.insert(decision_id.clone()) => {
                        Some(id.clone())
                    }
                    _ => None,
                }
            })
            .collect()
    }

    #[test]
    fn two_server_llm_calls_in_one_decision_both_issue() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = submit_decision_with(
            &mut agg,
            vec![
                call_llm_action("llm-1", LlmHandler::Server),
                call_llm_action("llm-2", LlmHandler::Server),
            ],
        );
        let requested: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                EventPayload::LlmCallRequested(r) => Some(r.call_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            requested,
            vec!["llm-1".to_string(), "llm-2".to_string()],
            "both calls issue — the single-pending gate is gone; got {events:?}"
        );
        assert_eq!(agg.state.llm_calls.len(), 2);
        assert!(agg
            .state
            .llm_calls
            .values()
            .all(|c| c.tracking.status == EffectStatus::Pending));
        assert_eq!(agg.state.effects().len(), 2, "both in flight");
    }

    #[test]
    fn worker_handled_llm_fanout_delegates_both_executes() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = submit_decision_with(
            &mut agg,
            vec![
                call_llm_action("llm-1", LlmHandler::Worker),
                call_llm_action("llm-2", LlmHandler::Worker),
            ],
        );
        let mut execs = fired_llm_execute(&events);
        execs.sort();
        assert_eq!(
            execs,
            vec!["llm-1".to_string(), "llm-2".to_string()],
            "each worker-handled call delegates an llm.execute; got {events:?}"
        );
        assert_eq!(agg.state.effects().len(), 2);
    }

    #[test]
    fn reverse_order_llm_completion_settles_independently() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        request_llm(&mut agg, "llm-2", LlmHandler::Server);
        assert_eq!(agg.state.effects().len(), 2);

        let e2 = complete_llm(&mut agg, "llm-2", 0, &system());
        assert!(settled_llm_ids(&e2).contains(&"llm-2".to_string()));
        let remaining: Vec<String> = agg.state.effects().iter().map(|e| e.id.clone()).collect();
        assert_eq!(
            remaining,
            vec!["llm-1".to_string()],
            "only the unsettled call remains in flight"
        );

        let e1 = complete_llm(&mut agg, "llm-1", 0, &system());
        assert!(settled_llm_ids(&e1).contains(&"llm-1".to_string()));
        assert!(agg.state.effects().is_empty());
    }

    #[test]
    fn re_requesting_a_completed_llm_id_noops() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        complete_llm(&mut agg, "llm-1", 0, &system());
        let again = request_llm(&mut agg, "llm-1", LlmHandler::Server);
        assert!(
            again.is_empty(),
            "re-request of a Completed id is an idempotent no-op; got {again:?}"
        );
    }

    #[test]
    fn llm_and_tool_with_the_same_id_coexist() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "shared");
        request_llm(&mut agg, "shared", LlmHandler::Server);
        assert!(agg.state.tool_calls.contains_key("shared"));
        assert!(agg.state.llm_calls.contains_key("shared"));
        assert_eq!(
            agg.state.effects().len(),
            2,
            "distinct maps keep a tool and an llm call with the same id apart"
        );
    }

    #[test]
    fn reissue_llm_after_interrupt_with_the_same_id() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: String::new(),
                payload: serde_json::Value::Null,
            },
            &system(),
        );
        assert_eq!(
            agg.state.llm_calls.get("llm-1").unwrap().tracking.status,
            EffectStatus::Failed,
            "interrupt voids the pending call"
        );
        dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::Value::Null,
            },
            &system(),
        );
        let events = request_llm(&mut agg, "llm-1", LlmHandler::Server);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::LlmCallRequested(_))),
            "a Failed id re-issues on the same key; got {events:?}"
        );
        assert_eq!(
            agg.state.llm_calls.get("llm-1").unwrap().tracking.status,
            EffectStatus::Pending
        );
    }

    #[test]
    fn machine_settles_worker_handled_llm_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Worker);
        let events = complete_llm(&mut agg, "llm-1", 0, &machine());
        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::LlmCallCompleted(_))));
        assert!(settled_llm_ids(&events).contains(&"llm-1".to_string()));
        assert_eq!(
            agg.state.llm_calls.get("llm-1").unwrap().tracking.status,
            EffectStatus::Completed
        );
    }

    #[test]
    fn machine_fails_worker_handled_llm_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Worker);
        let events = dispatch(
            &mut agg,
            CommandPayload::FailLlmCall {
                call_id: "llm-1".to_string(),
                attempt: Some(0),
                error: "boom".to_string(),
                retryable: false,
                code: None,
                detail: None,
            },
            &machine(),
        );
        assert!(events
            .iter()
            .any(|e| matches!(e, EventPayload::LlmCallErrored(_))));
        assert!(
            settled_llm_ids(&events).contains(&"llm-1".to_string()),
            "no-retry failure settles the call; got {events:?}"
        );
    }

    #[test]
    fn machine_cannot_settle_engine_handled_llm_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        let err = agg
            .state
            .handle(
                CommandPayload::CompleteLlmCall {
                    call_id: "llm-1".to_string(),
                    attempt: Some(0),
                    response: llm_response("hi"),
                },
                &machine(),
            )
            .expect_err("an engine-handled call is not the machine's to settle");
        assert!(
            matches!(err, SessionError::EffectWrongHandler),
            "expected EffectWrongHandler; got {err:?}"
        );
    }

    #[test]
    fn machine_wrong_attempt_on_worker_llm_is_mismatch() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Worker);
        let err = agg
            .state
            .handle(
                CommandPayload::CompleteLlmCall {
                    call_id: "llm-1".to_string(),
                    attempt: Some(7),
                    response: llm_response("hi"),
                },
                &machine(),
            )
            .expect_err("a stale attempt is a mismatch");
        assert!(
            matches!(err, SessionError::EffectAttemptMismatch),
            "expected EffectAttemptMismatch; got {err:?}"
        );
    }

    #[test]
    fn machine_settle_of_unknown_llm_is_not_found() {
        let agg = create_session("sess-1", "tenant-a", "user-1");
        let err = agg
            .state
            .handle(
                CommandPayload::CompleteLlmCall {
                    call_id: "nope".to_string(),
                    attempt: Some(0),
                    response: llm_response("hi"),
                },
                &machine(),
            )
            .expect_err("unknown effect");
        assert!(
            matches!(err, SessionError::EffectNotFound),
            "expected EffectNotFound; got {err:?}"
        );
    }

    #[test]
    fn system_duplicate_llm_completion_is_silent_noop() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        complete_llm(&mut agg, "llm-1", 0, &system());
        let again = complete_llm(&mut agg, "llm-1", 0, &system());
        assert!(
            again.is_empty(),
            "a duplicate executor completion stays a silent no-op; got {again:?}"
        );
    }

    #[test]
    fn done_then_late_llm_completion_still_settles() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);
        dispatch(
            &mut agg,
            CommandPayload::FinishTurn {
                data: serde_json::Value::Null,
            },
            &system(),
        );
        let events = complete_llm(&mut agg, "llm-1", 0, &system());
        assert!(
            settled_llm_ids(&events).contains(&"llm-1".to_string()),
            "a late settle after done still fires a decision; got {events:?}"
        );
    }

    // ── turn.finished notification ───────────────────────────────────────

    fn create_session_with_retry(retry: RetryPolicy) -> SessionAggregate {
        let mut agg = SessionAggregate::new(
            "sess-1".to_string(),
            "tenant-a".to_string(),
            SessionState::new("sess-1".to_string()),
        );
        dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: "tenant-a".to_string(),
                    id: Some("user-1".to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: retry,
            },
            &system(),
        );
        drain_session_start(&mut agg);
        agg
    }

    /// Open a turn via a client message and answer it with `done`, driving pass 1
    /// (TurnCompleted + a queued/promoted turn.finished decision).
    fn drive_turn_done(
        agg: &mut SessionAggregate,
        turn_id: &str,
        data: serde_json::Value,
    ) -> Vec<EventPayload> {
        let setup = dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("seed", Role::User, "seed"),
                    stream: false,
                }),
                turn_id: Some(turn_id.to_string()),
            },
            &system(),
        );
        let d = decision_with(&setup, |t| matches!(t, Trigger::ClientMessage { .. }))
            .expect("client message opens a decision");
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d,
                transcript: vec![],
                actions: vec![Action::Done { data }],
                state: None,
                agent: None,
            },
            &machine(),
        )
    }

    fn turn_finished_decision(events: &[EventPayload]) -> Option<String> {
        decision_with(events, |t| matches!(t, Trigger::TurnFinished { .. }))
    }

    fn has_session_done(events: &[EventPayload]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, EventPayload::SessionDone(_)))
    }

    fn turn_completed(events: &[EventPayload]) -> Option<&TurnCompleted> {
        events.iter().find_map(|e| match e {
            EventPayload::TurnCompleted(tc) => Some(tc),
            _ => None,
        })
    }

    #[test]
    fn turn_finished_notifies_worker_and_defers_completion() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = drive_turn_done(&mut agg, "t1", serde_json::json!("answer"));

        // Pass 1 emits no terminal — the turn.finished trigger IS the pass-1 signal.
        assert!(
            turn_completed(&events).is_none(),
            "pass 1 does not complete the turn; got {events:?}"
        );
        assert!(
            !has_session_done(&events),
            "SessionDone is deferred; got {events:?}"
        );
        let queued = events
            .iter()
            .find_map(|e| match e {
                EventPayload::DecisionRequestQueued(p) => match &p.trigger {
                    Trigger::TurnFinished { turn_id, data, .. } => {
                        Some((p.decision_id.clone(), turn_id.clone(), data.clone()))
                    }
                    _ => None,
                },
                _ => None,
            })
            .expect("turn.finished is queued");
        assert_eq!(queued.1, "t1");
        assert_eq!(queued.2, serde_json::json!("answer"), "carries turn output");
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(w) if w.decision_id == queued.0
            )),
            "the deferred decision is promoted; got {events:?}"
        );
        // The frozen output is captured for pass 2.
        let f = agg.state.finalizing.as_ref().expect("finalizing set");
        assert_eq!(f.turn_id, "t1");
        assert_eq!(f.data, serde_json::json!("answer"));
    }

    #[test]
    fn turn_finished_echo_completes_the_turn() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let p1 = drive_turn_done(&mut agg, "t1", serde_json::json!("answer"));
        let tf = turn_finished_decision(&p1).expect("turn.finished queued");

        let p2 = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: tf,
                transcript: vec![],
                actions: vec![Action::Done {
                    data: serde_json::Value::Null,
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );

        // Pass 2 is the run terminal: exactly one TurnCompleted (the frozen output,
        // no error) then SessionDone.
        let tc = turn_completed(&p2).expect("pass 2 completes the turn");
        assert_eq!(tc.turn_id, "t1");
        assert_eq!(
            tc.data,
            serde_json::json!("answer"),
            "frozen output survives"
        );
        assert!(tc.error.is_none(), "a clean finalize is not an error");
        assert_eq!(
            p2.iter()
                .filter(|e| matches!(e, EventPayload::TurnCompleted(_)))
                .count(),
            1,
            "exactly one TurnCompleted; got {p2:?}"
        );
        assert!(has_session_done(&p2), "SessionDone follows; got {p2:?}");
        assert!(
            turn_finished_decision(&p2).is_none(),
            "no new turn.finished queued; got {p2:?}"
        );
        assert_eq!(agg.state.completed_turn_ids.len(), 1);
        assert!(agg.state.finalizing.is_none(), "finalizing cleared");
    }

    #[test]
    fn turn_finished_worker_runs_side_effect_before_completion() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let p1 = drive_turn_done(&mut agg, "t1", serde_json::json!("answer"));
        let tf = turn_finished_decision(&p1).expect("turn.finished queued");

        let p2 = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: tf,
                transcript: vec![],
                actions: vec![
                    call_llm_action("side-1", LlmHandler::Server),
                    Action::Done {
                        data: serde_json::Value::Null,
                    },
                ],
                state: None,
                agent: None,
            },
            &machine(),
        );

        assert!(
            p2.iter().any(|e| matches!(
                e,
                EventPayload::LlmCallRequested(r) if r.call_id == "side-1"
            )),
            "the side effect dispatches; got {p2:?}"
        );
        assert!(
            turn_completed(&p2).is_some_and(|tc| tc.error.is_none()),
            "the turn completes after the worker's own done; got {p2:?}"
        );
        assert!(has_session_done(&p2));
    }

    #[test]
    fn no_turn_completes_immediately() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let events = submit_decision_with(
            &mut agg,
            vec![Action::Done {
                data: serde_json::Value::Null,
            }],
        );
        assert!(
            has_session_done(&events),
            "a turn-less session goes straight to SessionDone; got {events:?}"
        );
        assert!(
            turn_finished_decision(&events).is_none(),
            "no turn.finished without a turn; got {events:?}"
        );
        assert!(
            turn_completed(&events).is_none(),
            "no TurnCompleted without a turn; got {events:?}"
        );
    }

    #[test]
    fn turn_finished_terminal_failure_completes_as_failed_run() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1"); // no-retry
        let p1 = drive_turn_done(&mut agg, "t1", serde_json::json!("answer"));
        let tf = turn_finished_decision(&p1).expect("turn.finished queued");

        let events = dispatch(
            &mut agg,
            CommandPayload::FailWorkerDecision {
                decision_id: tf.clone(),
                error: "worker crashed".to_string(),
                retryable: false,
            },
            &machine(),
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionErrored(_))),
            "the finalizer errors; got {events:?}"
        );
        // The run still terminates — but as a failed run carrying the error, with the
        // output preserved.
        let tc = turn_completed(&events).expect("a failed finalizer still completes the turn");
        assert_eq!(tc.turn_id, "t1");
        assert_eq!(tc.data, serde_json::json!("answer"), "output stays durable");
        assert_eq!(tc.error.as_deref(), Some("worker crashed"));
        assert!(has_session_done(&events));
        assert!(!agg.state.worker_decisions.contains_key(&tf));
        assert_eq!(agg.state.completed_turn_ids.len(), 1);
        assert!(agg.state.finalizing.is_none(), "finalizing cleared");
    }

    #[test]
    fn turn_finished_retryable_failure_does_not_complete() {
        let mut agg = create_session_with_retry(RetryPolicy {
            timeout_secs: None,
            max_retries: 2,
            backoff_base_secs: 1,
            backoff_max_secs: 1,
        });
        let p1 = drive_turn_done(&mut agg, "t1", serde_json::json!("answer"));
        let tf = turn_finished_decision(&p1).expect("turn.finished queued");

        let events = dispatch(
            &mut agg,
            CommandPayload::FailWorkerDecision {
                decision_id: tf.clone(),
                error: "transient".to_string(),
                retryable: true,
            },
            &machine(),
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionErrored(_))),
            "the failure is recorded; got {events:?}"
        );
        assert!(
            turn_completed(&events).is_none() && !has_session_done(&events),
            "a retryable failure neither completes nor settles; got {events:?}"
        );
        assert_eq!(
            agg.state
                .worker_decisions
                .get(&tf)
                .map(|d| d.tracking.status.clone()),
            Some(EffectStatus::RetryScheduled),
            "the finalizer is rescheduled for redelivery"
        );
        assert!(
            agg.state.finalizing.is_some(),
            "still finalizing pending redelivery"
        );
    }

    #[test]
    fn turn_finished_deadline_completes_when_exhausted() {
        let mut agg = create_session_with_retry(RetryPolicy {
            timeout_secs: Some(60),
            max_retries: 0,
            backoff_base_secs: 0,
            backoff_max_secs: 0,
        });
        let p1 = drive_turn_done(&mut agg, "t1", serde_json::json!("answer"));
        let tf = turn_finished_decision(&p1).expect("turn.finished queued");

        let events = dispatch(
            &mut agg,
            CommandPayload::Wake {
                now: Utc::now() + chrono::Duration::hours(1),
            },
            &system(),
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionErrored(p) if p.decision_id == tf
            )),
            "the timed-out finalizer errors; got {events:?}"
        );
        let tc = turn_completed(&events).expect("a terminal timeout completes the turn");
        assert_eq!(tc.error.as_deref(), Some("deadline exceeded"));
        assert!(has_session_done(&events));
    }

    // ── Branch-scoped worker state ───────────────────────────────────────

    use serde_json::json;

    fn open_decision(agg: &mut SessionAggregate, text: &str) -> String {
        let setup = dispatch(
            agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, text),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        decision_with(&setup, |t| matches!(t, Trigger::ClientMessage { .. }))
            .expect("user message opens a decision")
    }

    fn submit_state(
        agg: &mut SessionAggregate,
        decision_id: String,
        transcript: Vec<DraftMessage>,
        state: Option<serde_json::Value>,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript,
                actions: vec![],
                state: state.map(WorkerState::from),
                agent: None,
            },
            &machine(),
        )
    }

    fn state_updates(events: &[EventPayload]) -> Vec<&WorkerStateUpdated> {
        events
            .iter()
            .filter_map(|e| match e {
                EventPayload::WorkerStateUpdated(p) => Some(p),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn echoed_state_writes_nothing() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        let events = submit_state(
            &mut agg,
            d1,
            vec![node_msg("u1", Role::User, "hi")],
            Some(json!({"a": 1, "b": 2})),
        );
        assert_eq!(
            state_updates(&events).len(),
            1,
            "the first write records a version; got {events:?}"
        );

        // Echo with shuffled key order: structural equality dedups it.
        let d2 = open_decision(&mut agg, "again");
        let events = submit_state(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("u2", Role::User, "again"),
            ],
            Some(json!({"b": 2, "a": 1})),
        );
        assert!(
            state_updates(&events).is_empty(),
            "an echoed state writes nothing; got {events:?}"
        );
    }

    #[test]
    fn null_valued_key_differs_from_absent_key() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![node_msg("u1", Role::User, "hi")],
            Some(json!({})),
        );
        let d2 = open_decision(&mut agg, "again");
        let events = submit_state(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("u2", Role::User, "again"),
            ],
            Some(json!({"a": null})),
        );
        assert_eq!(
            state_updates(&events).len(),
            1,
            "null is not absence; got {events:?}"
        );
    }

    #[test]
    fn changed_state_anchors_at_the_post_reconcile_head() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        let events = submit_state(
            &mut agg,
            d1,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            Some(json!({"v": 1})),
        );
        let updates = state_updates(&events);
        assert_eq!(updates.len(), 1);
        assert_eq!(
            updates[0].anchor.as_deref(),
            Some("a1"),
            "the version anchors to the last appended node"
        );
        assert_eq!(
            agg.state.resolve_state_for(agg.state.head_id.as_deref()).0,
            json!({"v": 1})
        );
    }

    #[test]
    fn omitted_state_keeps_the_current_version() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![node_msg("u1", Role::User, "hi")],
            Some(json!({"v": 1})),
        );
        let d2 = open_decision(&mut agg, "more");
        let events = submit_state(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("u2", Role::User, "more"),
            ],
            None,
        );
        assert!(state_updates(&events).is_empty());
        assert_eq!(
            agg.state.resolve_state_for(agg.state.head_id.as_deref()).0,
            json!({"v": 1})
        );
    }

    fn agent_updates(events: &[EventPayload]) -> Vec<&AgentConfigUpdated> {
        events
            .iter()
            .filter_map(|e| match e {
                EventPayload::AgentConfigUpdated(p) => Some(p),
                _ => None,
            })
            .collect()
    }

    fn agent_config(model: &str) -> AgentConfig {
        AgentConfig {
            format: None,
            model: model.to_string(),
            system: None,
            stream: true,
            handler: None,
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
            mcp: Vec::new(),
        }
    }

    fn submit_agent(
        agg: &mut SessionAggregate,
        decision_id: String,
        transcript: Vec<DraftMessage>,
        agent: Option<AgentConfig>,
    ) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id,
                transcript,
                actions: vec![],
                state: None,
                agent,
            },
            &machine(),
        )
    }

    // ── Connectors ───────────────────────────────────────────────────────

    fn connector_config(ids: &[&str]) -> AgentConfig {
        AgentConfig {
            mcp: ids
                .iter()
                .map(|id| McpServer {
                    id: id.to_string(),
                    tools: None,
                })
                .collect(),
            ..agent_config("m1")
        }
    }

    fn remote_tool(name: &str) -> RemoteTool {
        RemoteTool {
            name: name.to_string(),
            description: "a remote tool".to_string(),
            input: None,
            output: None,
            annotations: Default::default(),
        }
    }

    fn sync_requests(events: &[EventPayload]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                EventPayload::ConnectorSyncRequested(p) => Some(p.connection_id.as_str()),
                _ => None,
            })
            .collect()
    }

    fn promotions(events: &[EventPayload]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.as_str()),
                _ => None,
            })
            .collect()
    }

    fn settle_sync(agg: &mut SessionAggregate, id: &str, tools: &[&str]) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::CompleteConnectorSync {
                connection_id: id.to_string(),
                attempt: None,
                prefix: Some(id.to_string()),
                tools: tools.iter().map(|t| remote_tool(t)).collect(),
            },
            &system(),
        )
    }

    /// A session whose config names `ids`, with every fetch settled and each
    /// connection offering `tools`.
    fn session_with_connectors(ids: &[&str], tools: &[&str]) -> SessionAggregate {
        let mut agg =
            create_session_with_config("sess-1", "tenant-a", "user-1", Some(connector_config(ids)));
        for id in ids {
            settle_sync(&mut agg, id, tools);
        }
        agg
    }

    #[test]
    fn declaring_a_connector_fetches_it_and_parks_the_turn() {
        let mut agg = create_session_with_config(
            "sess-1",
            "tenant-a",
            "user-1",
            Some(connector_config(&["sentry"])),
        );
        assert!(
            agg.state.connector_syncs.contains_key("sentry"),
            "the config write fetches the connection it names"
        );

        // A message now cannot be decided: the config names tools the engine
        // has not fetched, so the turn cannot be authored against it.
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        assert!(
            promotions(&events).is_empty(),
            "a decision parks behind an unsettled fetch; got {events:?}"
        );
        assert_eq!(
            agg.state.queued_decisions().len(),
            1,
            "the decision is queued, not lost"
        );

        let events = settle_sync(&mut agg, "sentry", &["search_issues"]);
        assert_eq!(
            promotions(&events).len(),
            1,
            "settling the fetch releases the parked decision; got {events:?}"
        );
    }

    #[test]
    fn work_started_beside_a_new_connector_queues_its_decision_rather_than_running_it() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d = open_decision(&mut agg, "hi");

        // One decision that both declares a connector and starts a worker tool
        // call. The tool call's `tool.execute` must not go out: the fetch this
        // batch just requested has not settled.
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d,
                transcript: vec![node_msg("u1", Role::User, "hi")],
                actions: vec![Action::CallTool {
                    id: "tc-1".to_string(),
                    name: "get_time".to_string(),
                    arguments: "{}".to_string(),
                    retry: RetryPolicy::no_retry(),
                }],
                state: None,
                agent: Some(connector_config(&["sentry"])),
            },
            &machine(),
        );

        assert_eq!(sync_requests(&events), ["sentry"]);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(_))),
            "the tool.execute decision is created; got {events:?}"
        );
        assert!(
            promotions(&events).is_empty(),
            "but not promoted while the fetch is in flight; got {events:?}"
        );
        assert_eq!(
            agg.state.queued_decisions().len(),
            1,
            "it waits in the queue"
        );

        let events = settle_sync(&mut agg, "sentry", &["search_issues"]);
        assert_eq!(
            promotions(&events).len(),
            1,
            "and runs once the offer lands; got {events:?}"
        );
    }

    #[test]
    fn resuming_an_interrupt_still_waits_on_an_unsettled_fetch() {
        let mut agg = create_session_with_config(
            "sess-1",
            "tenant-a",
            "user-1",
            Some(connector_config(&["sentry"])),
        );
        dispatch(
            &mut agg,
            CommandPayload::Interrupt {
                interrupt_id: "int-1".to_string(),
                reason: "hold".to_string(),
                payload: serde_json::json!({}),
            },
            &machine(),
        );

        // Resume promotes directly rather than through the usual gate, because
        // that gate would park on the interrupt this batch just cleared. The
        // fetch still has to hold it.
        let events = dispatch(
            &mut agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: "int-1".to_string(),
                payload: serde_json::json!({}),
            },
            &machine(),
        );
        assert!(
            promotions(&events).is_empty(),
            "an interrupt clearing does not release a turn the fetch still holds; got {events:?}"
        );

        let events = settle_sync(&mut agg, "sentry", &["search_issues"]);
        assert_eq!(
            promotions(&events).len(),
            1,
            "the fetch releases it; got {events:?}"
        );
    }

    #[test]
    fn a_fetch_is_keyed_on_the_connection_not_the_agent_version() {
        let mut agg = session_with_connectors(&["sentry"], &["search_issues"]);

        // A config rewritten for an unrelated reason — a new declared tool —
        // must not cost another round trip.
        let mut rewritten = connector_config(&["sentry"]);
        rewritten.tools.push(AgentTool {
            name: "get_time".to_string(),
            description: String::new(),
            input: None,
            output: None,
            handler: None,
        });
        let d = open_decision(&mut agg, "hi");
        let events = submit_agent(
            &mut agg,
            d,
            vec![node_msg("u1", Role::User, "hi")],
            Some(rewritten),
        );
        assert!(
            sync_requests(&events).is_empty(),
            "an unrelated config rewrite refetches nothing; got {events:?}"
        );

        // Adding a connection fetches only the new one.
        let d = open_decision(&mut agg, "again");
        let events = submit_agent(
            &mut agg,
            d,
            vec![node_msg("u2", Role::User, "again")],
            Some(connector_config(&["sentry", "github"])),
        );
        assert_eq!(
            sync_requests(&events),
            ["github"],
            "only the connection that was never fetched; got {events:?}"
        );
    }

    #[test]
    fn a_terminally_failed_fetch_releases_the_turn_rather_than_parking_it() {
        let mut agg = create_session_with_config(
            "sess-1",
            "tenant-a",
            "user-1",
            Some(connector_config(&["sentry"])),
        );
        dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::FailConnectorSync {
                connection_id: "sentry".to_string(),
                attempt: None,
                error: "connection refused".to_string(),
                retryable: false,
                needs_reauth: false,
            },
            &system(),
        );
        assert_eq!(
            promotions(&events).len(),
            1,
            "an unreachable connector unblocks the worker to decide; got {events:?}"
        );
        assert!(
            !agg.state
                .has_pending_connector_sync(agg.state.head_id.as_deref()),
            "a terminal failure is settled, so it parks nothing further"
        );
        assert!(
            agg.state
                .connector_tools(agg.state.head_id.as_deref())
                .tools
                .is_empty(),
            "a failed fetch contributes no tools"
        );
    }

    #[test]
    fn a_retryable_failure_keeps_parking_until_it_is_exhausted() {
        let mut agg = create_session_with_config(
            "sess-1",
            "tenant-a",
            "user-1",
            Some(connector_config(&["sentry"])),
        );
        let events = dispatch(
            &mut agg,
            CommandPayload::FailConnectorSync {
                connection_id: "sentry".to_string(),
                attempt: None,
                error: "503".to_string(),
                retryable: true,
                needs_reauth: false,
            },
            &system(),
        );
        assert!(
            promotions(&events).is_empty(),
            "a retry is still unsettled, so it still parks; got {events:?}"
        );
        assert!(agg
            .state
            .has_pending_connector_sync(agg.state.head_id.as_deref()));
        assert!(
            agg.state.wake_at().is_some(),
            "the retry is scheduled, so the session wakes for it"
        );
    }

    #[test]
    fn a_hung_fetch_times_out_rather_than_parking_the_session_forever() {
        let agg = create_session_with_config(
            "sess-1",
            "tenant-a",
            "user-1",
            Some(connector_config(&["sentry"])),
        );
        let deadline = agg.state.connector_syncs["sentry"]
            .tracking
            .deadline
            .expect("a fetch is bounded");
        let events = agg
            .state
            .handle_wake(deadline + chrono::Duration::seconds(1))
            .expect("wake");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ConnectorSyncErrored(_))),
            "a fetch past its deadline fails; got {events:?}"
        );
    }

    #[test]
    fn connector_tools_reach_the_model_and_route_to_the_engine() {
        let mut agg = session_with_connectors(&["sentry"], &["search_issues"]);

        assert_eq!(
            agg.state.tool_handler_for("sentry__search_issues"),
            ToolHandler::Server,
            "a connector-resolved name runs on the engine"
        );
        assert_eq!(
            agg.state.tool_handler_for("something_else"),
            ToolHandler::Worker,
            "an undeclared name still gets its contract error on the worker"
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "call-1".to_string(),
                request: LlmRequest {
                    model: "m1".to_string(),
                    messages: vec![],
                    tools: Some(vec![]),
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
                format: None,
            },
            &system(),
        );
        let offered: Vec<String> = events
            .iter()
            .find_map(|e| match e {
                EventPayload::LlmCallRequested(p) => p.request.tools.clone(),
                _ => None,
            })
            .expect("an llm call")
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(
            offered,
            ["sentry__search_issues"],
            "the engine adds the connector's tools, which no worker could name"
        );

        // A connector call is the engine's to run: no worker execute trigger.
        let events = dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "sentry__search_issues".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &system(),
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(_))),
            "the engine executes it; the worker is not asked to; got {events:?}"
        );
        assert_eq!(
            agg.state.tool_calls["tc-1"].handler,
            ToolHandler::Server,
            "the handler is frozen onto the call"
        );
    }

    #[test]
    fn a_re_prompt_does_not_offer_a_connector_tool_twice() {
        let mut agg = session_with_connectors(&["sentry"], &["search_issues"]);

        // A re-prompt is authored from the previous call's stored spec, which
        // already carries the connector's tools. Adding them again is a 400
        // from every provider: tool names must be unique.
        let already = LlmRequest {
            model: "m1".to_string(),
            messages: vec![],
            tools: Some(vec![LlmTool {
                name: "sentry__search_issues".to_string(),
                description: "a remote tool".to_string(),
                input: None,
                output: None,
            }]),
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        };
        let events = dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                call_id: "call-1".to_string(),
                request: already,
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
                format: None,
            },
            &system(),
        );
        let names: Vec<String> = events
            .iter()
            .find_map(|e| match e {
                EventPayload::LlmCallRequested(p) => p.request.tools.clone(),
                _ => None,
            })
            .expect("an llm call")
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, ["sentry__search_issues"], "offered once, not twice");
    }

    #[test]
    fn a_declared_tool_keeps_its_name_and_its_handler_against_a_connector() {
        let mut config = connector_config(&["sentry"]);
        config.tools.push(AgentTool {
            name: "sentry__search_issues".to_string(),
            description: String::new(),
            input: None,
            output: None,
            handler: None,
        });
        let mut agg = create_session_with_config("sess-1", "tenant-a", "user-1", Some(config));
        settle_sync(&mut agg, "sentry", &["search_issues"]);

        assert_eq!(
            agg.state.tool_handler_for("sentry__search_issues"),
            ToolHandler::Worker,
            "the config claims the name, so the connector never takes it"
        );
        let merged = agg.state.connector_tools(agg.state.head_id.as_deref());
        assert!(merged.tools.is_empty());
        assert_eq!(
            merged.collisions,
            ["sentry__search_issues"],
            "and the drop is reported rather than silent"
        );
    }

    #[test]
    fn a_filter_change_re_derives_without_another_fetch() {
        let mut agg = session_with_connectors(&["sentry"], &["search_issues", "create_issue"]);
        assert_eq!(
            agg.state
                .connector_tools(agg.state.head_id.as_deref())
                .tools
                .len(),
            2
        );

        let narrowed = AgentConfig {
            mcp: vec![McpServer {
                id: "sentry".to_string(),
                tools: Some(McpTools {
                    include: vec!["search_*".to_string()],
                    ..Default::default()
                }),
            }],
            ..agent_config("m1")
        };
        let d = open_decision(&mut agg, "hi");
        let events = submit_agent(
            &mut agg,
            d,
            vec![node_msg("u1", Role::User, "hi")],
            Some(narrowed),
        );
        assert!(
            sync_requests(&events).is_empty(),
            "filtering is pure, so narrowing costs no round trip; got {events:?}"
        );
        let names: Vec<String> = agg
            .state
            .connector_tools(agg.state.head_id.as_deref())
            .tools
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, ["sentry__search_issues"], "the offer re-filters");
    }

    #[test]
    fn a_fork_keeps_the_offer_it_already_fetched() {
        let agg = session_with_connectors(&["sentry"], &["search_issues"]);

        // Rewind past the fetch: an offer is a fact about the remote, not about
        // a branch, so it survives the way the call maps do.
        let rewound = agg.state.clone().rewind(0, None);
        assert!(
            rewound.connector_syncs["sentry"].is_ready(),
            "a fork refetches nothing"
        );
    }

    #[test]
    fn changed_agent_config_anchors_at_head_and_dedups() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");

        let d1 = open_decision(&mut agg, "hi");
        let events = submit_agent(
            &mut agg,
            d1,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            Some(agent_config("m1")),
        );
        let updates = agent_updates(&events);
        assert_eq!(updates.len(), 1, "the first write records a config version");
        assert_eq!(
            updates[0].anchor.as_deref(),
            Some("a1"),
            "the config anchors to the last appended node"
        );
        assert_eq!(
            agg.state.resolve_agent_for(agg.state.head_id.as_deref()),
            Some(agent_config("m1"))
        );

        // Echo the same config: structural equality dedups it.
        let d2 = open_decision(&mut agg, "again");
        let events = submit_agent(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
                node_msg("u2", Role::User, "again"),
            ],
            Some(agent_config("m1")),
        );
        assert!(
            agent_updates(&events).is_empty(),
            "an echoed config writes nothing; got {events:?}"
        );
    }

    #[test]
    fn omitted_agent_keeps_the_current_config() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_agent(
            &mut agg,
            d1,
            vec![node_msg("u1", Role::User, "hi")],
            Some(agent_config("m1")),
        );
        let d2 = open_decision(&mut agg, "more");
        let events = submit_agent(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("u2", Role::User, "more"),
            ],
            None,
        );
        assert!(agent_updates(&events).is_empty());
        assert_eq!(
            agg.state.resolve_agent_for(agg.state.head_id.as_deref()),
            Some(agent_config("m1"))
        );
    }

    #[test]
    fn create_session_emits_session_start_before_client_input() {
        let mut agg = SessionAggregate::new(
            "sess-1".to_string(),
            "tenant-a".to_string(),
            SessionState::new("sess-1".to_string()),
        );
        let events = dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: "tenant-a".to_string(),
                    id: Some("user-1".to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
            },
            &system(),
        );
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::SessionCreated(_),
                    EventPayload::DecisionRequestQueued(q),
                    EventPayload::WorkerDecisionRequested(_),
                ] if matches!(q.trigger, Trigger::SessionStart)
            ),
            "CreateSession opens session.start as the first decision; got {events:?}"
        );
    }

    #[test]
    fn session_start_config_is_visible_to_a_queued_client_decision() {
        let mut agg = SessionAggregate::new(
            "sess-1".to_string(),
            "tenant-a".to_string(),
            SessionState::new("sess-1".to_string()),
        );
        dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: "tenant-a".to_string(),
                    id: Some("user-1".to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
            },
            &system(),
        );

        // A client message arrives before session.start completes: it queues.
        let setup = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        assert!(
            setup
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(_))),
            "the client decision queues behind session.start; got {setup:?}"
        );

        // The worker declares its config on session.start.
        let start = agg
            .state
            .worker_decisions
            .values()
            .find(|d| matches!(d.trigger, Trigger::SessionStart))
            .map(|d| d.decision_id.clone())
            .expect("a pending session.start decision");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: start,
                transcript: vec![],
                actions: vec![],
                state: None,
                agent: Some(agent_config("m1")),
            },
            &machine(),
        );

        // Completing session.start promotes the queued client decision, whose
        // derived snapshot now resolves the just-written config.
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(w)
                    if matches!(
                        agg.state.worker_decisions.get(&w.decision_id).map(|d| &d.trigger),
                        Some(Trigger::ClientMessage { .. })
                    )
            )),
            "the queued client decision is promoted; got {events:?}"
        );
        assert_eq!(
            agg.state.resolve_agent_for(agg.state.head_id.as_deref()),
            Some(agent_config("m1"))
        );
    }

    /// `session.start` is unsettled while RetryScheduled, not just while
    /// Pending: a client message arriving between failure and retry must park
    /// behind it, or the turn runs configless and the retry is starved by the
    /// now-live client decision.
    #[test]
    fn client_message_parks_while_session_start_retry_is_scheduled() {
        let mut agg = SessionAggregate::new(
            "sess-1".to_string(),
            "tenant-a".to_string(),
            SessionState::new("sess-1".to_string()),
        );
        let created = dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: "tenant-a".to_string(),
                    id: Some("user-1".to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy {
                    timeout_secs: None,
                    max_retries: 2,
                    backoff_base_secs: 1,
                    backoff_max_secs: 1,
                },
            },
            &system(),
        );
        let start = created
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(w) => Some(w.decision_id.clone()),
                _ => None,
            })
            .expect("CreateSession opens a session.start decision");

        dispatch(
            &mut agg,
            CommandPayload::FailWorkerDecision {
                decision_id: start.clone(),
                error: "transient".to_string(),
                retryable: true,
            },
            &machine(),
        );
        assert_eq!(
            agg.state
                .worker_decisions
                .get(&start)
                .map(|d| d.tracking.status.clone()),
            Some(EffectStatus::RetryScheduled),
            "session.start is rescheduled, not settled"
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(_))),
            "the client decision queues; got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))),
            "the client decision parks behind the scheduled session.start retry, \
             exactly as it does while session.start is Pending; got {events:?}"
        );

        // The due retry re-delivers session.start ahead of the queued client
        // decision — the reverse order starves the retry behind live traffic.
        let events = dispatch(
            &mut agg,
            CommandPayload::Wake {
                now: Utc::now() + chrono::Duration::hours(1),
            },
            &system(),
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(w) if w.decision_id == start
            )),
            "the wake re-delivers session.start first; got {events:?}"
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: start,
                transcript: vec![],
                actions: vec![],
                state: None,
                agent: Some(agent_config("m1")),
            },
            &machine(),
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(w)
                    if matches!(
                        agg.state.worker_decisions.get(&w.decision_id).map(|d| &d.trigger),
                        Some(Trigger::ClientMessage { .. })
                    )
            )),
            "the queued client decision is promoted; got {events:?}"
        );
        assert_eq!(
            agg.state.resolve_agent_for(agg.state.head_id.as_deref()),
            Some(agent_config("m1"))
        );
    }

    /// A terminally-failed `session.start` leaves the session unable to ever
    /// configure a turn. Parked work must not be promoted into a configless
    /// turn that settles as a silent no-op, and new work must be refused with
    /// a typed error, not swallowed. Recovery is the retry policy's job.
    #[test]
    fn terminal_session_start_failure_restarts_on_the_next_message() {
        let mut agg = SessionAggregate::new(
            "sess-1".to_string(),
            "tenant-a".to_string(),
            SessionState::new("sess-1".to_string()),
        );
        let created = dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: "tenant-a".to_string(),
                    id: Some("user-1".to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
            },
            &system(),
        );
        let start = created
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(w) => Some(w.decision_id.clone()),
                _ => None,
            })
            .expect("CreateSession opens a session.start decision");

        // A message arrives while session.start is pending: it queues behind it.
        let queued = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hi"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        let queued_id = decision_with(&queued, |t| matches!(t, Trigger::ClientMessage { .. }))
            .expect("the client decision queues behind session.start");

        let events = dispatch(
            &mut agg,
            CommandPayload::FailWorkerDecision {
                decision_id: start,
                error: "worker crashed".to_string(),
                retryable: false,
            },
            &machine(),
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionErrored(_))),
            "the failure is recorded; got {events:?}"
        );
        // No turn was ever started, so there is no run to end — the error event
        // is the whole record. A TurnCompleted here would invent a turn.
        assert!(
            turn_completed(&events).is_none(),
            "no turn to complete without one started; got {events:?}"
        );

        // The queued decision can never resolve a config: promoting it runs the
        // turn configless and settles it as a silent no-op.
        let events = wake(&mut agg);
        assert!(
            !events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(w) if w.decision_id == queued_id
            )),
            "a queued client decision must not be promoted after session.start \
             failed terminally; got {events:?}"
        );

        // The next user message retries the start rather than bouncing: fix the
        // worker, say something, and the session picks up where it died.
        let retried = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: node_msg("", Role::User, "hello again"),
                    stream: false,
                }),
                turn_id: None,
            },
            &system(),
        );
        let restart = decision_with(&retried, |t| matches!(t, Trigger::SessionStart))
            .expect("a new user message re-queues session.start");
        let follow_up = decision_with(&retried, |t| matches!(t, Trigger::ClientMessage { .. }))
            .expect("the message queues too");
        assert!(
            retried.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(w) if w.decision_id == restart
            )),
            "the restart is promoted; got {retried:?}"
        );
        assert!(
            !retried.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(w) if w.decision_id == follow_up
            )),
            "the message parks behind the restart; got {retried:?}"
        );

        // The restart lands a config, and the message runs against it.
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: restart,
                transcript: vec![],
                actions: vec![],
                state: None,
                agent: Some(agent_config("m1")),
            },
            &machine(),
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(w) if w.decision_id == follow_up
            )),
            "the recovered session promotes the waiting message; got {events:?}"
        );
        assert!(
            !agg.state.session_start_failed,
            "the session is no longer poisoned"
        );
    }

    #[test]
    fn fork_anchors_new_state_and_resolves_as_of_the_prefix_without_one() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![node_msg("u1", Role::User, "hi")],
            Some(json!({"v": 1})),
        );
        let d2 = open_decision(&mut agg, "more");
        submit_state(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
                node_msg("u2", Role::User, "more"),
            ],
            Some(json!({"v": 2})),
        );
        assert_eq!(
            agg.state.resolve_state_for(agg.state.head_id.as_deref()).0,
            json!({"v": 2})
        );

        let d3 = open_decision(&mut agg, "redo");
        let events = submit_state(
            &mut agg,
            d3,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("x1", Role::User, "redo"),
            ],
            Some(json!({"v": 3})),
        );
        let updates = state_updates(&events);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].anchor.as_deref(), Some("x1"));
        assert_eq!(
            agg.state.resolve_state_for(agg.state.head_id.as_deref()).0,
            json!({"v": 3})
        );

        // Fork with no state opinion resolves as-of the fork point.
        let d4 = open_decision(&mut agg, "retry");
        let events = submit_state(
            &mut agg,
            d4,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("y1", Role::User, "retry"),
            ],
            None,
        );
        assert!(state_updates(&events).is_empty());
        assert_eq!(agg.state.head_id.as_deref(), Some("y1"));
        assert_eq!(
            agg.state.resolve_state_for(agg.state.head_id.as_deref()).0,
            json!({"v": 1}),
            "the fork is uncontaminated by the abandoned branches"
        );
    }

    #[test]
    fn effect_anchor_is_the_post_reconcile_head() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![node_msg("u1", Role::User, "hi")],
                actions: vec![Action::CallTool {
                    id: "t1".to_string(),
                    name: "slow".to_string(),
                    arguments: "{}".to_string(),
                    retry: RetryPolicy::no_retry(),
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );
        assert_eq!(
            agg.state.tool_calls.get("t1").unwrap().anchor.as_deref(),
            Some("u1"),
            "the anchor is the head after this submit's appends"
        );
    }

    fn head_moves(events: &[EventPayload]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                EventPayload::HeadMoved(h) => Some(h.head_id.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn truncating_view_moves_head_and_forks_the_regenerated_reply() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            None,
        );
        assert_eq!(agg.state.head_id.as_deref(), Some("a1"));

        // Regenerate: the view stops at u1 — nothing to write, head rebases.
        let d2 = open_decision(&mut agg, "regen");
        let events = submit_state(&mut agg, d2, vec![node_msg("u1", Role::User, "hi")], None);
        assert_eq!(head_moves(&events), ["u1"], "got {events:?}");
        assert_eq!(agg.state.head_id.as_deref(), Some("u1"));

        // The regenerated reply forks: a sibling of a1, not its child.
        let d3 = open_decision(&mut agg, "next");
        submit_state(
            &mut agg,
            d3,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a2", Role::Assistant, "hello again"),
            ],
            None,
        );
        assert_eq!(agg.state.head_id.as_deref(), Some("a2"));
        let tree = agg.state.message_tree();
        let a2 = tree.nodes.iter().find(|n| n.message.id == "a2").unwrap();
        assert_eq!(
            a2.parent_id.as_deref(),
            Some("u1"),
            "forks at the truncation point"
        );
    }

    #[test]
    fn full_resend_does_not_move_the_head() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            None,
        );
        let d2 = open_decision(&mut agg, "again");
        let events = submit_state(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            None,
        );
        assert!(head_moves(&events).is_empty(), "got {events:?}");
        assert_eq!(agg.state.head_id.as_deref(), Some("a1"));
    }

    #[test]
    fn viewless_decision_keeps_the_head() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        let d2 = open_decision(&mut agg, "more");
        let events = submit_state(&mut agg, d2, vec![], None);
        assert!(head_moves(&events).is_empty(), "got {events:?}");
        assert_eq!(agg.state.head_id.as_deref(), Some("u1"));
    }

    #[test]
    fn known_branch_view_switches_the_head() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            None,
        );
        // Fork to a second branch via an edit.
        let d2 = open_decision(&mut agg, "edit");
        submit_state(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("b1", Role::User, "hi, edited"),
            ],
            None,
        );
        assert_eq!(agg.state.head_id.as_deref(), Some("b1"));

        // Submitting the first branch's view switches back to it.
        let d3 = open_decision(&mut agg, "switch");
        let events = submit_state(
            &mut agg,
            d3,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            None,
        );
        assert_eq!(head_moves(&events), ["a1"], "got {events:?}");
        assert_eq!(agg.state.head_id.as_deref(), Some("a1"));
    }

    #[test]
    fn truncation_voids_work_anchored_on_the_abandoned_branch() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            None,
        );
        request_client_tool(&mut agg, "tc-1"); // anchored at a1

        let d2 = open_decision(&mut agg, "regen");
        let events = submit_state(&mut agg, d2, vec![node_msg("u1", Role::User, "hi")], None);
        assert_eq!(head_moves(&events), ["u1"]);
        assert_eq!(voided_ids(&events), ["tc-1"], "got {events:?}");
    }

    fn settle_decisions(events: &[EventPayload]) -> Vec<&Trigger> {
        events
            .iter()
            .filter_map(|e| {
                let trigger = match e {
                    EventPayload::DecisionRequestQueued(p) => &p.trigger,
                    _ => return None,
                };
                matches!(
                    trigger,
                    Trigger::ToolFinished { .. }
                        | Trigger::SubAgentFinished { .. }
                        | Trigger::LlmFinished { .. }
                )
                .then_some(trigger)
            })
            .collect()
    }

    #[test]
    fn settle_without_attempt_settles_the_current_attempt() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        request_client_tool(&mut agg, "tc-1");
        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: None,
                result: "ok".to_string(),
            },
            &machine(),
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "an attempt-less settle lands on the current attempt; got {events:?}"
        );

        // A supplied attempt still fences.
        request_client_tool(&mut agg, "tc-2");
        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-2".to_string(),
                    attempt: Some(7),
                    result: "ok".to_string(),
                },
                &machine(),
            )
            .expect_err("mismatched attempt is fenced");
        assert!(matches!(err, SessionError::EffectAttemptMismatch));
    }

    fn voided_ids(events: &[EventPayload]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                EventPayload::CallVoided(v) => Some(v.id.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn fork_voids_a_pending_tool_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        request_client_tool(&mut agg, "tc-1"); // anchored at u1

        let d2 = open_decision(&mut agg, "redo");
        let events = submit_state(&mut agg, d2, vec![node_msg("x1", Role::User, "redo")], None);
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::CallVoided(v)
                    if v.kind == EffectKind::ToolCall && v.id == "tc-1"
            )),
            "the fork voids the stranded call; got {events:?}"
        );

        // A late settle is rejected: the effect died with its branch.
        let err = agg
            .state
            .handle(
                CommandPayload::CompleteToolCall {
                    tool_call_id: "tc-1".to_string(),
                    attempt: Some(0),
                    result: "ok".to_string(),
                },
                &machine(),
            )
            .expect_err("settling voided work is an error");
        assert!(matches!(err, SessionError::EffectNotPending));
    }

    #[test]
    fn fork_spares_work_anchored_on_the_shared_prefix() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        request_client_tool(&mut agg, "tc-1"); // anchored at u1

        // Head advances past the anchor then forks below it; u1 stays on-path.
        let d2 = open_decision(&mut agg, "more");
        submit_state(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "?"),
            ],
            None,
        );
        let d3 = open_decision(&mut agg, "redo");
        let events = submit_state(
            &mut agg,
            d3,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("b1", Role::Assistant, "!"),
            ],
            None,
        );
        assert!(
            voided_ids(&events).is_empty(),
            "work above the fork point is untouched; got {events:?}"
        );

        let events = complete_tool(&mut agg, "tc-1", "ok");
        assert_eq!(fired_tool_result(&events), vec!["tc-1".to_string()]);
    }

    #[test]
    fn fork_voids_a_retrying_effect() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        declare_client_tool(&mut agg, "flaky");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-1".to_string(),
                name: "flaky".to_string(),
                arguments: "{}".to_string(),
                retry: RetryPolicy {
                    timeout_secs: None,
                    max_retries: 2,
                    backoff_base_secs: 1,
                    backoff_max_secs: 1,
                },
            },
            &system(),
        );
        dispatch(
            &mut agg,
            CommandPayload::FailToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                error: "flake".to_string(),
                retryable: true,
            },
            &machine(),
        );

        let d2 = open_decision(&mut agg, "redo");
        let events = submit_state(&mut agg, d2, vec![node_msg("x1", Role::User, "redo")], None);
        assert_eq!(voided_ids(&events), vec!["tc-1"], "got {events:?}");

        let events = dispatch(
            &mut agg,
            CommandPayload::Wake {
                now: Utc::now() + chrono::Duration::hours(1),
            },
            &system(),
        );
        assert!(events.is_empty(), "got {events:?}");
    }

    #[test]
    fn promoting_submit_drops_a_queued_settle_for_the_branch_it_forked_away() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        request_client_tool(&mut agg, "tc-1"); // anchored at u1

        // Settle queues behind the pending edit; a user message queues behind the settle.
        let d2 = open_decision(&mut agg, "redo");
        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                result: "ok".to_string(),
            },
            &machine(),
        );
        let settle_id = decision_with(&events, |t| matches!(t, Trigger::ToolFinished { .. }))
            .expect("on-path settle queues behind the pending decision");
        open_decision(&mut agg, "also this");

        let events = submit_state(&mut agg, d2, vec![node_msg("x1", Role::User, "redo")], None);
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                EventPayload::DecisionRequestDropped(p) => Some(p.decision_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(dropped, vec![settle_id.as_str()], "got {events:?}");
        assert!(
            settle_decisions(&events).is_empty(),
            "the stale settle is not delivered; got {events:?}"
        );
        let promoted = events.iter().find_map(|e| match e {
            EventPayload::WorkerDecisionRequested(p) => agg
                .state
                .worker_decisions
                .get(&p.decision_id)
                .map(|d| &d.trigger),
            _ => None,
        });
        assert!(
            matches!(promoted, Some(Trigger::ClientMessage { .. })),
            "the next live decision is promoted past the dropped settle; got {events:?}"
        );
        assert!(
            agg.state.queued_decisions().is_empty(),
            "nothing is left queued"
        );
    }

    #[test]
    fn fork_voids_a_pending_llm_call() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        request_llm(&mut agg, "llm-1", LlmHandler::Server); // anchored at u1

        let d2 = open_decision(&mut agg, "redo");
        let events = submit_state(&mut agg, d2, vec![node_msg("x1", Role::User, "redo")], None);
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::CallVoided(v)
                    if v.kind == EffectKind::LlmCall && v.id == "llm-1"
            )),
            "the fork voids the in-flight call; got {events:?}"
        );

        // The executor's late result no-ops.
        let events = complete_llm(&mut agg, "llm-1", 0, &system());
        assert!(events.is_empty(), "got {events:?}");
    }

    #[test]
    fn fork_drops_a_queued_execute_decision() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        let call_tool = |id: &str| Action::CallTool {
            id: id.to_string(),
            name: "slow".to_string(),
            arguments: "{}".to_string(),
            retry: RetryPolicy::no_retry(),
        };
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![node_msg("u1", Role::User, "hi")],
                actions: vec![call_tool("t1"), call_tool("t2")],
                state: None,
                agent: None,
            },
            &machine(),
        );
        // t1's execute is delivered; t2's queues behind it.
        let exec_t1 = decision_with(
            &events,
            |t| matches!(t, Trigger::ToolExecute { id, .. } if id == "t1"),
        )
        .expect("first execute is delivered");
        let exec_t2 = decision_with(
            &events,
            |t| matches!(t, Trigger::ToolExecute { id, .. } if id == "t2"),
        )
        .expect("second execute queues");

        let events = submit_state(
            &mut agg,
            exec_t1,
            vec![node_msg("x1", Role::User, "redo")],
            None,
        );
        let mut voided = voided_ids(&events);
        voided.sort_unstable();
        assert_eq!(voided, vec!["t1", "t2"], "got {events:?}");
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                EventPayload::DecisionRequestDropped(p) => Some(p.decision_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(dropped, vec![exec_t2.as_str()], "got {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))),
            "nothing is left to promote; got {events:?}"
        );
    }

    #[test]
    fn submit_settling_work_it_forked_away_voids_it_instead() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        request_client_tool(&mut agg, "tc-1"); // anchored at u1

        let d2 = open_decision(&mut agg, "redo");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d2,
                transcript: vec![node_msg("x1", Role::User, "redo")],
                actions: vec![Action::ToolResult {
                    id: "tc-1".to_string(),
                    attempt: None,
                    result: "late".to_string(),
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );
        assert_eq!(voided_ids(&events), vec!["tc-1"], "got {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "the settle dies with the branch; got {events:?}"
        );
        assert!(settle_decisions(&events).is_empty(), "got {events:?}");
    }

    #[test]
    fn submit_settling_a_call_its_own_interrupt_voided_swallows_it() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        request_llm(&mut agg, "llm-1", LlmHandler::Server);

        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![node_msg("u1", Role::User, "hi")],
                actions: vec![
                    Action::Interrupt {
                        interrupt_id: "int-1".to_string(),
                        reason: "hold".to_string(),
                        payload: serde_json::Value::Null,
                    },
                    Action::LlmResult {
                        id: "llm-1".to_string(),
                        attempt: None,
                        response: llm_response("late"),
                    },
                ],
                state: None,
                agent: None,
            },
            &machine(),
        );
        assert_eq!(voided_ids(&events), vec!["llm-1"], "got {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::LlmCallCompleted(_))),
            "the settle dies with the voided call; got {events:?}"
        );
        assert_eq!(
            agg.state.llm_calls.get("llm-1").unwrap().tracking.status,
            EffectStatus::Failed,
        );
    }

    #[test]
    fn void_guard_matches_kind_not_just_id() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        request_client_tool(&mut agg, "shared"); // anchored at u1

        // Sub-agent with the same id, anchored one node deeper.
        let d2 = open_decision(&mut agg, "more");
        submit_state(
            &mut agg,
            d2,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "?"),
            ],
            None,
        );
        dispatch(
            &mut agg,
            CommandPayload::RequestSubAgent {
                session_id: "child-x".to_string(),
                agent_id: "helper".to_string(),
                tool_call_id: "shared".to_string(),
                retry: RetryPolicy::no_retry(),
            },
            &system(),
        );

        // Forking voids the sub-agent, not the tool; the tool settle must survive the id collision.
        let d3 = open_decision(&mut agg, "redo");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d3,
                transcript: vec![
                    node_msg("u1", Role::User, "hi"),
                    node_msg("b1", Role::Assistant, "!"),
                ],
                actions: vec![Action::ToolResult {
                    id: "shared".to_string(),
                    attempt: None,
                    result: "ok".to_string(),
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );
        assert_eq!(voided_ids(&events), vec!["shared"], "got {events:?}");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "the tool settle lands despite the voided sub-agent sharing its id; got {events:?}"
        );
    }

    #[test]
    fn fork_drops_a_retrying_settle_decision() {
        let mut agg = SessionAggregate::new(
            "sess-1".to_string(),
            "tenant-a".to_string(),
            SessionState::new("sess-1".to_string()),
        );
        dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: "tenant-a".to_string(),
                    id: Some("user-1".to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy {
                    timeout_secs: None,
                    max_retries: 2,
                    backoff_base_secs: 1,
                    backoff_max_secs: 1,
                },
            },
            &system(),
        );
        drain_session_start(&mut agg);

        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        request_client_tool(&mut agg, "tc-1"); // anchored at u1

        // The on-path settle is delivered, then the worker errors retryably.
        let events = dispatch(
            &mut agg,
            CommandPayload::CompleteToolCall {
                tool_call_id: "tc-1".to_string(),
                attempt: Some(0),
                result: "ok".to_string(),
            },
            &machine(),
        );
        let settle_id = decision_with(&events, |t| matches!(t, Trigger::ToolFinished { .. }))
            .expect("on-path settle is delivered");
        dispatch(
            &mut agg,
            CommandPayload::FailWorkerDecision {
                decision_id: settle_id.clone(),
                error: "worker crashed".to_string(),
                retryable: true,
            },
            &machine(),
        );

        let d2 = open_decision(&mut agg, "redo");
        let events = submit_state(&mut agg, d2, vec![node_msg("x1", Role::User, "redo")], None);
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                EventPayload::DecisionRequestDropped(p) => Some(p.decision_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(dropped, vec![settle_id.as_str()], "got {events:?}");
        assert!(
            settle_decisions(&events).is_empty(),
            "the stale settle is not re-delivered; got {events:?}"
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::Wake {
                now: Utc::now() + chrono::Duration::hours(1),
            },
            &system(),
        );
        assert!(events.is_empty(), "got {events:?}");
    }

    // ── Branch-scoped interrupts ─────────────────────────────────────────

    fn interrupt(agg: &mut SessionAggregate, id: &str) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::Interrupt {
                interrupt_id: id.to_string(),
                reason: "paused".to_string(),
                payload: serde_json::Value::Null,
            },
            &system(),
        )
    }

    fn resume(agg: &mut SessionAggregate, id: &str) -> Vec<EventPayload> {
        dispatch(
            agg,
            CommandPayload::ResumeInterrupt {
                interrupt_id: id.to_string(),
                payload: serde_json::Value::Null,
            },
            &system(),
        )
    }

    /// A session with `u1 -> a1` recorded and the head at `a1`.
    fn parked_session() -> SessionAggregate {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            None,
        );
        interrupt(&mut agg, "int-1");
        agg
    }

    #[test]
    fn client_interrupt_anchors_at_the_head() {
        let agg = parked_session();
        let open = agg.state.open_interrupt("int-1").expect("open interrupt");
        assert_eq!(open.anchor.as_deref(), Some("a1"));
        assert!(agg.state.head_parked());
    }

    #[test]
    fn edited_view_escapes_a_parked_head_and_the_interrupt_survives() {
        let mut agg = parked_session();
        let events = submit_messages(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("", Role::User, "actually, do this instead"),
            ],
        );
        assert!(
            decision_with(&events, |t| matches!(t, Trigger::ClientTranscript { .. })).is_some(),
            "the escaping view is delivered; got {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))),
            "dispatched live, not queued; got {events:?}"
        );
        assert!(
            agg.state.open_interrupt("int-1").is_some(),
            "the interrupt stays open on its branch"
        );
    }

    #[test]
    fn appending_to_a_parked_branch_is_rejected() {
        let agg = parked_session();
        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload: ClientPayload::Messages(ClientMessages {
                        messages: vec![
                            node_msg("u1", Role::User, "hi"),
                            node_msg("a1", Role::Assistant, "hello"),
                            node_msg("", Role::User, "and then?"),
                        ],
                        stream: false,
                        client: Default::default(),
                    }),
                    turn_id: None,
                },
                &system(),
            )
            .expect_err("an append lands on the parked branch");
        assert!(matches!(err, SessionError::SessionInterrupted));
    }

    #[test]
    fn global_interrupt_gates_all_new_views() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        interrupt(&mut agg, "int-1"); // empty tree: anchorless, global
        assert_eq!(agg.state.open_interrupt("int-1").unwrap().anchor, None);
        let err = agg
            .state
            .handle(
                CommandPayload::SubmitClientPayload {
                    payload: ClientPayload::Messages(ClientMessages {
                        messages: vec![node_msg("", Role::User, "hello")],
                        stream: false,
                        client: Default::default(),
                    }),
                    turn_id: None,
                },
                &system(),
            )
            .expect_err("a global interrupt parks every path");
        assert!(matches!(err, SessionError::SessionInterrupted));
    }

    #[test]
    fn answer_carrying_view_is_accepted_and_queued_while_parked() {
        let mut agg = parked_session();
        request_client_tool(&mut agg, "tc-1"); // anchored at a1, spared by the interrupt
        let events = submit_messages(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
                tool_msg("tc-1", "the answer"),
            ],
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::ToolCallCompleted(_))),
            "the answer settles; got {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(_))),
            "the follow-up queues until resume; got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))),
            "nothing dispatches while parked; got {events:?}"
        );
    }

    /// Escape a parked `u1 -> a1` session onto a sibling branch `u1 -> e1`.
    fn escape_to_e1(agg: &mut SessionAggregate) {
        let events = submit_messages(
            agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("e1", Role::User, "actually, do this instead"),
            ],
        );
        let d = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("the escape dispatches");
        submit_state(
            agg,
            d,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("e1", Role::User, "actually, do this instead"),
            ],
            None,
        );
    }

    #[test]
    fn two_parked_branches_coexist_and_resume_independently() {
        let mut agg = parked_session();
        escape_to_e1(&mut agg);
        assert_eq!(agg.state.head_id.as_deref(), Some("e1"));
        assert!(!agg.state.head_parked(), "the new branch starts unparked");

        interrupt(&mut agg, "int-2"); // anchored at e1
        assert_eq!(agg.state.open_interrupts.len(), 2);
        assert!(agg.state.head_parked());

        let events = resume(&mut agg, "int-1");
        assert!(
            matches!(events.as_slice(), [EventPayload::InterruptResumed(_)]),
            "off-head resume clears silently; got {events:?}"
        );
        assert!(agg.state.head_parked(), "int-2 still parks the head");

        let events = resume(&mut agg, "int-2");
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::DecisionRequestQueued(_),
                    EventPayload::WorkerDecisionRequested(_),
                ]
            ),
            "on-head resume fires the trigger; got {events:?}"
        );
        assert!(agg.state.open_interrupts.is_empty());
        assert!(!agg.state.head_parked());
    }

    #[test]
    fn resume_with_a_live_escape_decision_queues_the_trigger() {
        let mut agg = parked_session();
        // Escape dispatched but not yet submitted: a live decision.
        submit_messages(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("", Role::User, "meanwhile, on another branch"),
            ],
        );
        assert!(agg.state.has_pending_worker_decision());

        let events = resume(&mut agg, "int-1");
        assert!(
            matches!(
                events.as_slice(),
                [
                    EventPayload::InterruptResumed(_),
                    EventPayload::DecisionRequestQueued(_),
                ]
            ),
            "the resume trigger queues behind the live decision; got {events:?}"
        );
    }

    #[test]
    fn interrupt_voiding_is_scoped_to_the_parked_path() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let ctx = CommitContext {
            span: SpanContext::root(),
            occurred_at: Utc::now(),
        };
        let msg = |id: &str, parent: Option<&str>| {
            EventPayload::NewMessage(NewMessage {
                message: node_msg(id, Role::User, "m").record(),
                parent_id: parent.map(str::to_string),
            })
        };
        let llm = |id: &str| {
            EventPayload::LlmCallRequested(LlmCallRequested {
                call_id: id.to_string(),
                attempt: 0,
                request: request_with(vec![]),
                stream: false,
                retry: RetryPolicy::no_retry(),
                handler: LlmHandler::Server,
                format: None,
            })
        };
        agg.commit(vec![msg("u1", None), msg("a1", Some("u1"))], &ctx);
        agg.commit(vec![llm("L1")], &ctx); // anchored at a1
        agg.commit(vec![msg("e1", Some("u1"))], &ctx); // head moves to the sibling
        agg.commit(vec![llm("L2")], &ctx); // anchored at e1

        let events = interrupt(&mut agg, "int-1"); // anchored at e1
        assert_eq!(
            voided_ids(&events),
            vec!["L2"],
            "voiding spares the other branch; got {events:?}"
        );
    }

    #[test]
    fn worker_interrupt_anchors_at_the_post_reconcile_head() {
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d1,
                transcript: vec![
                    node_msg("u1", Role::User, "hi"),
                    node_msg("x1", Role::Assistant, "confirm?"),
                ],
                actions: vec![Action::Interrupt {
                    interrupt_id: "int-1".to_string(),
                    reason: "confirmation".to_string(),
                    payload: serde_json::Value::Null,
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );
        let anchor = events
            .iter()
            .find_map(|e| match e {
                EventPayload::SessionInterrupted(p) => Some(p.anchor.clone()),
                _ => None,
            })
            .expect("the action raises the interrupt");
        assert_eq!(anchor.as_deref(), Some("x1"), "anchored at head_after");
        assert_eq!(agg.state.head_id.as_deref(), Some("x1"));
        assert!(agg.state.head_parked());
    }

    #[test]
    fn worker_interrupt_on_an_escaped_branch_is_not_deduped_by_the_old_one() {
        let mut agg = parked_session();
        let events = submit_messages(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("e1", Role::User, "other branch"),
            ],
        );
        let d = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("the escape dispatches");
        let events = dispatch(
            &mut agg,
            CommandPayload::SubmitWorkerDecision {
                decision_id: d,
                transcript: vec![
                    node_msg("u1", Role::User, "hi"),
                    node_msg("e1", Role::User, "other branch"),
                ],
                actions: vec![Action::Interrupt {
                    interrupt_id: "int-2".to_string(),
                    reason: "confirmation".to_string(),
                    payload: serde_json::Value::Null,
                }],
                state: None,
                agent: None,
            },
            &machine(),
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::SessionInterrupted(_))),
            "idempotence is per-path, not per-session; got {events:?}"
        );
        assert_eq!(agg.state.open_interrupts.len(), 2);
    }

    #[test]
    fn promotion_and_wake_skip_parked_branches() {
        let mut agg = parked_session();
        request_client_tool(&mut agg, "tc-1"); // anchored at a1
        let events = complete_tool(&mut agg, "tc-1", "ok");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::DecisionRequestQueued(_))),
            "the answer queues while parked; got {events:?}"
        );

        assert_eq!(agg.state.wake_at(), None, "a parked head schedules no wake");
        let events = wake(&mut agg);
        assert!(
            events.is_empty(),
            "wake does not promote a parked decision; got {events:?}"
        );

        let events = resume(&mut agg, "int-1");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))),
            "resume fires the interrupt.resumed decision; got {events:?}"
        );
    }

    #[test]
    fn wake_promotes_a_queued_decision_on_an_unparked_branch() {
        let mut agg = parked_session();
        escape_to_e1(&mut agg);
        // Plant an anchorless queued decision: promotable from the unparked head.
        agg.commit(
            vec![EventPayload::DecisionRequestQueued(DecisionRequestQueued {
                decision_id: "d-queued".to_string(),
                trigger: Trigger::ClientMessage {
                    messages: vec![node_msg("", Role::User, "queued")],
                    client: ClientContext::default(),
                },
            })],
            &CommitContext {
                span: SpanContext::root(),
                occurred_at: Utc::now(),
            },
        );
        assert!(
            agg.state.wake_at().is_some(),
            "a promotable queued decision wakes immediately"
        );
        let events = wake(&mut agg);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventPayload::WorkerDecisionRequested(_))),
            "the off-head interrupt does not block promotion; got {events:?}"
        );
    }

    #[test]
    fn anchorless_interrupt_event_replays_as_global() {
        let payload = serde_json::json!({
            "type": "session.interrupted",
            "interrupt_id": "int-old",
            "origin": "frontend",
            "reason": "paused",
            "payload": null,
        });
        let event: EventPayload = serde_json::from_value(payload).expect("old event deserializes");
        let mut agg = create_session("sess-1", "tenant-a", "user-1");
        let d1 = open_decision(&mut agg, "hi");
        submit_state(&mut agg, d1, vec![node_msg("u1", Role::User, "hi")], None);
        agg.commit(
            vec![event],
            &CommitContext {
                span: SpanContext::root(),
                occurred_at: Utc::now(),
            },
        );
        let open = agg.state.open_interrupt("int-old").expect("open");
        assert_eq!(open.anchor, None);
        assert!(
            agg.state.head_parked(),
            "an anchorless interrupt parks every path"
        );
    }

    #[test]
    fn escape_decision_retry_fires_while_the_head_is_parked() {
        let mut agg = SessionAggregate::new(
            "sess-1".to_string(),
            "tenant-a".to_string(),
            SessionState::new("sess-1".to_string()),
        );
        dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: "tenant-a".to_string(),
                    id: Some("user-1".to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy {
                    timeout_secs: None,
                    max_retries: 2,
                    backoff_base_secs: 1,
                    backoff_max_secs: 1,
                },
            },
            &system(),
        );
        drain_session_start(&mut agg);
        let d1 = open_decision(&mut agg, "hi");
        submit_state(
            &mut agg,
            d1,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("a1", Role::Assistant, "hello"),
            ],
            None,
        );
        interrupt(&mut agg, "int-1");

        let events = submit_messages(
            &mut agg,
            vec![
                node_msg("u1", Role::User, "hi"),
                node_msg("", Role::User, "escape"),
            ],
        );
        let escape = events
            .iter()
            .find_map(|e| match e {
                EventPayload::WorkerDecisionRequested(p) => Some(p.decision_id.clone()),
                _ => None,
            })
            .expect("the escape dispatches while parked");
        dispatch(
            &mut agg,
            CommandPayload::FailWorkerDecision {
                decision_id: escape.clone(),
                error: "worker flaked".to_string(),
                retryable: true,
            },
            &machine(),
        );

        assert!(
            agg.state.wake_at().is_some(),
            "the escape's retry schedules a wake despite the parked head"
        );
        let events = dispatch(
            &mut agg,
            CommandPayload::Wake {
                now: Utc::now() + chrono::Duration::hours(1),
            },
            &system(),
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                EventPayload::WorkerDecisionRequested(p) if p.decision_id == escape
            )),
            "the wake re-fires the escape decision; got {events:?}"
        );
    }

    #[test]
    fn parked_branch_deadlines_are_suppressed_but_live_branch_timers_run() {
        let deadline_policy = RetryPolicy {
            timeout_secs: Some(60),
            max_retries: 0,
            backoff_base_secs: 1,
            backoff_max_secs: 1,
        };
        let mut agg = parked_session();
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-parked".to_string(),
                name: "slow".to_string(),
                arguments: "{}".to_string(),
                retry: deadline_policy.clone(),
            },
            &system(),
        );
        assert_eq!(
            agg.state.wake_at(),
            None,
            "a parked branch's deadline schedules nothing"
        );

        // The escape voids tc-parked; the live branch's deadline still runs.
        escape_to_e1(&mut agg);
        dispatch(
            &mut agg,
            CommandPayload::RequestToolCall {
                tool_call_id: "tc-live".to_string(),
                name: "slow".to_string(),
                arguments: "{}".to_string(),
                retry: deadline_policy,
            },
            &system(),
        );
        assert!(
            agg.state.wake_at().is_some(),
            "a live branch's deadline keeps running"
        );
    }
}
