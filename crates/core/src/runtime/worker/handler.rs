use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::protocol::{ChannelKind, DecisionResponse, DecisionTrigger, Message};
use crate::runtime::blob::BlobStore;
use crate::runtime::event_store::{EventFilter, EventStore, Seq};
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCursorStore,
    ProcessorError,
};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::propose::{propose, Proposing};
use crate::runtime::session::state::SessionState;
use crate::runtime::session::wire::to_wire_trigger;
use crate::runtime::session::SessionEvent;
use serde_json::Value;

use super::{AgentDirectory, WorkerDecisionRequest, WorkerQueue};

pub trait ChannelProposer: Send + Sync {
    fn channel(&self) -> ChannelKind;

    fn translate(&self, _cx: &Proposal<'_>) -> Option<DecisionResponse> {
        None
    }

    fn render(&self, _cx: &Proposal<'_>, _proposed: &DecisionResponse) -> Option<Value> {
        None
    }
}

pub struct Proposal<'a> {
    pub trigger: &'a DecisionTrigger,
    pub state: &'a SessionState,
    pub events: &'a [SessionEvent],
    pub transcript: &'a [Message],
}

struct WorkerDecisionProjection {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn WorkerQueue>,
    agents: Arc<dyn AgentDirectory>,
    proposers: Vec<Arc<dyn ChannelProposer>>,
    blobs: Arc<dyn BlobStore>,
}

impl WorkerDecisionProjection {
    fn new(
        store: Arc<dyn EventStore>,
        queue: Arc<dyn WorkerQueue>,
        agents: Arc<dyn AgentDirectory>,
        proposers: Vec<Arc<dyn ChannelProposer>>,
        blobs: Arc<dyn BlobStore>,
    ) -> Self {
        Self {
            store,
            queue,
            agents,
            proposers,
            blobs,
        }
    }
}

#[async_trait::async_trait]
impl EventProcessor for WorkerDecisionProjection {
    fn name(&self) -> &'static str {
        "worker_decision_enqueue"
    }

    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError> {
        if let Some(decision) = extract(
            self.store.as_ref(),
            self.agents.as_ref(),
            &self.proposers,
            self.blobs.as_ref(),
            event,
        )
        .await?
        {
            tracing::debug!(
                session_id = %decision.session_id,
                decision_id = %decision.decision_id,
                agent_id = %decision.agent_id,
                trigger_type = ?decision.trigger,
                "enqueuing worker decision"
            );
            self.queue.enqueue(decision).await;
        }
        Ok(())
    }
}

pub fn spawn_worker_processor(
    store: Arc<dyn EventStore>,
    cursor_store: Arc<dyn ProcessorCursorStore>,
    queue: Arc<dyn WorkerQueue>,
    agents: Arc<dyn AgentDirectory>,
    proposers: Vec<Arc<dyn ChannelProposer>>,
    blobs: Arc<dyn BlobStore>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(WorkerDecisionProjection::new(
        store.clone(),
        queue,
        agents,
        proposers,
        blobs,
    ));
    EventProcessorRunner::new(
        store,
        cursor_store,
        projection,
        EventProcessorRunnerConfig::default(),
        cancel,
    )
    .spawn()
}

async fn extract(
    store: &dyn EventStore,
    agents: &dyn AgentDirectory,
    proposers: &[Arc<dyn ChannelProposer>],
    blobs: &dyn BlobStore,
    event: SessionEvent,
) -> Result<Option<WorkerDecisionRequest>, ProcessorError> {
    let req = match &event.payload {
        EventPayload::DecisionDispatched(req) => req,
        _ => return Ok(None),
    };
    let meta = &event.meta;

    let (Some(agent_id), Some(owner)) = (meta.agent_id.as_ref(), meta.owner.as_ref()) else {
        return Ok(None);
    };

    let Some(wd) = meta.decisions.iter().find(|d| d.decision_id == req.id) else {
        return Ok(None);
    };

    let session = store
        .load(&event.tenant_id, &event.session_id)
        .await
        .map_err(|e| ProcessorError::Apply(format!("load session for decision: {e}")))?;

    let Some(trigger) = session
        .state
        .worker_decision(&req.id)
        .map(|d| d.trigger.clone())
    else {
        return Ok(None);
    };

    let worker = session.state.worker.clone();
    let state = session.state.rewind(event.seq, meta.head_id.as_deref());

    let message_tree = state.message_tree();

    let transcript = message_tree
        .head_id
        .as_deref()
        .map(|h| message_tree.path_to(h))
        .unwrap_or_default();
    let at = state.at_head();
    let open_llm_calls = at.open_llm_calls();

    let pending_calls = meta.pending_work(&req.id);
    let worker_state = at.resolve_state_for();
    let agent_config = at.resolve_agent_for();

    let trigger = to_wire_trigger(
        trigger,
        &transcript,
        &message_tree,
        &open_llm_calls,
        blobs,
        &event.tenant_id,
    )
    .await
    .map_err(|e| {
        ProcessorError::Apply(format!(
            "resolving a decision for delivery: {}",
            e.error.message
        ))
    })?;
    let proposed = propose(
        &trigger,
        &Proposing {
            state: at,
            pending_calls,
        },
    )
    .or_else(|| {
        matches!(trigger, DecisionTrigger::SessionStart)
            .then(|| {
                agent_config
                    .clone()
                    .or_else(|| agents.agent(&event.tenant_id, agent_id)?.config)
            })
            .flatten()
            .map(|config| DecisionResponse {
                agent: Some(config),
                ..Default::default()
            })
    })
    .unwrap_or_default();
    let owning = ChannelKind::owning(owner);
    let proposer = proposers.iter().find(|p| Some(p.channel()) == owning);

    let turn_events: Vec<SessionEvent> = match (proposer.is_none(), state.turn_started_seq) {
        (false, Some(start)) => store
            .query_events(&EventFilter {
                session_id: Some(event.session_id.clone()),
                tenant_id: Some(event.tenant_id.clone()),
                after_seq: Some(Seq(start.saturating_sub(1))),
                limit: None,
            })
            .await
            .map_err(|e| ProcessorError::Apply(format!("load turn events: {e}")))?
            .into_iter()
            .filter(|e| e.seq <= event.seq)
            .collect(),
        _ => Vec::new(),
    };
    let turn_events = match event.meta.turn_id.as_deref() {
        Some(stamped)
            if !turn_events.iter().any(
                |e| matches!(&e.payload, EventPayload::TurnStarted(t) if t.turn_id == stamped),
            ) =>
        {
            Vec::new()
        }
        _ => turn_events,
    };
    let cx = Proposal {
        trigger: &trigger,
        state: &state,
        events: &turn_events,
        transcript: &transcript,
    };
    let mut proposed = match (proposer, &trigger) {
        (Some(p), DecisionTrigger::ClientAction { .. }) => p.translate(&cx).unwrap_or(proposed),
        _ => proposed,
    };
    if let Some(view) = proposer.and_then(|p| p.render(&cx, &proposed)) {
        let key = owning
            .expect("a proposer ran, so the session has a channel")
            .to_string();
        debug_assert!(
            !proposed.channels.contains_key(&key),
            "{key} rendered over what it translated"
        );
        proposed.channels.insert(key, view);
    }

    Ok(Some(WorkerDecisionRequest {
        session_id: event.session_id.clone(),
        decision_id: req.id.clone(),
        agent_id: agent_id.clone(),
        identity: owner.clone(),
        trigger,
        proposed,
        state: worker_state,
        agent: agent_config,
        worker,
        calls: meta.calls.clone(),
        pending_calls,
        transcript,
        message_tree,
        ancestry: meta.ancestry.clone(),
        span: event.span.clone(),
        attempts: wd.attempts,
        deadline: wd.deadline,
        turn_id: meta.turn_id.clone(),
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;

    use std::collections::BTreeMap;

    use super::{extract, AgentDirectory};
    use crate::connectors::RemoteTool;
    use crate::protocol::{
        AgentConfig, Approve, ClientMessage, ClientPayload, Content, DraftMessage, EffectKind,
        Issuer, LlmRequest, LlmResponse, McpServer, Requester, RetryPolicy, Role, SessionOwner,
        StoredResult, Subject, ToolCall, ToolCallFunction,
    };
    use crate::runtime::event_store::{
        AppendInput, BroadcastBus, EventBus, EventFilter, EventStore, EventTap, StoreError,
    };
    use crate::runtime::llm::{LlmBlock, LlmBlocks};
    use crate::runtime::session::command::{CommandPayload, TurnTarget};
    use crate::runtime::session::decision::LlmHandler;
    use crate::runtime::session::effects::Outcome;
    use crate::runtime::session::events::EventPayload;
    use crate::runtime::session::state::SessionState;
    use crate::runtime::session::wire::resolve_response;
    use crate::runtime::session::{CommitContext, SessionAggregate, SessionEvent};
    use crate::runtime::span::SpanContext;
    use crate::runtime::worker::directory::{AgentEntry, StaticAgentDirectory};
    use crate::runtime::worker::EmptyAgentDirectory;
    use crate::runtime::Caller;

    struct FrozenStore {
        session: SessionAggregate,
        events: BroadcastBus,
    }

    #[async_trait::async_trait]
    impl EventStore for FrozenStore {
        async fn append(&self, _input: AppendInput) -> Result<(), StoreError> {
            Err(StoreError::Internal("read-only test store".into()))
        }

        async fn load(
            &self,
            _tenant_id: &str,
            _session_id: &str,
        ) -> Result<SessionAggregate, StoreError> {
            Ok(self.session.clone())
        }

        async fn query_events(
            &self,
            _filter: &EventFilter,
        ) -> Result<Vec<SessionEvent>, StoreError> {
            Ok(vec![])
        }

        fn subscribe(&self) -> EventTap {
            self.events.subscribe()
        }
    }

    fn dispatch(
        agg: &mut SessionAggregate,
        cmd: CommandPayload,
        caller: &Caller,
    ) -> Vec<SessionEvent> {
        let now = Utc::now();
        let events = agg.handle(cmd, caller, now).expect("setup command failed");
        agg.commit(
            events,
            &CommitContext {
                span: SpanContext::root(),
                occurred_at: now,
            },
        )
    }

    fn system() -> Caller {
        Caller::System {
            tenant_id: "tenant-a".to_string(),
        }
    }

    fn user_msg(text: &str) -> DraftMessage {
        DraftMessage {
            id: None,
            role: Role::User,
            content: Some(Content::Text(text.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    fn config(model: &str) -> AgentConfig {
        AgentConfig {
            llm: Some("claude".to_string()),
            model: model.to_string(),
            ..Default::default()
        }
    }

    fn directory(config: AgentConfig) -> StaticAgentDirectory {
        StaticAgentDirectory::new(
            "tenant-a".to_string(),
            std::collections::BTreeMap::from([(
                "agent-1".to_string(),
                AgentEntry {
                    config: Some(config),
                    hosting: crate::worker::Hosting::Engine,
                },
            )]),
            Default::default(),
        )
    }

    async fn wire_request_for_client_message(agent: Option<AgentConfig>) -> serde_json::Value {
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
                    requester: Requester::new(
                        Subject::new(Issuer::app(), "user-1".to_string()),
                        Default::default(),
                    ),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
                agent: None,
                worker: None,
            },
            &system(),
        );
        let start = created
            .iter()
            .find_map(|e| match &e.payload {
                EventPayload::DecisionDispatched(w) => Some(w.id.clone()),
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
                channels: Default::default(),
            },
            &system(),
        );
        let event = dispatch(
            &mut agg,
            CommandPayload::SubmitClientPayload {
                payload: ClientPayload::Message(ClientMessage {
                    message: user_msg("hi"),
                    stream: false,
                }),
                turn: TurnTarget::Detached,
                queue: false,
            },
            &system(),
        )
        .into_iter()
        .find(|e| matches!(e.payload, EventPayload::DecisionDispatched(_)))
        .expect("the client decision goes live");

        let store = FrozenStore {
            session: agg,
            events: BroadcastBus::new(1),
        };
        let req = extract(
            &store,
            &EmptyAgentDirectory,
            &[],
            &crate::runtime::blob::NOWHERE,
            event,
        )
        .await
        .expect("extract succeeds")
        .expect("a deliverable request");
        serde_json::to_value(&req).expect("wire request serializes")
    }

    async fn wire_request_for_session_start(agents: &dyn AgentDirectory) -> serde_json::Value {
        let mut agg = SessionAggregate::new(
            "sess-1".to_string(),
            "tenant-a".to_string(),
            SessionState::new("sess-1".to_string()),
        );
        let event = dispatch(
            &mut agg,
            CommandPayload::CreateSession {
                agent_id: "agent-1".to_string(),
                owner: SessionOwner {
                    tenant_id: "tenant-a".to_string(),
                    requester: Requester::new(
                        Subject::new(Issuer::app(), "user-1".to_string()),
                        Default::default(),
                    ),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
                agent: None,
                worker: None,
            },
            &system(),
        )
        .into_iter()
        .find(|e| matches!(e.payload, EventPayload::DecisionDispatched(_)))
        .expect("CreateSession opens a session.start decision");

        let store = FrozenStore {
            session: agg,
            events: BroadcastBus::new(1),
        };
        let req = extract(&store, agents, &[], &crate::runtime::blob::NOWHERE, event)
            .await
            .expect("extract succeeds")
            .expect("a deliverable request");
        serde_json::to_value(&req).expect("wire request serializes")
    }

    #[tokio::test]
    async fn session_start_is_seeded_with_the_declared_config() {
        let wire = wire_request_for_session_start(&directory(config("m1"))).await;
        assert_eq!(wire["proposed"]["agent"]["model"], "m1");
        assert_eq!(wire["proposed"]["agent"]["llm"], "claude");
    }

    #[tokio::test]
    async fn session_start_without_a_declaration_carries_no_proposal() {
        let wire = wire_request_for_session_start(&EmptyAgentDirectory).await;
        assert!(
            wire["proposed"]["agent"].is_null(),
            "got {}",
            wire["proposed"]
        );
        assert!(wire["proposed"]["actions"]
            .as_array()
            .is_none_or(|a| a.is_empty()));
    }

    #[tokio::test]
    async fn a_non_start_trigger_is_not_seeded() {
        let wire = wire_request_for_client_message(None).await;
        assert!(
            wire["proposed"]["agent"].is_null(),
            "the directory does not reach a client.messages proposal; got {}",
            wire["proposed"]
        );
    }

    #[tokio::test]
    async fn no_config_client_message_proposes_an_interrupt() {
        let wire = wire_request_for_client_message(None).await;
        let actions = wire["proposed"]["actions"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert!(
            actions.iter().any(|a| a["type"] == "interrupt"),
            "with no config the proposal pauses the session so a blind echoer \
             fails loudly instead of submitting a silent no-op; got {}",
            wire["proposed"]
        );
    }

    async fn drive(agg: &mut SessionAggregate, events: Vec<SessionEvent>) -> Vec<SessionEvent> {
        let mut queue: Vec<SessionEvent> = events;
        let mut settled = Vec::new();
        while let Some(event) = queue.pop() {
            if !matches!(event.payload, EventPayload::DecisionDispatched(_)) {
                continue;
            }
            let store = FrozenStore {
                session: agg.clone(),
                events: BroadcastBus::new(1),
            };
            let request = extract(
                &store,
                &EmptyAgentDirectory,
                &[],
                &crate::runtime::blob::NOWHERE,
                event,
            )
            .await
            .expect("extract succeeds")
            .expect("a deliverable request");
            let blocks =
                LlmBlocks::new(BTreeMap::from([("claude".to_string(), LlmBlock::engine())]));
            let resolved = resolve_response(
                request.proposed,
                request.agent.as_ref(),
                Some(&request.trigger),
                &blocks,
                &crate::runtime::blob::NOWHERE,
                "tenant-a",
            )
            .await
            .expect("the proposal resolves");
            let next = dispatch(
                agg,
                CommandPayload::SubmitWorkerDecision {
                    decision_id: request.decision_id,
                    transcript: resolved.messages,
                    actions: resolved.actions,
                    state: resolved.state,
                    agent: resolved.agent,
                    channels: resolved.channels,
                },
                &system(),
            );
            queue.extend(next.iter().cloned());
            settled.extend(next);
        }
        settled
    }

    async fn two_calls_awaiting_approval() -> SessionAggregate {
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
                    requester: Requester::new(
                        Subject::new(Issuer::app(), "user-1".to_string()),
                        Default::default(),
                    ),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
                agent: None,
                worker: None,
            },
            &system(),
        );
        let start = created
            .iter()
            .find_map(|e| match &e.payload {
                EventPayload::DecisionDispatched(w) => Some(w.id.clone()),
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
                agent: Some(AgentConfig {
                    mcp: vec![McpServer {
                        id: "sentry".into(),
                        tools: None,
                        auth_failure: Default::default(),
                        tool_sync_failure: Default::default(),
                        approve: Approve::Destructive,
                    }],
                    ..config("m1")
                }),
                channels: Default::default(),
            },
            &system(),
        );
        dispatch(
            &mut agg,
            CommandPayload::settle(
                EffectKind::ConnectorSync,
                "mcp.sentry".to_string(),
                None,
                Outcome::Connector {
                    server: None,
                    prefix: Some("sentry".to_string()),
                    tools: vec![RemoteTool {
                        name: "delete_issue".to_string(),
                        title: None,
                        description: "Delete an issue.".to_string(),
                        input: None,
                        output: None,
                        annotations: crate::connectors::ToolAnnotations {
                            destructive: Some(true),
                            ..Default::default()
                        },
                    }],
                    instructions: None,
                },
            ),
            &system(),
        );
        dispatch(
            &mut agg,
            CommandPayload::RequestLlmCall {
                llm: "claude".to_string(),
                call_id: "call-1".to_string(),
                request: LlmRequest {
                    model: "m1".to_string(),
                    messages: vec![],
                    tools: None,
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
        let finished = dispatch(
            &mut agg,
            CommandPayload::settle(
                EffectKind::LlmCall,
                "call-1".to_string(),
                Some(0),
                Outcome::Llm(Box::new(LlmResponse {
                    model: "m1".to_string(),
                    content: None,
                    tool_calls: vec![delete_call("tc-1", "7"), delete_call("tc-2", "9")],
                    finish_reason: Some("tool_calls".to_string()),
                    usage: None,
                    cost: None,
                    images: vec![],
                    reasoning: None,
                })),
            ),
            &system(),
        );
        drive(&mut agg, finished).await;
        agg
    }

    fn delete_call(id: &str, issue: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: "sentry__delete_issue".to_string(),
                arguments: serde_json::json!({ "id": issue }).to_string(),
            },
        }
    }

    fn resume(interrupt_id: &str, approved: bool) -> CommandPayload {
        CommandPayload::ResumeInterrupt {
            interrupt_id: interrupt_id.to_string(),
            payload: serde_json::json!({ "approved": approved }),
        }
    }

    #[tokio::test]
    async fn parallel_calls_are_asked_about_one_at_a_time() {
        let mut agg = two_calls_awaiting_approval().await;
        assert!(
            agg.state.open_interrupt("mcp-approve:tc-1").is_some(),
            "the first call is asked about; open: {:?}",
            agg.state.open_interrupts
        );
        assert!(
            agg.state.tool_call("tc-1").is_none() && agg.state.tool_call("tc-2").is_none(),
            "nothing runs while a question is open"
        );

        let events = dispatch(&mut agg, resume("mcp-approve:tc-1", true), &system());
        drive(&mut agg, events).await;
        assert!(
            agg.state.tool_call("tc-1").is_some(),
            "the approved call is dispatched"
        );
        assert!(
            agg.state.open_interrupt("mcp-approve:tc-2").is_some(),
            "and the next question follows it; open: {:?}",
            agg.state.open_interrupts
        );

        let events = dispatch(
            &mut agg,
            CommandPayload::settle(
                EffectKind::ToolCall,
                "tc-1".to_string(),
                Some(0),
                Outcome::Tool {
                    result: StoredResult::text("deleted issue 7"),
                },
            ),
            &system(),
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e.payload, EventPayload::DecisionDispatched(_))),
            "a result lands, but its decision waits for the open question"
        );

        let events = dispatch(&mut agg, resume("mcp-approve:tc-2", false), &system());
        drive(&mut agg, events).await;

        assert!(
            agg.state.open_interrupts.is_empty(),
            "a call already away must not be asked about again; open: {:?}",
            agg.state.open_interrupts
        );
        assert!(
            agg.state.tool_call("tc-2").is_none(),
            "the declined call never runs"
        );
        let prompt = last_prompt(&agg).expect("the model is prompted with both outcomes");
        let answers: Vec<(&str, String)> = prompt
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| {
                (
                    m.tool_call_id.as_deref().unwrap_or_default(),
                    m.content
                        .as_ref()
                        .and_then(Content::text)
                        .unwrap_or_default()
                        .to_string(),
                )
            })
            .collect();
        assert_eq!(
            answers.len(),
            2,
            "every call the model made is answered exactly once; got {answers:?}"
        );
        assert!(answers
            .iter()
            .any(|(id, text)| *id == "tc-1" && text.contains("deleted issue 7")));
        assert!(answers
            .iter()
            .any(|(id, text)| *id == "tc-2" && text.contains("declined")));
    }

    fn last_prompt(agg: &SessionAggregate) -> Option<Vec<crate::protocol::Message>> {
        agg.state
            .effects_of(EffectKind::LlmCall)
            .filter(|e| e.id != "call-1")
            .last()
            .and_then(|e| e.llm())
            .map(|c| c.prompt.clone())
    }

    #[tokio::test]
    async fn configured_client_message_still_carries_a_proposal() {
        let wire = wire_request_for_client_message(Some(config("m1"))).await;
        assert!(
            wire["proposed"]["actions"]
                .as_array()
                .is_some_and(|a| !a.is_empty()),
            "a configured client.message proposes the LLM continuation; got {}",
            wire["proposed"]
        );
    }
}
