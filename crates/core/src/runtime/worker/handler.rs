use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::protocol::{DecisionResponse, DecisionTrigger, Message};
use crate::runtime::event_store::{EventFilter, EventStore, Seq};
use crate::runtime::processor::{
    EventProcessor, EventProcessorRunner, EventProcessorRunnerConfig, ProcessorCursorStore,
    ProcessorError,
};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::propose::propose;
use crate::runtime::session::state::SessionState;
use crate::runtime::session::wire::to_wire_trigger;
use crate::runtime::session::SessionEvent;

use super::{AgentDirectory, WorkerDecisionRequest, WorkerQueue};

/// A channel's own proposal for a decision, added at delivery. Takes the
/// proposal so far and returns it amended, replaced, or untouched. Must be a
/// pure function of its inputs, so a redelivery proposes the same thing.
pub trait ChannelProposer: Send + Sync {
    fn propose(
        &self,
        session_id: &str,
        trigger: &DecisionTrigger,
        state: &SessionState,
        events: &[SessionEvent],
        transcript: &[Message],
        proposed: DecisionResponse,
    ) -> DecisionResponse;
}

struct WorkerDecisionProjection {
    store: Arc<dyn EventStore>,
    queue: Arc<dyn WorkerQueue>,
    agents: Arc<dyn AgentDirectory>,
    proposers: Vec<Arc<dyn ChannelProposer>>,
}

impl WorkerDecisionProjection {
    fn new(
        store: Arc<dyn EventStore>,
        queue: Arc<dyn WorkerQueue>,
        agents: Arc<dyn AgentDirectory>,
        proposers: Vec<Arc<dyn ChannelProposer>>,
    ) -> Self {
        Self {
            store,
            queue,
            agents,
            proposers,
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
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let projection = Arc::new(WorkerDecisionProjection::new(
        store.clone(),
        queue,
        agents,
        proposers,
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

/// Builds the wire request for a decision event: the tree comes rewound from
/// `load_as_of`, statuses come stamped on the event's meta. A load error must
/// propagate (the runner retries the batch) — the queue never redelivers a
/// dropped decision.
async fn extract(
    store: &dyn EventStore,
    agents: &dyn AgentDirectory,
    proposers: &[Arc<dyn ChannelProposer>],
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

    // The trigger's single stored copy rode the decision's queued event into
    // state; it is immutable per decision. Absent means the decision settled
    // after this event — nothing left to deliver.
    let Some(trigger) = session
        .state
        .worker_decision(&req.id)
        .map(|d| d.trigger.clone())
    else {
        return Ok(None);
    };

    let state = session.state.rewind(event.seq, meta.head_id.as_deref());

    let message_tree = state.message_tree();

    let transcript = message_tree
        .head_id
        .as_deref()
        .map(|h| message_tree.path_to(h))
        .unwrap_or_default();
    // Call entries are keyed by immutable prompt/spec; the as-of path picks
    // the as-of subset even from the current map.
    let open_llm_calls = state.open_llm_calls(&message_tree);

    let pending_calls = meta.pending_work(&req.id);
    let worker_state = state.resolve_state_for(message_tree.head_id.as_deref());
    let agent_config = state.resolve_agent_for(message_tree.head_id.as_deref());

    let trigger = to_wire_trigger(trigger, &transcript, &message_tree, &open_llm_calls);
    let proposed = propose(
        &trigger,
        &transcript,
        &open_llm_calls,
        pending_calls,
        agent_config.as_ref(),
        &req.id,
    )
    // `session.start` is the one trigger the engine cannot derive from the
    // session: the config is the app's declaration, not the session's history.
    // Seeding it here makes the declared identity the proposal, so an
    // echo-worker inherits it and an engine-hosted agent starts configured.
    // An agent that seeds nothing gets no proposal, and its worker authors the
    // config exactly as it would have before the file could declare one.
    .or_else(|| {
        matches!(trigger, DecisionTrigger::SessionStart)
            .then(|| agents.agent(&event.tenant_id, agent_id)?.config)
            .flatten()
            .map(|config| DecisionResponse {
                agent: Some(config),
                ..Default::default()
            })
    })
    .unwrap_or_default();
    // The current turn's events, as of this delivery. Bounded by the turn's
    // start, so the read costs one turn and not the whole session.
    let turn_events: Vec<SessionEvent> = match (proposers.is_empty(), state.turn_started_seq) {
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
    // Give no events rather than the wrong turn's.
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
    let proposed = proposers.iter().fold(proposed, |p, proposer| {
        proposer.propose(
            &event.session_id,
            &trigger,
            &state,
            &turn_events,
            &transcript,
            p,
        )
    });

    Ok(Some(WorkerDecisionRequest {
        session_id: event.session_id.clone(),
        decision_id: req.id.clone(),
        agent_id: agent_id.clone(),
        identity: owner.clone(),
        trigger,
        proposed,
        state: worker_state,
        agent: agent_config,
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

    use super::{extract, AgentDirectory};
    use crate::protocol::OwnerKind;
    use crate::protocol::{
        AgentConfig, ClientMessage, ClientPayload, Content, DraftMessage, RetryPolicy, Role,
        SessionOwner,
    };
    use crate::runtime::event_store::{
        AppendInput, BroadcastBus, EventBus, EventFilter, EventStore, EventTap, StoreError,
    };
    use crate::runtime::session::command::{CommandPayload, TurnTarget};
    use crate::runtime::session::events::EventPayload;
    use crate::runtime::session::state::SessionState;
    use crate::runtime::session::{CommitContext, SessionAggregate, SessionEvent};
    use crate::runtime::span::SpanContext;
    use crate::runtime::worker::directory::{AgentEntry, StaticAgentDirectory};
    use crate::runtime::worker::EmptyAgentDirectory;
    use crate::runtime::Caller;

    /// Read-only store serving one hydrated session, as `extract` loads it.
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
        }
    }

    /// Drive a real session to a live client.message decision (session.start
    /// settled with `agent`), run `extract` on its promotion event, and return
    /// the serialized wire request.
    fn config(model: &str) -> AgentConfig {
        AgentConfig {
            llm: Some("claude".to_string()),
            model: model.to_string(),
            system: None,
            retry: None,
            tools: Vec::new(),
            sub_agents: Vec::new(),
            mcp: Vec::new(),
            defer_tools: None,
            announce_mcp: Default::default(),
        }
    }

    /// A directory declaring `agent-1` with `config`, engine-hosted.
    fn directory(config: AgentConfig) -> StaticAgentDirectory {
        StaticAgentDirectory::new(
            "tenant-a".to_string(),
            std::collections::BTreeMap::from([(
                "agent-1".to_string(),
                AgentEntry {
                    config: Some(config),
                    worker: None,
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
                    kind: OwnerKind::Frontend,
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
        let req = extract(&store, &EmptyAgentDirectory, &[], event)
            .await
            .expect("extract succeeds")
            .expect("a deliverable request");
        serde_json::to_value(&req).expect("wire request serializes")
    }

    /// The `session.start` request for a fresh session, as `agents` declares it.
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
                    kind: OwnerKind::Frontend,
                    tenant_id: "tenant-a".to_string(),
                    id: Some("user-1".to_string()),
                    metadata: HashMap::new(),
                },
                ancestry: vec![],
                worker_retry: RetryPolicy::no_retry(),
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
        let req = extract(&store, agents, &[], event)
            .await
            .expect("extract succeeds")
            .expect("a deliverable request");
        serde_json::to_value(&req).expect("wire request serializes")
    }

    /// The declared config *is* the `session.start` proposal, so an echo-worker
    /// inherits it and an engine-hosted agent starts configured.
    #[tokio::test]
    async fn session_start_is_seeded_with_the_declared_config() {
        let wire = wire_request_for_session_start(&directory(config("m1"))).await;
        assert_eq!(wire["proposed"]["agent"]["model"], "m1");
        assert_eq!(wire["proposed"]["agent"]["llm"], "claude");
    }

    /// Nothing declared, nothing seeded: the proposal stays empty, so a
    /// proposed-first worker still reaches its own config branch.
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

    /// Only `session.start` is seeded: every other trigger's proposal is
    /// derived from the session, and a declaration must not overwrite it.
    #[tokio::test]
    async fn a_non_start_trigger_is_not_seeded() {
        let wire = wire_request_for_client_message(None).await;
        assert!(
            wire["proposed"]["agent"].is_null(),
            "the directory does not reach a client.messages proposal; got {}",
            wire["proposed"]
        );
    }

    /// `proposed` is always present — but never a silent no-op. With no config
    /// the engine cannot author the turn, and an empty proposal echoed by a
    /// blind worker settles the decision as a no-op: the user's message is
    /// dropped with no model call, no error, no terminal event. The proposal
    /// for this case is an interrupt (worker-echoed, like `llm.failed`), so an
    /// echoing worker pauses the session loudly instead.
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
