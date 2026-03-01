use std::sync::Arc;

use ractor::{Actor, ActorProcessingErr, ActorRef};
use uuid::Uuid;

use super::aggregate_actor::{AggregateError, AggregateMessage};
use super::event_store::EventStore;
use super::RuntimeError;
use crate::domain::aggregate::{Aggregate, DomainEvent};
use crate::domain::event::ClientIdentity;
use crate::domain::session::AgentState;

use super::SessionMessage;

// ---------------------------------------------------------------------------
// Notification — transient signals, never persisted
// ---------------------------------------------------------------------------

/// Transient signals broadcast to observers but never persisted.
#[derive(Debug, Clone)]
pub enum Notification {
    LlmStreamChunk {
        call_id: String,
        chunk_index: u32,
        text: String,
        span: crate::domain::span::SpanContext,
    },
}

// ---------------------------------------------------------------------------
// SessionUpdate — what observers receive
// ---------------------------------------------------------------------------

/// Distinguishes persisted domain events from ephemeral notifications.
pub enum SessionUpdate {
    /// A persisted, replayable domain event.
    Event(DomainEvent<AgentState>),
    /// A transient notification — never persisted, for real-time observers only.
    Notification(Arc<Notification>),
}

/// Callback invoked for each update (event or notification).
pub type OnSessionUpdate = Box<dyn Fn(&SessionUpdate) + Send + Sync>;

// ---------------------------------------------------------------------------
// SessionClientActor
// ---------------------------------------------------------------------------

pub struct SessionClientActor;

pub struct SessionClientState {
    session_id: Uuid,
    core: Aggregate<AgentState>,
    on_event: Option<OnSessionUpdate>,
}

pub struct SessionClientArgs {
    pub session_id: Uuid,
    pub auth: ClientIdentity,
    pub aggregate_actor_id: Uuid,
    pub store: Arc<dyn EventStore>,
    pub on_event: Option<OnSessionUpdate>,
}

impl Actor for SessionClientActor {
    type Msg = SessionMessage;
    type State = SessionClientState;
    type Arguments = SessionClientArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let core = match args.store.load(args.session_id, &args.auth.tenant_id).await {
            Ok(load) => serde_json::from_value(load.snapshot).unwrap_or_else(|_| {
                Aggregate::new(AgentState::new(args.session_id))
            }),
            Err(_) => Aggregate::new(AgentState::new(args.session_id)),
        };

        let group = super::session_group(args.session_id);
        ractor::pg::join(group, vec![myself.get_cell()]);

        let observer_group = super::session_observer_group(args.session_id);
        ractor::pg::join(observer_group, vec![myself.get_cell()]);

        Ok(SessionClientState {
            session_id: args.session_id,
            core,
            on_event: args.on_event,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            SessionMessage::Execute(cmd, reply) => {
                // Forward command to aggregate actor via registry
                let name = super::aggregate_actor_name(state.session_id);
                if let Some(cell) = ractor::registry::where_is(name) {
                    let actor: ActorRef<AggregateMessage<AgentState>> = cell.into();
                    let result = actor
                        .call(
                            |rpc_reply| AggregateMessage::Execute {
                                cmd: cmd.payload,
                                span: cmd.span,
                                occurred_at: cmd.occurred_at,
                                reply: rpc_reply,
                            },
                            Some(ractor::concurrency::Duration::from_millis(5000)),
                        )
                        .await;

                    match result {
                        Ok(ractor::rpc::CallResult::Success(inner)) => {
                            let mapped = inner.map_err(|e| match e {
                                AggregateError::Command(err) => RuntimeError::Session(err),
                                AggregateError::Store(err) => RuntimeError::Store(err),
                            });
                            let _ = reply.send(mapped);
                        }
                        Ok(ractor::rpc::CallResult::Timeout) => {
                            let _ = reply.send(Err(RuntimeError::ActorCall(
                                "aggregate actor call timed out".into(),
                            )));
                        }
                        Ok(ractor::rpc::CallResult::SenderError) => {
                            let _ = reply.send(Err(RuntimeError::ActorCall(
                                "aggregate actor sender error".into(),
                            )));
                        }
                        Err(e) => {
                            tracing::error!(session = %state.session_id, error = %e, "call to aggregate actor failed");
                            let _ = reply.send(Err(RuntimeError::ActorCall(e.to_string())));
                        }
                    }
                } else {
                    let _ = reply.send(Err(RuntimeError::SessionNotFound));
                }
            }
            SessionMessage::Events(typed_events) => {
                for typed in &typed_events {
                    state
                        .core
                        .apply(&typed.payload, typed.sequence, typed.occurred_at);
                    if let Some(f) = &state.on_event {
                        f(&SessionUpdate::Event(typed.as_ref().clone()));
                    }
                }
            }
            SessionMessage::Notify(notification) => {
                if let Some(f) = &state.on_event {
                    f(&SessionUpdate::Notification(notification));
                }
            }
            SessionMessage::GetState(reply) => {
                let _ = reply.send(state.core.state.clone());
            }
            _ => {} // Wake, Cancel, Cast, SetClientTools — not for clients
        }
        Ok(())
    }
}
