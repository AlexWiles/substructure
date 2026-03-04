use std::sync::Arc;

use ractor::{Actor, ActorProcessingErr, ActorRef};
use uuid::Uuid;

use crate::runtime::aggregate::{Aggregate, DomainEvent};
use crate::runtime::aggregate::actor::{AggregateError, AggregateMessage};
use crate::runtime::config::ClientIdentity;
use crate::runtime::event_store::EventStore;
use crate::runtime::span::SpanContext;
use crate::runtime::types::{RuntimeError, RuntimeMessage, SessionMessage};
use crate::runtime::Runtime;
use super::state::SessionState;

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
        span: SpanContext,
    },
}

// ---------------------------------------------------------------------------
// SessionUpdate — what observers receive
// ---------------------------------------------------------------------------

/// Distinguishes persisted domain events from ephemeral notifications.
pub enum SessionUpdate {
    Event(Box<DomainEvent<SessionState>>),
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
    tenant_id: String,
    core: Aggregate<SessionState>,
    on_event: Option<OnSessionUpdate>,
    runtime: ActorRef<RuntimeMessage>,
}

pub struct SessionClientArgs {
    pub session_id: Uuid,
    pub auth: ClientIdentity,
    pub aggregate_actor_id: Uuid,
    pub store: Arc<dyn EventStore>,
    pub on_event: Option<OnSessionUpdate>,
    pub runtime: ActorRef<RuntimeMessage>,
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
                Aggregate::new(SessionState::new(args.session_id))
            }),
            Err(_) => Aggregate::new(SessionState::new(args.session_id)),
        };

        let group = super::routing::session_group(args.session_id);
        ractor::pg::join(group, vec![myself.get_cell()]);

        let observer_group = super::routing::session_observer_group(args.session_id);
        ractor::pg::join(observer_group, vec![myself.get_cell()]);

        Ok(SessionClientState {
            session_id: args.session_id,
            tenant_id: args.auth.tenant_id.clone(),
            core,
            on_event: args.on_event,
            runtime: args.runtime,
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
                // Find or create the aggregate actor
                let cell = match Runtime::ensure_aggregate(
                    &state.runtime,
                    state.session_id,
                    "session",
                    &state.tenant_id,
                )
                .await
                {
                    Ok(cell) => cell,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return Ok(());
                    }
                };

                let actor: ActorRef<AggregateMessage<SessionState>> = cell.into();
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
            }
            SessionMessage::Events(typed_events) => {
                for typed in &typed_events {
                    state
                        .core
                        .apply(&typed.payload, typed.sequence, typed.occurred_at);
                    if let Some(f) = &state.on_event {
                        f(&SessionUpdate::Event(Box::new(typed.as_ref().clone())));
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
