use std::sync::Arc;

use ractor::{ActorRef, RpcReplyPort};
use uuid::Uuid;

use crate::runtime::aggregate::DomainEvent;
use crate::runtime::config::ClientIdentity;
use crate::runtime::session::types::CompletionDelivery;
use crate::runtime::session::{SessionCommand, SessionError, SessionState};
use crate::runtime::span::{SpanContext, TraceId};

use super::event_store::{Event, StoreError};
use super::session::client::Notification;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("actor call failed: {0}")]
    ActorCall(String),
    #[error("unknown LLM client: {0}")]
    UnknownLlmClient(String),
    #[error("session not found")]
    SessionNotFound,
}

// ---------------------------------------------------------------------------
// SessionMessage — used by session clients and process group routing
// ---------------------------------------------------------------------------

pub enum SessionMessage {
    Execute(
        SessionCommand,
        RpcReplyPort<Result<Vec<Arc<Event>>, RuntimeError>>,
    ),
    Cast(SessionCommand),
    GetState(RpcReplyPort<SessionState>),
    Events(Vec<Arc<DomainEvent<SessionState>>>),
    /// Timer-triggered or scheduler-triggered wake.
    Wake,
    /// Cancel this session (used by parent to cancel sub-agent).
    Cancel,
    /// Transient notification — broadcast to observers, never persisted.
    Notify(Arc<Notification>),
}

// ---------------------------------------------------------------------------
// SessionInit — what the runtime needs to start a session
// ---------------------------------------------------------------------------

pub struct SessionInit {
    pub agent_name: String,
    pub auth: ClientIdentity,
    pub on_done: Option<CompletionDelivery>,
    pub span: SpanContext,
    pub stream: bool,
}

// ---------------------------------------------------------------------------
// SubAgentRequest
// ---------------------------------------------------------------------------

pub struct SubAgentRequest {
    pub session_id: Uuid,
    pub agent_name: String,
    pub message: String,
    pub auth: ClientIdentity,
    pub delivery: CompletionDelivery,
    pub span: SpanContext,
    pub stream: bool,
}

// ---------------------------------------------------------------------------
// SupervisorMessage — uninhabited (supervisor processes no messages)
// ---------------------------------------------------------------------------

pub enum SupervisorMessage {}

// ---------------------------------------------------------------------------
// SessionHandle — per-session interface
// ---------------------------------------------------------------------------

pub struct SessionHandle {
    pub session_id: Uuid,
    /// The trace_id assigned to this session on creation.
    pub trace_id: Option<TraceId>,
    pub(super) session_client: ActorRef<SessionMessage>,
}

impl SessionHandle {
    pub async fn send_command(&self, cmd: SessionCommand) -> Result<Vec<Arc<Event>>, RuntimeError> {
        let result = ractor::call_t!(self.session_client, SessionMessage::Execute, 5000, cmd)
            .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;
        result
    }

    pub async fn get_state(&self) -> SessionState {
        ractor::call_t!(self.session_client, SessionMessage::GetState, 5000)
            .expect("failed to query session state")
    }

    pub fn shutdown(self) {
        self.session_client.stop(None);
    }
}
