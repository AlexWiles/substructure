use std::sync::Arc;

use ractor::{ActorRef, RpcReplyPort};
use uuid::Uuid;

use crate::domain::agent::AgentConfig;
use crate::domain::aggregate::DomainEvent;
use crate::domain::event::{ClientIdentity, CompletionDelivery, SpanContext};
use crate::domain::session::{AgentState, CommandPayload, SessionCommand};

use super::event_store::{Event, StoreError};
use super::session_client::Notification;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Session(#[from] crate::domain::session::SessionError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("actor call failed: {0}")]
    ActorCall(String),
    #[error("unknown LLM client: {0}")]
    UnknownLlmClient(String),
    #[error("unknown agent: {0}")]
    UnknownAgent(String),
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
    GetState(RpcReplyPort<AgentState>),
    Events(Vec<Arc<DomainEvent<AgentState>>>),
    /// Timer-triggered or scheduler-triggered wake.
    Wake,
    /// Cancel this session (used by parent to cancel sub-agent).
    Cancel,
    /// Set client-provided tools (from AG-UI RunAgentInput).
    SetClientTools(Vec<crate::domain::openai::Tool>),
    /// Transient notification — broadcast to observers, never persisted.
    Notify(Arc<Notification>),
}

// ---------------------------------------------------------------------------
// SessionInit — what the runtime needs to start a session
// ---------------------------------------------------------------------------

pub struct SessionInit {
    pub agent: AgentConfig,
    pub auth: ClientIdentity,
    pub on_done: Option<CompletionDelivery>,
    pub span: SpanContext,
}

// ---------------------------------------------------------------------------
// RuntimeMessage — commands handled by the RuntimeActor
// ---------------------------------------------------------------------------

pub enum RuntimeMessage {
    StartSession(
        Uuid,
        Box<SessionInit>,
        RpcReplyPort<Result<SessionHandle, RuntimeError>>,
    ),
    RunSubAgent(SubAgentRequest),
    WakeAggregate {
        aggregate_id: Uuid,
        aggregate_type: String,
        tenant_id: String,
    },
    /// Find-or-start the aggregate actor, then deliver a command.
    DeliverToSession {
        session_id: Uuid,
        payload: CommandPayload,
        span: SpanContext,
    },
}

pub struct SubAgentRequest {
    pub session_id: Uuid,
    pub agent_name: String,
    pub message: String,
    pub auth: ClientIdentity,
    pub delivery: CompletionDelivery,
    pub span: SpanContext,
    pub token_budget: Option<u64>,
    pub stream: bool,
}

// ---------------------------------------------------------------------------
// SessionHandle — per-session interface
// ---------------------------------------------------------------------------

pub struct SessionHandle {
    pub session_id: Uuid,
    /// The trace_id assigned to this session on creation.
    pub trace_id: Option<crate::domain::span::TraceId>,
    pub(super) session_client: ActorRef<SessionMessage>,
}

impl SessionHandle {
    pub async fn send_command(&self, cmd: SessionCommand) -> Result<Vec<Arc<Event>>, RuntimeError> {
        let result =
            ractor::call_t!(self.session_client, SessionMessage::Execute, 5000, cmd)
                .map_err(|e| RuntimeError::ActorCall(e.to_string()))?;
        result
    }

    pub async fn get_state(&self) -> AgentState {
        ractor::call_t!(self.session_client, SessionMessage::GetState, 5000)
            .expect("failed to query session state")
    }

    pub fn shutdown(self) {
        self.session_client.stop(None);
    }
}
