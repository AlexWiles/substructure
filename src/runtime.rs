use std::sync::Arc;

use async_trait::async_trait;
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use rand::RngExt;
use uuid::Uuid;

use crate::command::{SessionCommand, SessionError};
use crate::event::*;
use crate::session::SessionState;
use crate::store::{EventStore, InMemoryEventStore, StoreError, Version};

// ---------------------------------------------------------------------------
// LlmClient trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait LlmClient: Send + Sync + 'static {
    async fn call(&self, request: &LlmRequest) -> Result<LlmResponse, String>;
}

// ---------------------------------------------------------------------------
// Effect — returned by react() to drive side effects
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Effect {
    Command(SessionCommand),
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

// ---------------------------------------------------------------------------
// SessionMessage — commands from API + events from dispatcher
// ---------------------------------------------------------------------------

pub enum SessionMessage {
    Execute(
        SpanContext,
        SessionCommand,
        RpcReplyPort<Result<Vec<Event>, RuntimeError>>,
    ),
    Cast(SpanContext, SessionCommand),
    GetState(RpcReplyPort<SessionState>),
    Events(Vec<Arc<Event>>),
}

// ---------------------------------------------------------------------------
// SessionActor — merged session + reactor
// ---------------------------------------------------------------------------

pub struct SessionActor;

pub struct SessionActorState {
    pub session_id: Uuid,
    pub session: SessionState,
    pub store: Arc<InMemoryEventStore>,
    pub client: Arc<dyn LlmClient>,
}

impl SessionActorState {
    pub fn new(
        session_id: Uuid,
        store: Arc<InMemoryEventStore>,
        client: Arc<dyn LlmClient>,
    ) -> Self {
        SessionActorState {
            session_id,
            session: SessionState::new(session_id),
            store,
            client,
        }
    }
}

async fn execute(
    actor_state: &mut SessionActorState,
    span: SpanContext,
    cmd: SessionCommand,
) -> Result<Vec<Event>, RuntimeError> {
    let payloads = actor_state.session.handle(cmd)?;
    if payloads.is_empty() {
        return Ok(vec![]);
    }

    // Use local stream_version — not store.version() — so the version check
    // catches writes from other nodes that this actor hasn't seen yet.
    let version = Version(actor_state.session.stream_version);
    let events = actor_state
        .store
        .append(actor_state.session_id, version, span, payloads)
        .await?;

    // Apply events to in-memory state immediately (so GetState is fresh).
    // React happens later when the dispatcher delivers these events back.
    for event in &events {
        actor_state.session.apply(event);
    }

    Ok(events)
}

// ---------------------------------------------------------------------------
// React — decides what effects to emit for an event
// ---------------------------------------------------------------------------

fn new_call_id() -> String {
    Uuid::new_v4().to_string()
}

async fn react(
    session_id: Uuid,
    state: &mut SessionState,
    client: &dyn LlmClient,
    event: &Event,
) -> Vec<Effect> {
    match &event.payload {
        EventPayload::MessageUser(_) => {
            if state.active_llm_call().is_some() {
                // dirty flag already set by apply()
                println!(
                    "[session:{}] MessageUser -> LLM call already pending, dirty",
                    session_id
                );
                vec![]
            } else {
                match state.build_llm_request() {
                    Some(request) => {
                        println!(
                            "[session:{}] MessageUser -> requesting LLM call",
                            session_id
                        );
                        vec![Effect::Command(SessionCommand::RequestLlmCall {
                            call_id: new_call_id(),
                            request,
                        })]
                    }
                    None => vec![],
                }
            }
        }

        EventPayload::LlmCallRequested(payload) => {
            println!(
                "[session:{}] LlmCallRequested [{}] -> calling LLM client",
                session_id, payload.call_id,
            );
            match client.call(&payload.request).await {
                Ok(response) => vec![Effect::Command(
                    SessionCommand::CompleteLlmCall {
                        call_id: payload.call_id.clone(),
                        response,
                    },
                )],
                Err(error) => vec![Effect::Command(SessionCommand::FailLlmCall {
                    call_id: payload.call_id.clone(),
                    error,
                })],
            }
        }

        EventPayload::LlmCallCompleted(payload) => {
            if state.dirty {
                println!(
                    "[session:{}] LlmCallCompleted [{}] -> stale (dirty), re-triggering",
                    session_id, payload.call_id,
                );
                state.dirty = false;
                match state.build_llm_request() {
                    Some(request) => {
                        vec![Effect::Command(SessionCommand::RequestLlmCall {
                            call_id: new_call_id(),
                            request,
                        })]
                    }
                    None => vec![],
                }
            } else {
                let (content, tool_calls, token_count) =
                    extract_assistant_message(&payload.response);

                println!(
                    "[session:{}] LlmCallCompleted [{}] -> sending assistant message",
                    session_id, payload.call_id,
                );

                let mut effects =
                    vec![Effect::Command(SessionCommand::SendAssistantMessage {
                        content,
                        tool_calls: tool_calls.clone(),
                        token_count,
                    })];

                for tc in &tool_calls {
                    effects.push(Effect::Command(SessionCommand::RequestToolCall {
                        tool_call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    }));
                }

                effects
            }
        }

        EventPayload::LlmCallErrored(payload) => {
            if state.dirty {
                println!(
                    "[session:{}] LlmCallErrored [{}] -> dirty, re-triggering",
                    session_id, payload.call_id,
                );
                state.dirty = false;
                match state.build_llm_request() {
                    Some(request) => {
                        vec![Effect::Command(SessionCommand::RequestLlmCall {
                            call_id: new_call_id(),
                            request,
                        })]
                    }
                    None => vec![],
                }
            } else {
                println!(
                    "[session:{}] LlmCallErrored [{}] -> no action",
                    session_id, payload.call_id,
                );
                vec![]
            }
        }

        _ => vec![],
    }
}

fn extract_assistant_message(
    response: &LlmResponse,
) -> (Option<String>, Vec<ToolCall>, Option<u32>) {
    match response {
        LlmResponse::OpenAi(resp) => {
            let choice = &resp.choices[0];
            let content = choice.message.content.clone();
            let tool_calls = choice
                .message
                .tool_calls
                .as_ref()
                .map(|tcs| {
                    tcs.iter()
                        .map(|tc| ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let token_count = resp.usage.as_ref().map(|u| u.total_tokens);
            (content, tool_calls, token_count)
        }
    }
}

// ---------------------------------------------------------------------------
// Effect delivery — sends Cast to self
// ---------------------------------------------------------------------------

fn child_span(parent: &SpanContext) -> SpanContext {
    SpanContext {
        trace_id: parent.trace_id,
        span_id: rand::rng().random(),
        parent_span_id: Some(parent.span_id),
        trace_flags: parent.trace_flags,
        trace_state: parent.trace_state.clone(),
    }
}

fn deliver_effect(myself: &ActorRef<SessionMessage>, span: &SpanContext, effect: Effect) {
    match effect {
        Effect::Command(cmd) => {
            let _ = myself.send_message(SessionMessage::Cast(child_span(span), cmd));
        }
    }
}

// ---------------------------------------------------------------------------
// Actor implementation
// ---------------------------------------------------------------------------

impl Actor for SessionActor {
    type Msg = SessionMessage;
    type State = SessionActorState;
    type Arguments = SessionActorState;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(args)
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            SessionMessage::Execute(span, cmd, reply) => {
                let result = execute(state, span, cmd).await;
                let _ = reply.send(result);
            }
            SessionMessage::Cast(span, cmd) => {
                if let Err(e) = execute(state, span, cmd).await {
                    eprintln!("session cast error: {e}");
                }
            }
            SessionMessage::Events(events) => {
                for event in &events {
                    state.session.apply(event);
                    // React if new (skip events already reacted to)
                    let reacted = state
                        .session
                        .last_reacted
                        .is_some_and(|seq| event.sequence <= seq);
                    if !reacted {
                        let effects = react(
                            state.session_id,
                            &mut state.session,
                            &*state.client,
                            event,
                        )
                        .await;
                        for effect in effects {
                            deliver_effect(&myself, &event.span, effect);
                        }
                        state.session.last_reacted = Some(event.sequence);
                    }
                }
            }
            SessionMessage::GetState(reply) => {
                let _ = reply.send(state.session.clone());
            }
        }
        Ok(())
    }
}
