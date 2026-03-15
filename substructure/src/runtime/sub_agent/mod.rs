//! Event handler that spawns child sessions for sub-agent requests
//! and handles cross-session tool call resolution.

use std::sync::Arc;

use async_trait::async_trait;

use crate::runtime::aggregate::{execute, ConflictRetry, DomainEvent, EventHandler, ExecuteInput};
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::CommandPayload;
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::SessionState;

pub struct SubAgentHandler {
    store: Arc<dyn EventStore>,
}

impl SubAgentHandler {
    pub fn new(store: Arc<dyn EventStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl EventHandler<SessionState> for SubAgentHandler {
    async fn on_event(&self, event: &DomainEvent<SessionState>) {
        match &event.payload {
            // SubAgent requested → spawn child session
            EventPayload::SubAgentRequested(req) => {
                let child_session_id = req.session_id;
                let agent_id = req.agent_id.clone();

                let auth = match event.derived.as_ref().and_then(|d| d.auth.as_ref()) {
                    Some(auth) => auth.clone(),
                    None => return,
                };

                // Build child ancestry: parent's ancestry + parent session id
                let mut ancestry = event
                    .derived
                    .as_ref()
                    .map(|d| d.ancestry.clone())
                    .unwrap_or_default();
                ancestry.push(event.aggregate_id);

                // Create child session
                let create_result = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: child_session_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::CreateSession {
                            agent_id,
                            auth,
                            ancestry,
                            worker_retry: req.retry.clone(),
                        },
                        span: event.span.child("create_sub_agent"),
                    },
                    &ConflictRetry::default(),
                )
                .await;

                if create_result.is_err() {
                    let _ = execute::<SessionState>(
                        &*self.store,
                        ExecuteInput {
                            aggregate_id: event.aggregate_id,
                            tenant_id: event.tenant_id.clone(),
                            command: CommandPayload::FailSubAgent {
                                session_id: child_session_id,
                                error: "failed to create child session".to_string(),
                                retryable: false,
                            },
                            span: event.span.child("fail_sub_agent"),
                        },
                        &ConflictRetry::default(),
                    )
                    .await;
                    return;
                }

                // Confirm start on the parent
                let _ = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: event.aggregate_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::StartSubAgent {
                            session_id: child_session_id,
                        },
                        span: event.span.child("start_sub_agent"),
                    },
                    &ConflictRetry::default(),
                )
                .await;
            }

            // Send message to another session
            EventPayload::SessionMessageRequested(req) => {
                let _ = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: req.target_session_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::SendMessage {
                            message: req.message.clone(),
                            stream: false,
                            turn_id: None,
                        },
                        span: event.span.child("send_session_message"),
                    },
                    &ConflictRetry::default(),
                )
                .await;
            }

            // Tool call resolution → complete tool call on target session
            EventPayload::ToolCallResolutionRequested(req) => {
                let _ = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: req.target_session_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::CompleteToolCall {
                            tool_call_id: req.tool_call_id.clone(),
                            result: req.result.clone(),
                            worker_state: None,
                        },
                        span: event.span.child("resolve_tool_call"),
                    },
                    &ConflictRetry::default(),
                )
                .await;
            }

            // Child session done → notify parent
            EventPayload::SessionDone(done) => {
                let derived = match event.derived.as_ref() {
                    Some(d) => d,
                    None => return,
                };
                let parent_id = match derived.ancestry.last() {
                    Some(&id) => id,
                    None => return, // root session, nothing to notify
                };
                let agent_id = derived.agent_id.clone().unwrap_or_default();
                let turn_id = derived.turn_id.clone().unwrap_or_default();

                let _ = execute::<SessionState>(
                    &*self.store,
                    ExecuteInput {
                        aggregate_id: parent_id,
                        tenant_id: event.tenant_id.clone(),
                        command: CommandPayload::CompleteSubAgentTurn {
                            session_id: event.aggregate_id,
                            agent_id,
                            turn_id,
                            artifacts: done.artifacts.clone(),
                        },
                        span: event.span.child("sub_agent_turn_complete"),
                    },
                    &ConflictRetry::default(),
                )
                .await;
            }

            _ => {}
        }
    }
}
