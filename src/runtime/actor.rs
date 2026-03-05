use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use ractor::{Actor, ActorProcessingErr, ActorRef, SupervisionEvent};
use uuid::Uuid;

use crate::runtime::config::ClientIdentity;
use crate::runtime::config::{AgentConfig, BudgetPolicyConfig};
use crate::runtime::session::{
    CommandPayload, IncomingMessage, SessionCommand, SessionState, SessionStatus,
};
use crate::runtime::span::SpanContext;

use super::aggregate::actor::{self as aggregate_actor, AggregateActorHandle, AggregateMessage};
use super::aggregate::dispatcher::spawn_aggregate_dispatcher;
use super::budget;
use super::event_store::{AggregateFilter, EventStore};
use super::llm::{LlmProviderTrait, LlmTool, LlmToolFunction};
use super::mcp::{self, McpClient, ToolDefinition};
use super::session::client::{Notification, SessionClientActor, SessionClientArgs};
use super::session::routing::{aggregate_actor_name, notify_observers, session_route};
use super::session::{BudgetActorRef, McpToolEntry, NotifyChunkFn, SessionContext};
use super::types::{RuntimeError, RuntimeMessage, SessionHandle, SessionInit, SubAgentRequest};
use super::wake_scheduler::spawn_wake_scheduler;

// ---------------------------------------------------------------------------
// RuntimeActor — owns providers, spawns aggregate actors directly
// ---------------------------------------------------------------------------

pub(super) struct RuntimeActor;

pub(super) struct RuntimeState {
    pub(super) myself: ActorRef<RuntimeMessage>,
    pub(super) store: Arc<dyn EventStore>,
    pub(super) agents: HashMap<String, AgentConfig>,
    pub(super) llm_provider: Arc<dyn LlmProviderTrait>,
    pub(super) budget_policies: Vec<BudgetPolicyConfig>,
    pub(super) tool_result_max_bytes: Option<usize>,
}

pub(super) struct RuntimeArgs {
    pub(super) store: Arc<dyn EventStore>,
    pub(super) agents: HashMap<String, AgentConfig>,
    pub(super) llm_provider: Arc<dyn LlmProviderTrait>,
    pub(super) budget_policies: Vec<BudgetPolicyConfig>,
    #[cfg(feature = "otel")]
    pub(super) otel: Option<super::config::OtelConfig>,
    pub(super) tool_result_max_bytes: Option<usize>,
}

// ---------------------------------------------------------------------------
// RuntimeState methods
// ---------------------------------------------------------------------------

impl RuntimeState {
    /// Look up a running aggregate actor by session ID and send a command.
    /// Returns `true` if the actor was found and the message was sent.
    fn try_send_to_aggregate(
        &self,
        session_id: Uuid,
        payload: CommandPayload,
        span: SpanContext,
    ) -> bool {
        if let Some(cell) = ractor::registry::where_is(aggregate_actor_name(session_id)) {
            let actor: ActorRef<AggregateMessage<SessionState>> = cell.into();
            let _ = actor.send_message(AggregateMessage::Cast {
                cmd: payload,
                span,
                occurred_at: Utc::now(),
            });
            true
        } else {
            false
        }
    }

    /// Get or spawn a budget actor for the given tenant.
    async fn get_or_spawn_budget_actor(
        &self,
        tenant_id: &str,
    ) -> Result<Option<AggregateActorHandle<budget::BudgetLedger>>, RuntimeError> {
        if self.budget_policies.is_empty() {
            return Ok(None);
        }
        let actor_name = budget::budget_actor_name(tenant_id);
        if let Some(cell) = ractor::registry::where_is(actor_name) {
            return Ok(Some(AggregateActorHandle { actor: cell.into() }));
        }
        let handle = budget::spawn_budget_actor(
            tenant_id.to_string(),
            self.budget_policies.clone(),
            self.store.clone(),
            self.myself.get_cell(),
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("budget: {e}")))?;
        Ok(Some(handle))
    }

    #[tracing::instrument(skip(self, agent), fields(agent = %agent.name))]
    async fn get_or_spawn_mcp_actors(
        &self,
        agent: &AgentConfig,
    ) -> Result<Vec<Arc<dyn McpClient>>, RuntimeError> {
        if agent.mcp_servers.is_empty() {
            return Ok(Vec::new());
        }
        let mut clients: Vec<Arc<dyn McpClient>> = Vec::with_capacity(agent.mcp_servers.len());
        for config in &agent.mcp_servers {
            let name = mcp::mcp_actor_name(&agent.name, &config.name);
            let actor_ref: ActorRef<mcp::McpMessage> =
                if let Some(cell) = ractor::registry::where_is(name) {
                    cell.into()
                } else {
                    mcp::spawn_mcp_actor(&agent.name, config.clone(), self.myself.get_cell())
                        .await
                        .map_err(|e| RuntimeError::ActorCall(format!("mcp {}: {e}", config.name)))?
                };
            let client = mcp::McpActorClient::from_actor(actor_ref)
                .await
                .map_err(|e| RuntimeError::ActorCall(format!("mcp {}: {e}", config.name)))?;
            clients.push(Arc::new(client));
        }
        Ok(clients)
    }

    #[tracing::instrument(skip(self, init), fields(%session_id, trace_id = %init.span.trace_id))]
    pub(super) async fn start_session(
        &self,
        session_id: Uuid,
        init: SessionInit,
    ) -> Result<SessionHandle, RuntimeError> {
        let trace_id = init.span.trace_id;
        let auth = init.auth.clone();
        let agent = init.agent.clone();
        let actor_name = aggregate_actor_name(session_id);
        let already_running = ractor::registry::where_is(actor_name).is_some();

        if !already_running {
            let budget_actor = self.get_or_spawn_budget_actor(&auth.tenant_id).await?;
            let mcp_clients = self.get_or_spawn_mcp_actors(&init.agent).await?;

            let llm_provider = self.llm_provider.clone();
            let agents = self.agents.clone();
            let runtime_ref = self.myself.clone();
            let mcp_for_ctx = mcp_clients.clone();
            let auth_for_ctx = auth.clone();
            let agent_for_ctx = agent.clone();
            let tool_result_max_bytes = self.tool_result_max_bytes;

            let aggregate_handle = aggregate_actor::spawn_aggregate_actor(
                aggregate_actor::AggregateActorArgs {
                    aggregate_id: session_id,
                    store: self.store.clone(),
                    tenant_id: auth.tenant_id.clone(),
                    init: Box::new(SessionState::new),
                    idle_timeout: None,
                    context_init: Box::new(move |state| {
                        let resolved_agent = state.agent.clone().unwrap_or(agent_for_ctx);
                        Box::pin(async move {
                            let mut ctx = build_session_context(
                                session_id,
                                &auth_for_ctx,
                                &mcp_for_ctx,
                                &llm_provider,
                                &agents,
                                Some(&resolved_agent),
                                budget_actor,
                                false,
                                tool_result_max_bytes,
                            );
                            // Wire up send_to_session (find-or-start via runtime)
                            let runtime_for_send = runtime_ref.clone();
                            ctx.send_to_session =
                                Some(Arc::new(move |session_id, payload, span| {
                                    let _ = runtime_for_send.send_message(
                                        RuntimeMessage::DeliverToSession {
                                            session_id,
                                            payload,
                                            span,
                                        },
                                    );
                                }));
                            // Wire up sub-agent spawning
                            let runtime = runtime_ref.clone();
                            ctx.spawn_sub_agent = Some(Arc::new(move |params| {
                                let _ = runtime.send_message(RuntimeMessage::RunSubAgent(
                                    SubAgentRequest {
                                        session_id: params.session_id,
                                        agent_name: params.agent_name,
                                        message: params.message,
                                        auth: params.auth,
                                        delivery: params.delivery,
                                        span: params.span,
                                        stream: params.stream,
                                    },
                                ));
                            }));
                            ctx
                        })
                    }),
                },
                self.myself.get_cell(),
            )
            .await
            .map_err(|e| RuntimeError::ActorCall(format!("aggregate actor spawn: {e}")))?;

            // Check if session needs creation
            let session = aggregate_handle.get_aggregate().await;
            let is_new = session.state.agent.is_none();

            if is_new {
                aggregate_handle
                    .send_command(
                        CommandPayload::CreateSession {
                            agent: Box::new(init.agent),
                            auth: auth.clone(),
                            on_done: init.on_done,
                        },
                        init.span.child("session.create"),
                        Utc::now(),
                    )
                    .await
                    .map_err(|e| RuntimeError::ActorCall(format!("create session: {e}")))?;
            }

            // If resuming a completed sub-agent, deliver result to parent.
            // Deferred via message queue to avoid start_session ↔ wake_aggregate recursion.
            if session.state.status == SessionStatus::Done {
                if let Some(ref delivery) = session.state.on_done {
                    let result =
                        serde_json::to_string(&session.state.artifacts).unwrap_or_default();
                    let _ = self.myself.send_message(RuntimeMessage::DeliverToSession {
                        session_id: delivery.parent_session_id,
                        payload: CommandPayload::CompleteToolCall {
                            tool_call_id: delivery.tool_call_id.clone(),
                            name: delivery.tool_name.clone(),
                            result,
                        },
                        span: init.span.child("sub_agent.deliver"),
                    });
                }
            }
        }

        // Spawn a SessionClientActor for the caller
        let (client, _client_handle) = Actor::spawn(
            None,
            SessionClientActor,
            SessionClientArgs {
                session_id,
                auth,
                aggregate_actor_id: session_id,
                store: self.store.clone(),
                on_event: None,
                runtime: self.myself.clone(),
            },
        )
        .await
        .map_err(|e| RuntimeError::ActorCall(format!("session client startup failed: {e}")))?;

        Ok(SessionHandle {
            session_id,
            trace_id: Some(trace_id),
            session_client: client,
        })
    }

    #[tracing::instrument(skip(self), fields(%aggregate_id, %aggregate_type))]
    async fn wake_aggregate(&self, aggregate_id: Uuid, aggregate_type: &str, tenant_id: &str) {
        match aggregate_type {
            "session" => {
                // If the aggregate actor is already running, send Wake command
                if let Some(cell) = ractor::registry::where_is(aggregate_actor_name(aggregate_id)) {
                    let actor: ActorRef<AggregateMessage<SessionState>> = cell.into();
                    let _ = actor.send_message(AggregateMessage::Cast {
                        cmd: CommandPayload::Wake,
                        span: SpanContext::root().with_name("wake"),
                        occurred_at: Utc::now(),
                    });
                    return;
                }
                // Not running — look up the agent name via list_aggregates and start the session.
                let filter = AggregateFilter {
                    aggregate_ids: Some(vec![aggregate_id]),
                    ..Default::default()
                };
                let results = self.store.list_aggregates(&filter).await;
                let summary = match results.into_iter().next() {
                    Some(s) => s,
                    None => {
                        tracing::warn!(%aggregate_id, "wake: session not found in store");
                        return;
                    }
                };
                let agent_name = match summary.label {
                    Some(name) => name,
                    None => {
                        tracing::warn!(%aggregate_id, "wake: session has no agent label");
                        return;
                    }
                };
                let agent = match self.agents.get(&agent_name) {
                    Some(a) => a.clone(),
                    None => {
                        tracing::warn!(agent = %agent_name, session = %aggregate_id, "wake: unknown agent");
                        return;
                    }
                };
                let auth = ClientIdentity {
                    tenant_id: tenant_id.to_string(),
                    sub: None,
                    attrs: Default::default(),
                };
                let init = SessionInit {
                    agent,
                    auth,
                    on_done: None,
                    span: SpanContext::root().with_name("wake"),
                };
                if let Err(e) = self.start_session(aggregate_id, init).await {
                    tracing::warn!(session = %aggregate_id, error = %e, "wake: failed to start session");
                }
            }
            _ => {
                tracing::debug!(aggregate_type = %aggregate_type, %aggregate_id, "wake: no handler for aggregate type");
            }
        }
    }

    #[tracing::instrument(skip_all, fields(session_id = %req.session_id, agent = %req.agent_name, trace_id = %req.span.trace_id))]
    async fn run_sub_agent(&self, req: SubAgentRequest) -> Result<(), RuntimeError> {
        let agent = self
            .agents
            .get(&req.agent_name)
            .cloned()
            .ok_or_else(|| RuntimeError::UnknownAgent(req.agent_name.clone()))?;

        let msg_span = req.span.child("sub_agent.message");

        let init = SessionInit {
            agent,
            auth: req.auth,
            on_done: Some(req.delivery),
            span: req.span,
        };

        // start_session spawns the aggregate actor and creates the session
        let handle = self.start_session(req.session_id, init).await?;

        // Send user message to the sub-agent
        let _ = handle
            .send_command(SessionCommand {
                span: msg_span,
                occurred_at: Utc::now(),
                payload: CommandPayload::SendMessage {
                    message: IncomingMessage::User {
                        content: req.message,
                    },
                    stream: req.stream,
                },
            })
            .await;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Actor trait implementation
// ---------------------------------------------------------------------------

impl Actor for RuntimeActor {
    type Msg = RuntimeMessage;
    type State = RuntimeState;
    type Arguments = RuntimeArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        // Spawn infrastructure actors (linked to RuntimeActor)
        tracing::debug!("spawning event dispatcher");
        spawn_aggregate_dispatcher::<SessionState>(
            &args.store,
            Arc::new(session_route),
            myself.get_cell(),
        )
        .await
        .map_err(|e| format!("failed to spawn dispatcher: {e}"))?;
        tracing::debug!("spawning wake scheduler");
        spawn_wake_scheduler(args.store.clone(), myself.clone(), myself.get_cell())
            .await
            .map_err(|e| format!("failed to spawn wake scheduler: {e}"))?;

        #[cfg(feature = "otel")]
        if let Some(otel_config) = args.otel {
            tracing::debug!("spawning otel exporter");
            if let Err(e) = super::otel::spawn_otel_exporter(
                &otel_config.endpoint,
                otel_config.service_name,
                myself.get_cell(),
                &*args.store,
            )
            .await
            {
                tracing::warn!(error = %e, "failed to start otel exporter, continuing without it");
            }
        }

        let state = RuntimeState {
            myself: myself.clone(),
            store: args.store.clone(),
            agents: args.agents,
            llm_provider: args.llm_provider,
            budget_policies: args.budget_policies,
            tool_result_max_bytes: args.tool_result_max_bytes,
        };

        Ok(state)
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            RuntimeMessage::StartSession(session_id, init, reply) => {
                let result = state.start_session(session_id, *init).await;
                let _ = reply.send(result);
            }
            RuntimeMessage::RunSubAgent(req) => {
                if let Err(e) = state.run_sub_agent(req).await {
                    tracing::error!(error = %e, "sub-agent error");
                }
            }
            RuntimeMessage::WakeAggregate {
                aggregate_id,
                aggregate_type,
                tenant_id,
            } => {
                state
                    .wake_aggregate(aggregate_id, &aggregate_type, &tenant_id)
                    .await;
            }
            RuntimeMessage::DeliverToSession {
                session_id,
                payload,
                span,
            } => {
                // Find-or-start: wake the aggregate if needed, then deliver
                state.wake_aggregate(session_id, "session", "").await;
                state.try_send_to_aggregate(session_id, payload, span);
            }
            RuntimeMessage::EnsureAggregate {
                aggregate_id,
                aggregate_type,
                tenant_id,
                reply,
            } => {
                if ractor::registry::where_is(aggregate_actor_name(aggregate_id)).is_none() {
                    state
                        .wake_aggregate(aggregate_id, &aggregate_type, &tenant_id)
                        .await;
                }
                let _ = reply.send(Ok(()));
            }
        }
        Ok(())
    }

    async fn handle_supervisor_evt(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: SupervisionEvent,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match &message {
            SupervisionEvent::ActorFailed(who, err) => {
                tracing::error!(actor = ?who.get_name(), error = %err, "child actor failed");
            }
            SupervisionEvent::ActorTerminated(who, _, reason) => {
                if reason.is_some() {
                    tracing::error!(actor = ?who.get_name(), reason = ?reason, "child actor terminated unexpectedly");
                } else {
                    tracing::debug!(actor = ?who.get_name(), "child actor stopped");
                }
            }
            _ => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Build SessionContext — wires runtime resources into the domain context
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_session_context(
    session_id: Uuid,
    auth: &ClientIdentity,
    mcp_clients: &[Arc<dyn McpClient>],
    llm_provider: &Arc<dyn LlmProviderTrait>,
    agents: &HashMap<String, AgentConfig>,
    agent: Option<&AgentConfig>,
    budget_actor: Option<AggregateActorHandle<budget::BudgetLedger>>,
    stream: bool,
    tool_result_max_bytes: Option<usize>,
) -> SessionContext {
    let mcp_tools: HashMap<String, McpToolEntry> = mcp_clients
        .iter()
        .flat_map(|c| {
            let info = c.server_info();
            let server_name = info.name.clone();
            let server_version = info.version.clone();
            c.tools().iter().map(move |t| {
                (
                    t.name.clone(),
                    McpToolEntry {
                        server_name: server_name.clone(),
                        server_version: server_version.clone(),
                    },
                )
            })
        })
        .collect();

    let mut tools: Vec<LlmTool> = mcp_clients
        .iter()
        .flat_map(|c| c.tools().iter().map(|t| t.to_tool()))
        .collect();

    if let Some(agent) = agent {
        for name in &agent.sub_agents {
            if let Some(sub) = agents.get(name) {
                let tool_name = ToolDefinition::sanitized_name(name);
                tools.push(LlmTool {
                    tool_type: "function".to_string(),
                    function: LlmToolFunction {
                        name: tool_name,
                        description: sub.description.clone().unwrap_or_else(|| sub.name.clone()),
                        parameters: serde_json::json!({
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The message to send to the sub-agent"
                                }
                            },
                            "required": ["message"]
                        }),
                    },
                });
            }
        }
    }

    let all_tools = if tools.is_empty() { None } else { Some(tools) };

    let budget_ref = budget_actor.map(|a| BudgetActorRef { inner: Box::new(a) });

    let notify_chunk: NotifyChunkFn = Arc::new(|session_id, call_id, chunk_index, text, span| {
        notify_observers(
            session_id,
            Arc::new(Notification::LlmStreamChunk {
                call_id,
                chunk_index,
                text,
                span,
            }),
        );
    });

    let strategy = agent.map(|a| {
        Arc::from(super::session::strategy::resolve_strategy(&a.strategy))
    });

    SessionContext {
        mcp_tools,
        all_tools,
        session_id,
        auth: auth.clone(),
        stream,
        llm_provider: Some(llm_provider.clone()),
        mcp_clients: mcp_clients.to_vec(),
        agents: agents.clone(),
        client_tools: Vec::new(),
        budget_actor: budget_ref,
        notify_chunk: Some(notify_chunk),
        send_to_session: None,
        spawn_sub_agent: None,
        tool_result_max_bytes,
        strategy,
    }
}
