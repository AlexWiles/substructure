use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use tokio::task::AbortHandle;
use uuid::Uuid;

use std::future::Future;
use std::pin::Pin;

use crate::domain::aggregate::{Aggregate, AggregateState, AggregateStatus, DomainEvent};
use crate::domain::span::SpanContext;
use crate::runtime::event_store::{Event, EventStore, StoreError};

/// Default idle timeout for aggregate actors (5 minutes).
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Execution recording — inline instrumentation for OTel
// ---------------------------------------------------------------------------

pub struct ExecutionRecord {
    pub span_context: SpanContext,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub aggregate_type: &'static str,
    pub aggregate_id: Uuid,
    pub produced_events: Vec<Arc<Event>>,
}

pub type ExecutionRecorder = Arc<dyn Fn(ExecutionRecord) + Send + Sync>;

pub enum AggregateMessage<R: AggregateState> {
    Execute {
        cmd: R::Command,
        span: SpanContext,
        occurred_at: DateTime<Utc>,
        reply: RpcReplyPort<Result<Vec<Arc<Event>>, AggregateError<R::Error>>>,
    },
    Cast {
        cmd: R::Command,
        span: SpanContext,
        occurred_at: DateTime<Utc>,
    },
    GetState(RpcReplyPort<R>),
    GetAggregate(RpcReplyPort<Aggregate<R>>),
    Events(Vec<Arc<DomainEvent<R>>>),
    /// Mutate the context in-place (e.g. update client tools, stream flag).
    UpdateContext(Box<dyn FnOnce(&mut R::Context) + Send>),
    /// Idle timeout check — carries the generation counter it was spawned with.
    IdleCheck(u64),
}

#[derive(Debug, thiserror::Error)]
pub enum AggregateError<E> {
    #[error(transparent)]
    Command(E),
    #[error(transparent)]
    Store(#[from] StoreError),
}

pub struct AggregateActor<R: AggregateState> {
    _phantom: PhantomData<R>,
}

pub struct AggregateActorState<R: AggregateState> {
    pub aggregate_id: Uuid,
    pub aggregate: Aggregate<R>,
    pub store: Arc<dyn EventStore>,
    pub tenant_id: String,
    pub context: R::Context,
    pub recorder: Option<ExecutionRecorder>,
    idle_timeout: Duration,
    idle_generation: u64,
    idle_timer: Option<AbortHandle>,
}

pub struct AggregateActorArgs<R: AggregateState> {
    pub aggregate_id: Uuid,
    pub store: Arc<dyn EventStore>,
    pub tenant_id: String,
    pub init: Box<dyn Fn(Uuid) -> R + Send + Sync>,
    pub context_init: Box<dyn FnOnce(&R) -> Pin<Box<dyn Future<Output = R::Context> + Send>> + Send>,
    pub recorder: Option<ExecutionRecorder>,
    pub idle_timeout: Option<Duration>,
}

pub async fn spawn_aggregate_actor<R: AggregateState>(
    args: AggregateActorArgs<R>,
    supervisor: ractor::ActorCell,
) -> Result<AggregateActorHandle<R>, ractor::SpawnErr> {
    let name = format!("{}-{}", R::aggregate_type(), args.aggregate_id);
    let (actor, _) = Actor::spawn_linked(
        Some(name),
        AggregateActor::<R>::new(),
        args,
        supervisor,
    )
    .await?;

    Ok(AggregateActorHandle { actor })
}

pub struct AggregateActorHandle<R: AggregateState> {
    pub actor: ActorRef<AggregateMessage<R>>,
}

impl<R: AggregateState> AggregateActorHandle<R> {
    pub async fn send_command(
        &self,
        cmd: R::Command,
        span: SpanContext,
        occurred_at: DateTime<Utc>,
    ) -> Result<Vec<Arc<Event>>, AggregateError<R::Error>> {
        let result = self
            .actor
            .call(
                |reply| AggregateMessage::Execute {
                    cmd,
                    span,
                    occurred_at,
                    reply,
                },
                Some(ractor::concurrency::Duration::from_millis(5000)),
            )
            .await
            .map_err(|e| AggregateError::Store(StoreError::Internal(e.to_string())))?;
        match result {
            ractor::rpc::CallResult::Success(v) => v,
            ractor::rpc::CallResult::Timeout => Err(AggregateError::Store(
                StoreError::Internal("rpc timeout".into()),
            )),
            ractor::rpc::CallResult::SenderError => Err(AggregateError::Store(
                StoreError::Internal("rpc sender error".into()),
            )),
        }
    }

    pub async fn get_state(&self) -> R {
        ractor::call_t!(self.actor, AggregateMessage::GetState, 5000)
            .expect("failed to query aggregate state")
    }

    pub async fn get_aggregate(&self) -> Aggregate<R> {
        ractor::call_t!(self.actor, AggregateMessage::GetAggregate, 5000)
            .expect("failed to query aggregate")
    }
}

impl<R: AggregateState> AggregateActor<R> {
    pub fn new() -> Self {
        AggregateActor {
            _phantom: PhantomData,
        }
    }
}

/// Reset (or start) the idle shutdown timer. Each call increments the
/// generation counter so that stale timers are ignored when they fire.
fn reset_idle_timer<R: AggregateState>(
    myself: &ActorRef<AggregateMessage<R>>,
    state: &mut AggregateActorState<R>,
) {
    if let Some(handle) = state.idle_timer.take() {
        handle.abort();
    }
    state.idle_generation += 1;
    let gen = state.idle_generation;
    let timeout = state.idle_timeout;
    let actor = myself.clone();
    let handle = tokio::spawn(async move {
        tokio::time::sleep(timeout).await;
        let _ = actor.send_message(AggregateMessage::IdleCheck(gen));
    });
    state.idle_timer = Some(handle.abort_handle());
}

#[tracing::instrument(skip_all, fields(trace_id = %span.trace_id))]
async fn execute<R: AggregateState>(
    state: &mut AggregateActorState<R>,
    cmd: R::Command,
    span: SpanContext,
    occurred_at: DateTime<Utc>,
) -> Result<Vec<Arc<Event>>, AggregateError<R::Error>> {
    // Pure decision — no mutation
    let payloads = state
        .aggregate
        .state
        .handle_command(cmd, &state.context)
        .map_err(AggregateError::Command)?;

    if payloads.is_empty() {
        return Ok(vec![]);
    }

    let expected_version = state.aggregate.stream_version;

    // Mutation: apply events and wrap as domain events
    let domain_events = state.aggregate.commit(
        payloads,
        state.aggregate_id,
        span,
        occurred_at,
        &state.tenant_id,
    );

    let new_version = state.aggregate.stream_version;
    let raw_events: Vec<Event> = domain_events
        .into_iter()
        .map(|e| e.into_raw())
        .collect();

    let snapshot_value = serde_json::to_value(&state.aggregate)
        .map_err(|e| StoreError::Internal(e.to_string()))?;

    state
        .store
        .append(
            state.aggregate_id,
            &state.tenant_id,
            R::aggregate_type(),
            raw_events.clone(),
            snapshot_value,
            expected_version,
            new_version,
        )
        .await?;

    Ok(raw_events.into_iter().map(Arc::new).collect())
}

impl<R: AggregateState> Actor for AggregateActor<R> {
    type Msg = AggregateMessage<R>;
    type State = AggregateActorState<R>;
    type Arguments = AggregateActorArgs<R>;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let aggregate_id = args.aggregate_id;
        let aggregate: Aggregate<R> = match args.store.load(aggregate_id, &args.tenant_id).await {
            Ok(loaded) => serde_json::from_value(loaded.snapshot)
                .map_err(|e| format!("snapshot deserialize: {e}"))?,
            Err(StoreError::StreamNotFound) => Aggregate::new((args.init)(aggregate_id)),
            Err(e) => return Err(format!("load: {e}").into()),
        };

        let context = (args.context_init)(&aggregate.state).await;

        let mut state = AggregateActorState {
            aggregate_id,
            aggregate,
            store: args.store,
            tenant_id: args.tenant_id,
            context,
            recorder: args.recorder,
            idle_timeout: args.idle_timeout.unwrap_or(DEFAULT_IDLE_TIMEOUT),
            idle_generation: 0,
            idle_timer: None,
        };

        reset_idle_timer(&myself, &mut state);

        Ok(state)
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            AggregateMessage::IdleCheck(gen) => {
                if gen == state.idle_generation
                    && state.aggregate.status != AggregateStatus::Active
                {
                    tracing::debug!(
                        aggregate_id = %state.aggregate_id,
                        "idle timeout — stopping aggregate actor"
                    );
                    myself.stop(None);
                }
                return Ok(());
            }
            AggregateMessage::Execute {
                cmd,
                span,
                occurred_at,
                reply,
            } => {
                let start = SystemTime::now();
                let span_copy = span.clone();
                let result = execute(state, cmd, span, occurred_at).await;
                if let (Some(recorder), Ok(produced)) = (&state.recorder, &result) {
                    if !produced.is_empty() {
                        recorder(ExecutionRecord {
                            span_context: span_copy,
                            start_time: start,
                            end_time: SystemTime::now(),
                            aggregate_type: R::aggregate_type(),
                            aggregate_id: state.aggregate_id,
                            produced_events: produced.clone(),
                        });
                    }
                }
                let _ = reply.send(result);
            }
            AggregateMessage::Cast {
                cmd,
                span,
                occurred_at,
            } => {
                let start = SystemTime::now();
                let span_copy = span.clone();
                let result = execute(state, cmd, span, occurred_at).await;
                if let (Some(recorder), Ok(produced)) = (&state.recorder, &result) {
                    if !produced.is_empty() {
                        recorder(ExecutionRecord {
                            span_context: span_copy,
                            start_time: start,
                            end_time: SystemTime::now(),
                            aggregate_type: R::aggregate_type(),
                            aggregate_id: state.aggregate_id,
                            produced_events: produced.clone(),
                        });
                    }
                }
            }
            AggregateMessage::GetState(reply) => {
                let _ = reply.send(state.aggregate.state.clone());
            }
            AggregateMessage::GetAggregate(reply) => {
                let _ = reply.send(state.aggregate.clone());
            }
            AggregateMessage::Events(typed_events) => {
                for event in &typed_events {
                    // Start timing before on_event — it does the actual I/O
                    let start = SystemTime::now();
                    if let Some(cmd) = state.aggregate.state.on_event(&event.payload, &state.context, &event.span).await {
                        // Use triggering event's serde type tag as span name
                        let event_type = serde_json::to_value(&event.payload)
                            .ok()
                            .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(String::from))
                            .unwrap_or_else(|| "reaction".to_string());
                        let child_span = event.span.child(&event_type);
                        let result = execute(state, cmd, child_span.clone(), Utc::now()).await;

                        // Record the full on_event + execute cycle for OTel
                        if let (Some(recorder), Ok(produced)) = (&state.recorder, &result) {
                            recorder(ExecutionRecord {
                                span_context: child_span,
                                start_time: start,
                                end_time: SystemTime::now(),
                                aggregate_type: R::aggregate_type(),
                                aggregate_id: state.aggregate_id,
                                produced_events: produced.clone(),
                            });
                        }
                    }
                }
            }
            AggregateMessage::UpdateContext(f) => {
                f(&mut state.context);
            }
        }

        // Any real activity resets the idle timer.
        reset_idle_timer(&myself, state);

        Ok(())
    }
}
