use chrono::Utc;
use uuid::Uuid;

use crate::runtime::event_store::{AppendInput, EventStore, StoreError};
use crate::runtime::span::SpanContext;

use super::aggregate::{Aggregate, CommitContext};
use super::state::{AggregateState, DomainEvent};

#[derive(Debug, thiserror::Error)]
pub enum ExecuteError<E: std::fmt::Debug> {
    #[error("command error: {0:?}")]
    Command(E),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub struct ExecuteResult<R: AggregateState> {
    pub aggregate: Aggregate<R>,
    pub events: Vec<DomainEvent<R>>,
}

pub struct ExecuteInput<R: AggregateState> {
    pub aggregate_id: Uuid,
    pub tenant_id: String,
    pub command: R::Command,
    pub span: SpanContext,
}

pub async fn execute<R: AggregateState>(
    store: &dyn EventStore,
    input: ExecuteInput<R>,
) -> Result<ExecuteResult<R>, ExecuteError<R::Error>> {
    let start_time = Utc::now();

    // 1. Load or create aggregate.
    let (mut aggregate, expected_version) = match store.load(input.aggregate_id, &input.tenant_id).await {
        Ok(loaded) => {
            let agg: Aggregate<R> =
                serde_json::from_value(loaded.snapshot)?;
            (agg, loaded.stream_version)
        }
        Err(StoreError::StreamNotFound) => (Aggregate::new(R::initial(input.aggregate_id)), 0),
        Err(e) => return Err(e.into()),
    };

    // 2. Handle command.
    let events = aggregate
        .state
        .handle_command(input.command)
        .map_err(ExecuteError::Command)?;

    if events.is_empty() {
        return Ok(ExecuteResult {
            aggregate,
            events: vec![],
        });
    }

    // 3. Commit — apply events, wrap as domain events.
    let commit_ctx = CommitContext {
        aggregate_id: input.aggregate_id,
        tenant_id: input.tenant_id.clone(),
        span: input.span,
        occurred_at: Utc::now(),
    };
    let domain_events = aggregate.commit(events, &commit_ctx);

    // 4. Convert to store events.
    let end_time = Utc::now();
    let store_events: Vec<_> = domain_events
        .clone()
        .into_iter()
        .map(|e| e.into_raw(start_time, end_time))
        .collect::<Result<_, _>>()?;

    // 5. Append to store.
    let snapshot = serde_json::to_value(&aggregate)?;
    store
        .append(AppendInput {
            aggregate_id: input.aggregate_id,
            tenant_id: input.tenant_id,
            aggregate_type: R::AGGREGATE_TYPE.to_string(),
            events: store_events,
            snapshot,
            expected_version,
            new_version: aggregate.stream_version,
        })
        .await?;

    Ok(ExecuteResult {
        aggregate,
        events: domain_events,
    })
}
