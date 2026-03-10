use chrono::Utc;

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
    pub aggregate_id: uuid::Uuid,
    pub tenant_id: String,
    pub command: R::Command,
    pub span: SpanContext,
}

pub async fn execute<R: AggregateState>(
    store: &dyn EventStore,
    input: ExecuteInput<R>,
) -> Result<ExecuteResult<R>, ExecuteError<R::Error>> {
    let start_time = Utc::now();

    let (mut aggregate, expected_version) =
        Aggregate::<R>::load_or_create(store, input.aggregate_id, input.tenant_id).await?;

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

    // 5. Append to store (store is responsible for publishing to the bus).
    store
        .append(AppendInput {
            events: store_events,
            snapshot: aggregate.to_snapshot()?,
            expected_version,
        })
        .await?;

    Ok(ExecuteResult {
        aggregate,
        events: domain_events,
    })
}
