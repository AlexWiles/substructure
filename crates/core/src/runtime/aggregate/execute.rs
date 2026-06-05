use std::time::Duration;

use chrono::Utc;
use rand::Rng;

use crate::runtime::event_store::{AppendInput, EventStore, StoreError};
use crate::runtime::span::SpanContext;

use super::aggregate::{Aggregate, CommitContext};
use super::caller::Caller;
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
    pub aggregate_id: String,
    pub caller: Caller,
    pub command: R::Command,
    pub span: SpanContext,
}

#[derive(Debug, Clone)]
pub struct ConflictRetry {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub jitter: bool,
}

impl Default for ConflictRetry {
    fn default() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(500),
            jitter: true,
        }
    }
}

impl ConflictRetry {
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        }
    }

    fn delay_for(&self, attempt: u32) -> Duration {
        let exp = self.base_delay.saturating_mul(1 << attempt.min(16));
        let capped = exp.min(self.max_delay);
        if self.jitter {
            let millis = capped.as_millis() as u64;
            if millis == 0 {
                return Duration::ZERO;
            }
            let jittered = rand::rng().random_range(0..=millis);
            Duration::from_millis(jittered)
        } else {
            capped
        }
    }
}

pub async fn execute<R: AggregateState>(
    store: &dyn EventStore,
    input: ExecuteInput<R>,
    retry: &ConflictRetry,
) -> Result<ExecuteResult<R>, ExecuteError<R::Error>> {
    let mut attempt = 0u32;

    loop {
        let start_time = Utc::now();
        let command = input.command.clone();

        let (mut aggregate, expected_version) = Aggregate::<R>::load_or_create(
            store,
            input.aggregate_id.clone(),
            input.caller.tenant_id().to_string(),
        )
        .await?;

        let events = aggregate
            .state
            .handle_command(command, &input.caller)
            .map_err(ExecuteError::Command)?;

        if events.is_empty() {
            return Ok(ExecuteResult {
                aggregate,
                events: vec![],
            });
        }

        let commit_ctx = CommitContext {
            span: input.span.clone(),
            occurred_at: Utc::now(),
        };
        let domain_events = aggregate.commit(events, &commit_ctx);

        let end_time = Utc::now();
        let store_events: Vec<_> = domain_events
            .clone()
            .into_iter()
            .map(|e| e.into_raw(start_time, end_time))
            .collect::<Result<_, _>>()?;

        match store
            .append(AppendInput {
                events: store_events,
                snapshot: aggregate.to_snapshot()?,
                expected_version,
            })
            .await
        {
            Ok(()) => {
                return Ok(ExecuteResult {
                    aggregate,
                    events: domain_events,
                });
            }
            Err(StoreError::VersionConflict { .. }) if attempt < retry.max_retries => {
                let delay = retry.delay_for(attempt);
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                attempt += 1;
            }
            Err(e) => return Err(e.into()),
        }
    }
}
