use std::cmp::min;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::protocol::RetryPolicy;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryState {
    pub attempts: u32,
    pub next_at: Option<DateTime<Utc>>,
}

impl RetryPolicy {
    pub fn no_retry() -> Self {
        RetryPolicy {
            timeout_secs: None,
            max_retries: 0,
            backoff_base_secs: 0,
            backoff_max_secs: 0,
        }
    }

    /// The default for worker decisions. A dequeued decision is never redelivered,
    /// so one lost between dispatch and reply — a dead push loop, a restart — has no
    /// recovery path but its deadline. Deciding replays durable state and submits are
    /// idempotent by decision id, so retries are safe; they ride out worker deploys
    /// and transient transport failures instead of failing the run.
    pub fn worker_default() -> Self {
        RetryPolicy {
            timeout_secs: Some(300),
            max_retries: 10,
            backoff_base_secs: 2,
            backoff_max_secs: 60,
        }
    }

    /// The default for LLM calls when neither the action nor the agent config sets
    /// a policy. Bounds a hung provider connection and retries the failures the
    /// providers classify retryable (429/5xx/connect/timeout).
    pub fn llm_default() -> Self {
        RetryPolicy {
            timeout_secs: Some(180),
            max_retries: 5,
            backoff_base_secs: 2,
            backoff_max_secs: 30,
        }
    }

    /// The default for fetching a connection's tool list. Every decision that
    /// would prompt the model waits behind this, so it is deliberately shorter
    /// and shallower than an LLM call: a connection that cannot answer promptly
    /// should settle as failed and let the worker decide, not hold the turn.
    pub fn connector_default() -> Self {
        RetryPolicy {
            timeout_secs: Some(30),
            max_retries: 3,
            backoff_base_secs: 1,
            backoff_max_secs: 10,
        }
    }

    /// Compute the deadline from a start time. Returns None if no timeout is set.
    pub fn deadline(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        self.timeout_secs
            .map(|s| now + chrono::Duration::seconds(i64::from(s)))
    }

    /// Record a failure and return the new retry state.
    /// `next_at` will be set if retries remain, None if exhausted.
    pub fn record_failure(&self, state: &RetryState, now: DateTime<Utc>) -> RetryState {
        let attempts = state.attempts + 1;
        let next_at = self.next_retry_at(attempts, now);
        RetryState { attempts, next_at }
    }

    /// Returns true if the next failure (given current state) would exhaust retries,
    /// or if the failure is not retryable.
    pub fn exhausted(&self, state: &RetryState, retryable: bool) -> bool {
        !retryable || state.attempts + 1 >= self.max_retries
    }

    fn backoff_secs(&self, attempts: u32) -> Option<u32> {
        if attempts >= self.max_retries {
            return None;
        }
        Some(min(
            self.backoff_base_secs.saturating_pow(attempts),
            self.backoff_max_secs,
        ))
    }

    fn next_retry_at(&self, attempts: u32, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        self.backoff_secs(attempts)
            .map(|b| now + chrono::Duration::seconds(i64::from(b)))
    }
}

#[async_trait]
pub trait WorkerRetryResolver: Send + Sync {
    async fn resolve(&self, tenant_id: &str) -> RetryPolicy;
}

pub struct NoRetryResolver;

#[async_trait]
impl WorkerRetryResolver for NoRetryResolver {
    async fn resolve(&self, _tenant_id: &str) -> RetryPolicy {
        RetryPolicy::no_retry()
    }
}

pub struct DefaultWorkerRetryResolver;

#[async_trait]
impl WorkerRetryResolver for DefaultWorkerRetryResolver {
    async fn resolve(&self, _tenant_id: &str) -> RetryPolicy {
        RetryPolicy::worker_default()
    }
}
