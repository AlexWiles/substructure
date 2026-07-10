use std::cmp::min;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryState {
    pub attempts: u32,
    pub next_at: Option<DateTime<Utc>>,
}

/// Fully-resolved retry policy — no optional fields. Stored on call state and
/// read directly by retry logic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub timeout_secs: Option<u32>,
    pub max_retries: u32,
    pub backoff_base_secs: u32,
    pub backoff_max_secs: u32,
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
