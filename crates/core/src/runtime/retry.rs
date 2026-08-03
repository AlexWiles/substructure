use std::cmp::min;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::protocol::{RetryConfig, RetryPolicy};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryState {
    pub attempts: u32,
    pub next_at: Option<DateTime<Utc>>,
}

/// What a policy is being resolved for. Finer than `EffectKind` because a tool
/// call's default follows where it runs: a worker tool must be bounded, and a
/// client tool must not be — a deferred call waits for a human.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryTarget {
    Llm,
    WorkerTool,
    ClientTool,
    ConnectorTool,
    SubAgent,
    ConnectorSync,
    Decision,
}

impl RetryPolicy {
    /// One attempt, unbounded, never retried.
    pub fn no_retry() -> Self {
        RetryPolicy {
            attempt_timeout_secs: None,
            total_timeout_secs: None,
            max_attempts: 1,
            backoff_base_secs: 0,
            backoff_max_secs: 0,
        }
    }

    /// The engine's default when neither the action nor the agent config names a
    /// policy. Every target is bounded except the client tool, so a dead worker
    /// or a hung child fails its turn instead of stalling it forever.
    pub fn default_for(target: RetryTarget) -> Self {
        let (attempt, total, max_attempts, base, max) = match target {
            // Idempotent, and the providers classify their own retryable
            // failures (429/5xx/connect/timeout).
            RetryTarget::Llm => (Some(180), Some(1800), 5, 2, 30),
            // Bounded but never repeated: the engine cannot vouch for a tool's
            // idempotency, so a retry is the author's call, not ours.
            RetryTarget::WorkerTool => (Some(120), Some(600), 1, 0, 0),
            // Deferred by design — a client tool waits for a human, so a
            // deadline would settle it as failed while it is still legitimately
            // open. See `docs/130-deferred-tools.md`.
            RetryTarget::ClientTool => (None, None, 1, 0, 0),
            // The engine places this call itself, so a transport failure is its
            // own to absorb. Still shallow: the turn waits behind it.
            RetryTarget::ConnectorTool => (Some(60), Some(300), 2, 1, 10),
            // The spawn is idempotent — the child session id is deterministic
            // and an already-created session counts as success — so re-issuing
            // one is safe. `total` is generous because it also bounds the child
            // turn, which is `Running` and off the attempt clock.
            RetryTarget::SubAgent => (Some(60), Some(3600), 3, 2, 15),
            // Every decision that would prompt the model waits behind this, so
            // it is deliberately shorter and shallower than an LLM call: a
            // connection that cannot answer promptly should settle as failed and
            // let the worker decide, not hold the turn.
            RetryTarget::ConnectorSync => (Some(30), Some(120), 3, 1, 10),
            // A dequeued decision is never redelivered, so one lost between
            // dispatch and reply — a dead push loop, a restart — has no recovery
            // path but its deadline. Deciding replays durable state and submits
            // are idempotent by decision id, so retries are safe.
            RetryTarget::Decision => (Some(300), Some(1800), 10, 2, 60),
        };
        RetryPolicy {
            attempt_timeout_secs: attempt,
            total_timeout_secs: total,
            max_attempts,
            backoff_base_secs: base,
            backoff_max_secs: max,
        }
    }

    /// The policy in force, most specific first: what the action asked for, then
    /// what the agent declared for this kind, then the agent's `default`, then
    /// the engine's default for the target.
    pub fn resolve(
        action: Option<RetryPolicy>,
        config: Option<&RetryConfig>,
        target: RetryTarget,
    ) -> Self {
        action
            .or_else(|| config.and_then(|c| c.for_target(target)))
            .unwrap_or_else(|| RetryPolicy::default_for(target))
    }

    /// The deadline for one attempt started at `now`. None ⇒ waits indefinitely.
    pub fn attempt_deadline(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        self.attempt_timeout_secs
            .map(|s| now + chrono::Duration::seconds(i64::from(s)))
    }

    /// The deadline for the whole effect, measured from its first dispatch —
    /// backoff and `Running` time included. None ⇒ unbounded.
    pub fn total_deadline(&self, started_at: DateTime<Utc>) -> Option<DateTime<Utc>> {
        self.total_timeout_secs
            .map(|s| started_at + chrono::Duration::seconds(i64::from(s)))
    }

    /// Record a failure and return the new retry state.
    /// `next_at` will be set if retries remain, None if exhausted.
    pub fn record_failure(&self, state: &RetryState, now: DateTime<Utc>) -> RetryState {
        let attempts = state.attempts + 1;
        let next_at = self.next_retry_at(attempts, now);
        RetryState { attempts, next_at }
    }

    /// Returns true if the next failure (given current state) would exhaust the
    /// attempts, or if the failure is not retryable.
    pub fn exhausted(&self, state: &RetryState, retryable: bool) -> bool {
        !retryable || state.attempts + 1 >= self.attempts()
    }

    /// The attempt cap, floored at one: a policy still has to try once.
    fn attempts(&self) -> u32 {
        self.max_attempts.max(1)
    }

    fn backoff_secs(&self, attempts: u32) -> Option<u32> {
        if attempts >= self.attempts() {
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

impl RetryConfig {
    /// What this config declares for `target`, falling back to its own
    /// `default`. None ⇒ it says nothing, and the engine default applies.
    fn for_target(&self, target: RetryTarget) -> Option<RetryPolicy> {
        let declared = match target {
            RetryTarget::Llm => &self.llm,
            RetryTarget::WorkerTool | RetryTarget::ClientTool | RetryTarget::ConnectorTool => {
                &self.tool
            }
            RetryTarget::SubAgent => &self.sub_agent,
            RetryTarget::ConnectorSync => &self.connector,
            // Not the agent's to declare, and `default` must not reach it
            // either: it bounds the call that produces the config, so reading
            // the policy from there would be circular.
            RetryTarget::Decision => return None,
        };
        declared.clone().or_else(|| self.default.clone())
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
        RetryPolicy::default_for(RetryTarget::Decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            attempt_timeout_secs: Some(1),
            total_timeout_secs: Some(9),
            max_attempts,
            backoff_base_secs: 2,
            backoff_max_secs: 30,
        }
    }

    fn config(default: Option<u32>, tool: Option<u32>) -> RetryConfig {
        RetryConfig {
            default: default.map(policy),
            tool: tool.map(policy),
            ..Default::default()
        }
    }

    #[test]
    fn the_action_wins_over_every_config_level() {
        let resolved = RetryPolicy::resolve(
            Some(policy(7)),
            Some(&config(Some(3), Some(5))),
            RetryTarget::WorkerTool,
        );
        assert_eq!(resolved.max_attempts, 7);
    }

    #[test]
    fn a_declared_kind_wins_over_the_agent_default() {
        let resolved = RetryPolicy::resolve(
            None,
            Some(&config(Some(3), Some(5))),
            RetryTarget::WorkerTool,
        );
        assert_eq!(resolved.max_attempts, 5);
    }

    #[test]
    fn a_kind_that_declares_nothing_takes_the_agent_default() {
        let resolved =
            RetryPolicy::resolve(None, Some(&config(Some(3), None)), RetryTarget::WorkerTool);
        assert_eq!(resolved.max_attempts, 3, "no `tool` ⇒ `default` applies");
    }

    #[test]
    fn an_empty_config_falls_through_to_the_engine_default() {
        let resolved = RetryPolicy::resolve(None, Some(&config(None, None)), RetryTarget::Llm);
        assert_eq!(resolved, RetryPolicy::default_for(RetryTarget::Llm));
    }

    #[test]
    fn every_tool_kind_reads_the_one_tool_entry() {
        let cfg = config(None, Some(5));
        for target in [
            RetryTarget::WorkerTool,
            RetryTarget::ClientTool,
            RetryTarget::ConnectorTool,
        ] {
            assert_eq!(
                RetryPolicy::resolve(None, Some(&cfg), target).max_attempts,
                5,
                "{target:?} reads `tool`"
            );
        }
    }

    #[test]
    fn a_decision_ignores_the_agent_config() {
        let resolved =
            RetryPolicy::resolve(None, Some(&config(Some(3), None)), RetryTarget::Decision);
        assert_eq!(
            resolved,
            RetryPolicy::default_for(RetryTarget::Decision),
            "the call that produces the config cannot be bounded by it"
        );
    }

    #[test]
    fn only_a_client_tool_is_left_unbounded_by_default() {
        for target in [
            RetryTarget::Llm,
            RetryTarget::WorkerTool,
            RetryTarget::ConnectorTool,
            RetryTarget::SubAgent,
            RetryTarget::ConnectorSync,
            RetryTarget::Decision,
        ] {
            let p = RetryPolicy::default_for(target);
            assert!(p.attempt_timeout_secs.is_some(), "{target:?} attempt bound");
            assert!(p.total_timeout_secs.is_some(), "{target:?} total bound");
        }
        let client = RetryPolicy::default_for(RetryTarget::ClientTool);
        assert_eq!(
            client.attempt_timeout_secs, None,
            "deferred calls stay open"
        );
        assert_eq!(client.total_timeout_secs, None);
    }

    #[test]
    fn a_worker_tool_is_bounded_but_never_repeated() {
        let p = RetryPolicy::default_for(RetryTarget::WorkerTool);
        assert_eq!(p.attempt_timeout_secs, Some(120));
        assert_eq!(p.max_attempts, 1, "the engine cannot vouch for idempotency");
        assert!(p.exhausted(&RetryState::default(), true), "one try only");
    }

    #[test]
    fn max_attempts_counts_attempts_not_retries() {
        let p = policy(3);
        let mut state = RetryState::default();
        for attempt in 1..3 {
            assert!(!p.exhausted(&state, true), "attempt {attempt} of 3");
            state = p.record_failure(&state, Utc::now());
        }
        assert!(
            p.exhausted(&state, true),
            "3 attempts ⇒ 2 retries, then done"
        );
    }

    #[test]
    fn a_zero_attempt_policy_still_tries_once() {
        assert!(policy(0).exhausted(&RetryState::default(), true));
    }

    #[test]
    fn a_failure_that_is_not_retryable_is_terminal_with_attempts_left() {
        assert!(policy(9).exhausted(&RetryState::default(), false));
    }

    #[test]
    fn the_total_deadline_runs_from_the_first_dispatch() {
        let started = Utc::now();
        let later = started + chrono::Duration::seconds(60);
        let p = policy(3);
        assert_eq!(
            p.total_deadline(started),
            Some(started + chrono::Duration::seconds(9)),
            "measured from the start, so a retry cannot push it out"
        );
        assert_eq!(
            p.attempt_deadline(later),
            Some(later + chrono::Duration::seconds(1)),
            "the attempt clock restarts with each attempt"
        );
    }
}
