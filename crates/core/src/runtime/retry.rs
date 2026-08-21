use std::cmp::min;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::protocol::{RetryConfig, RetryOverride, RetryPolicy};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryState {
    pub attempts: u32,
    pub next_at: Option<DateTime<Utc>>,
}

/// What a policy is being resolved for. Finer than `EffectKind` because a tool
/// call's default follows where it runs: a worker tool must be bounded, and a
/// client tool must not be — an async call waits for a human.
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

    pub fn default_for(target: RetryTarget) -> Self {
        let (attempt, total, max_attempts, base, max) = match target {
            RetryTarget::Llm => (Some(180), Some(1800), 5, 2, 30),
            RetryTarget::WorkerTool => (Some(120), Some(600), 1, 0, 0),
            RetryTarget::ClientTool => (None, None, 1, 0, 0),
            RetryTarget::ConnectorTool => (Some(60), Some(300), 2, 1, 10),
            RetryTarget::SubAgent => (Some(10), Some(3600), 3, 2, 15),
            RetryTarget::ConnectorSync => (Some(5), Some(30), 2, 2, 10),
            RetryTarget::Decision => (Some(20), Some(300), 10, 2, 60),
        };
        RetryPolicy {
            attempt_timeout_secs: attempt,
            total_timeout_secs: total,
            max_attempts,
            backoff_base_secs: base,
            backoff_max_secs: max,
        }
    }

    /// The policy in force: the engine's default for the target, with each
    /// override layered on top — the agent's `default`, then what it declared
    /// for this kind, then what the action asked for.
    ///
    /// Layered rather than replaced so that naming one field changes one field.
    /// A whole-policy override would make every partial config a silent way to
    /// drop the bounds it did not restate.
    pub fn resolve(
        action: Option<&RetryOverride>,
        config: Option<&RetryConfig>,
        target: RetryTarget,
    ) -> Self {
        let mut policy = RetryPolicy::default_for(target);
        for layer in config.map(|c| c.layers_for(target)).unwrap_or_default() {
            policy = policy.with_override(layer);
        }
        match action {
            Some(o) => policy.with_override(o),
            None => policy,
        }
    }

    /// This policy as an override that pins every field — for re-proposing a
    /// call under exactly the policy it already ran on.
    ///
    /// An unset timeout stays unset, which an override cannot express, so it
    /// re-resolves to the default instead. Unreachable in practice: only a
    /// client tool defaults to unbounded, and its calls are never re-proposed.
    pub fn as_override(&self) -> RetryOverride {
        RetryOverride {
            attempt_timeout_secs: self.attempt_timeout_secs,
            total_timeout_secs: self.total_timeout_secs,
            max_attempts: Some(self.max_attempts),
            backoff_base_secs: Some(self.backoff_base_secs),
            backoff_max_secs: Some(self.backoff_max_secs),
        }
    }

    /// This policy with `o`'s named fields replaced and the rest left alone.
    pub fn with_override(mut self, o: &RetryOverride) -> Self {
        if let Some(v) = o.attempt_timeout_secs {
            self.attempt_timeout_secs = Some(v);
        }
        if let Some(v) = o.total_timeout_secs {
            self.total_timeout_secs = Some(v);
        }
        if let Some(v) = o.max_attempts {
            self.max_attempts = v;
        }
        if let Some(v) = o.backoff_base_secs {
            self.backoff_base_secs = v;
        }
        if let Some(v) = o.backoff_max_secs {
            self.backoff_max_secs = v;
        }
        self
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
    /// The overrides that apply to `target`, broadest first: `default`, then
    /// what the config declares for the kind. Both apply, so a `default` sets
    /// the shape and a kind adjusts it.
    fn layers_for(&self, target: RetryTarget) -> Vec<&RetryOverride> {
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
            RetryTarget::Decision => return Vec::new(),
        };
        self.default.iter().chain(declared.iter()).collect()
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

    /// An override naming only `max_attempts` — the partial shape that used to
    /// be impossible to write.
    fn attempts(n: u32) -> RetryOverride {
        RetryOverride {
            max_attempts: Some(n),
            ..Default::default()
        }
    }

    fn config(default: Option<RetryOverride>, tool: Option<RetryOverride>) -> RetryConfig {
        RetryConfig {
            default,
            tool,
            ..Default::default()
        }
    }

    #[test]
    fn an_override_changes_only_the_fields_it_names() {
        let base = RetryPolicy::default_for(RetryTarget::WorkerTool);
        let resolved = RetryPolicy::resolve(
            None,
            Some(&config(None, Some(attempts(5)))),
            RetryTarget::WorkerTool,
        );
        assert_eq!(resolved.max_attempts, 5, "the named field changes");
        assert_eq!(
            resolved.attempt_timeout_secs, base.attempt_timeout_secs,
            "an unnamed timeout keeps the default bound; it is not removed"
        );
        assert_eq!(resolved.total_timeout_secs, base.total_timeout_secs);
        assert_eq!(resolved.backoff_base_secs, base.backoff_base_secs);
        assert_eq!(resolved.backoff_max_secs, base.backoff_max_secs);
    }

    #[test]
    fn a_kind_layers_over_the_agent_default_rather_than_replacing_it() {
        let resolved = RetryPolicy::resolve(
            None,
            Some(&config(
                Some(RetryOverride {
                    attempt_timeout_secs: Some(11),
                    max_attempts: Some(3),
                    ..Default::default()
                }),
                Some(attempts(5)),
            )),
            RetryTarget::WorkerTool,
        );
        assert_eq!(resolved.max_attempts, 5, "the kind wins where both name it");
        assert_eq!(
            resolved.attempt_timeout_secs,
            Some(11),
            "`default` still applies where the kind is silent"
        );
    }

    #[test]
    fn the_action_is_the_last_layer() {
        let resolved = RetryPolicy::resolve(
            Some(&attempts(7)),
            Some(&config(Some(attempts(3)), Some(attempts(5)))),
            RetryTarget::WorkerTool,
        );
        assert_eq!(resolved.max_attempts, 7);
        assert_eq!(
            resolved.attempt_timeout_secs,
            RetryPolicy::default_for(RetryTarget::WorkerTool).attempt_timeout_secs,
            "a partial action override still keeps the engine bound"
        );
    }

    #[test]
    fn a_kind_that_declares_nothing_takes_the_agent_default() {
        let resolved = RetryPolicy::resolve(
            None,
            Some(&config(Some(attempts(3)), None)),
            RetryTarget::WorkerTool,
        );
        assert_eq!(resolved.max_attempts, 3);
    }

    #[test]
    fn an_empty_config_falls_through_to_the_engine_default() {
        let resolved = RetryPolicy::resolve(None, Some(&config(None, None)), RetryTarget::Llm);
        assert_eq!(resolved, RetryPolicy::default_for(RetryTarget::Llm));
    }

    #[test]
    fn every_tool_kind_reads_the_one_tool_entry() {
        let cfg = config(None, Some(attempts(5)));
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
    fn a_tool_override_does_not_bound_a_client_tool_by_accident() {
        let resolved = RetryPolicy::resolve(
            None,
            Some(&config(None, Some(attempts(3)))),
            RetryTarget::ClientTool,
        );
        assert_eq!(
            resolved.attempt_timeout_secs, None,
            "an async call stays open unless the override says otherwise"
        );
        assert_eq!(resolved.total_timeout_secs, None);
    }

    #[test]
    fn a_decision_ignores_the_agent_config() {
        let resolved = RetryPolicy::resolve(
            None,
            Some(&config(Some(attempts(3)), None)),
            RetryTarget::Decision,
        );
        assert_eq!(
            resolved,
            RetryPolicy::default_for(RetryTarget::Decision),
            "the call that produces the config cannot be bounded by it"
        );
    }

    #[test]
    fn as_override_round_trips_a_resolved_policy() {
        let policy = RetryPolicy::default_for(RetryTarget::Llm);
        assert_eq!(
            RetryPolicy::resolve(Some(&policy.as_override()), None, RetryTarget::WorkerTool),
            policy,
            "pinning every field ignores the target's own default"
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
        assert_eq!(client.attempt_timeout_secs, None, "async calls stay open");
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
        let p = RetryPolicy::default_for(RetryTarget::Llm).with_override(&attempts(3));
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
        let p = RetryPolicy::default_for(RetryTarget::Llm).with_override(&attempts(0));
        assert!(p.exhausted(&RetryState::default(), true));
    }

    #[test]
    fn a_failure_that_is_not_retryable_is_terminal_with_attempts_left() {
        let p = RetryPolicy::default_for(RetryTarget::Llm).with_override(&attempts(9));
        assert!(p.exhausted(&RetryState::default(), false));
    }

    #[test]
    fn the_total_deadline_runs_from_the_first_dispatch() {
        let started = Utc::now();
        let later = started + chrono::Duration::seconds(60);
        let p = RetryPolicy {
            attempt_timeout_secs: Some(1),
            total_timeout_secs: Some(9),
            max_attempts: 3,
            backoff_base_secs: 2,
            backoff_max_secs: 30,
        };
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
