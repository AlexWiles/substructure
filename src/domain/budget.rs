use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::aggregate::Aggregate;
use super::config::{parse_window, BudgetPolicyConfig, ExhaustionStrategy};

/// Derive a deterministic aggregate_id from tenant_id.
pub fn budget_aggregate_id(tenant_id: &str) -> Uuid {
    Uuid::new_v5(&Uuid::NAMESPACE_OID, tenant_id.as_bytes())
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BudgetEvent {
    #[serde(rename = "budget.usage_recorded")]
    UsageRecorded(UsageRecorded),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecorded {
    pub policy_name: String,
    pub bucket_key: String,
    pub session_id: Uuid,
    pub call_id: String,
    pub amount: u64,
    pub recorded_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Derived (none needed for now)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetDerived {}

// ---------------------------------------------------------------------------
// BudgetContext — bag of key-value pairs for group_by and match
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct BudgetContext {
    pub values: HashMap<String, String>,
}

impl BudgetContext {
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.values.insert(key.into(), value.into());
        self
    }

    /// Build a bucket key from ordered group_by fields.
    /// Returns `Some("")` for empty group_by (whole-tenant bucket).
    /// Returns `None` if any required field is missing from context.
    pub fn bucket_key(&self, group_by: &[String]) -> Option<String> {
        if group_by.is_empty() {
            return Some(String::new());
        }
        let parts: Option<Vec<&str>> = group_by
            .iter()
            .map(|k| self.values.get(k).map(|v| v.as_str()))
            .collect();
        parts.map(|p| p.join("|"))
    }

    /// Check if all match conditions are satisfied.
    /// An empty condition map always matches.
    pub fn matches(&self, conditions: &HashMap<String, String>) -> bool {
        conditions
            .iter()
            .all(|(k, v)| self.values.get(k).map_or(false, |actual| actual == v))
    }
}

// ---------------------------------------------------------------------------
// Aggregate state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetLedger {
    pub stream_version: u64,
    pub buckets: HashMap<String, BucketState>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BucketState {
    pub entries: VecDeque<UsageEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEntry {
    pub session_id: Uuid,
    pub call_id: String,
    pub amount: u64,
    pub recorded_at: DateTime<Utc>,
}

impl Aggregate for BudgetLedger {
    type Event = BudgetEvent;
    type Derived = BudgetDerived;

    fn aggregate_type() -> &'static str {
        "budget"
    }

    fn stream_version(&self) -> u64 {
        self.stream_version
    }

    fn apply(&mut self, event: &BudgetEvent, sequence: u64) {
        self.stream_version = sequence;
        match event {
            BudgetEvent::UsageRecorded(e) => {
                let key = composite_key(&e.policy_name, &e.bucket_key);
                let bucket = self.buckets.entry(key).or_default();
                bucket.entries.push_back(UsageEntry {
                    session_id: e.session_id,
                    call_id: e.call_id.clone(),
                    amount: e.amount,
                    recorded_at: e.recorded_at,
                });
            }
        }
    }
}

/// Build the composite bucket key: "policy_name|bucket_key".
pub fn composite_key(policy_name: &str, bucket_key: &str) -> String {
    format!("{policy_name}|{bucket_key}")
}

// ---------------------------------------------------------------------------
// Reservation types (runtime-only, not persisted)
// ---------------------------------------------------------------------------

pub struct ReservationEntry {
    pub composite_key: String,
    pub amount: u64,
}

pub enum ReservationResult {
    Granted,
    Denied {
        policy_name: String,
        strategy: ExhaustionStrategy,
        current: u64,
        limit: u64,
    },
}

// ---------------------------------------------------------------------------
// BudgetLedger query methods
// ---------------------------------------------------------------------------

impl BudgetLedger {
    /// Check all matching policies and determine if a reservation can be granted.
    ///
    /// Pure query — does NOT mutate the ledger. On success, returns the
    /// reservation entries that the caller should add to runtime state.
    pub fn try_reserve(
        &self,
        policies: &[BudgetPolicyConfig],
        ctx: &BudgetContext,
        estimated_tokens: u64,
        reservations: &HashMap<(Uuid, String), Vec<ReservationEntry>>,
        now: DateTime<Utc>,
    ) -> (ReservationResult, Vec<ReservationEntry>) {
        let mut pending_entries = Vec::new();

        for policy in policies {
            // Check match conditions
            if let Some(ref conditions) = policy.match_conditions {
                if !ctx.matches(conditions) {
                    continue;
                }
            }

            // Build bucket key; skip if context is missing required fields
            let bucket_key = match ctx.bucket_key(&policy.group_by) {
                Some(k) => k,
                None => continue,
            };

            let ck = composite_key(&policy.name, &bucket_key);

            // Settled usage
            let settled = self
                .buckets
                .get(&ck)
                .map(|b| usage_in_window(b, &policy.window, now))
                .unwrap_or(0);

            // Active reservations for this composite key
            let reserved: u64 = reservations
                .values()
                .flat_map(|entries| entries.iter())
                .filter(|e| e.composite_key == ck)
                .map(|e| e.amount)
                .sum();

            let current = settled + reserved;
            if current + estimated_tokens > policy.limit {
                return (
                    ReservationResult::Denied {
                        policy_name: policy.name.clone(),
                        strategy: policy.strategy.clone(),
                        current,
                        limit: policy.limit,
                    },
                    vec![], // atomic rollback — no entries returned
                );
            }

            pending_entries.push(ReservationEntry {
                composite_key: ck,
                amount: estimated_tokens,
            });
        }

        (ReservationResult::Granted, pending_entries)
    }

    /// Remove entries older than each policy's window.
    pub fn evict_expired(&mut self, policies: &[BudgetPolicyConfig], now: DateTime<Utc>) {
        for policy in policies {
            let window = match policy.window.as_ref().and_then(|w| parse_window(w)) {
                Some(d) => d,
                None => continue,
            };
            let cutoff = now - window;

            // Evict from all buckets matching this policy prefix
            let prefix = format!("{}|", policy.name);
            for (key, bucket) in &mut self.buckets {
                if key.starts_with(&prefix) {
                    while bucket
                        .entries
                        .front()
                        .is_some_and(|e| e.recorded_at < cutoff)
                    {
                        bucket.entries.pop_front();
                    }
                }
            }
        }
    }
}

/// Sum usage entries within the policy's window.
fn usage_in_window(bucket: &BucketState, window: &Option<String>, now: DateTime<Utc>) -> u64 {
    let cutoff = window
        .as_ref()
        .and_then(|w| parse_window(w))
        .map(|d| now - d);

    bucket
        .entries
        .iter()
        .filter(|e| cutoff.map_or(true, |c| e.recorded_at >= c))
        .map(|e| e.amount)
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_policies() -> Vec<BudgetPolicyConfig> {
        vec![
            BudgetPolicyConfig {
                name: "hourly".into(),
                group_by: vec!["user_id".into()],
                dimension: super::super::config::BudgetDimension::TotalTokens,
                limit: 10_000,
                window: Some("1h".into()),
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
            },
            BudgetPolicyConfig {
                name: "tenant_total".into(),
                group_by: vec![],
                dimension: super::super::config::BudgetDimension::TotalTokens,
                limit: 100_000,
                window: None,
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
            },
        ]
    }

    fn make_context(user_id: &str) -> BudgetContext {
        let mut ctx = BudgetContext::default();
        ctx.set("user_id", user_id);
        ctx
    }

    #[test]
    fn apply_usage_recorded() {
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();
        let sid = Uuid::new_v4();

        let event = BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "hourly".into(),
            bucket_key: "alice".into(),
            session_id: sid,
            call_id: "c1".into(),
            amount: 500,
            recorded_at: now,
        });

        ledger.apply(&event, 1);

        assert_eq!(ledger.stream_version, 1);
        let key = composite_key("hourly", "alice");
        let bucket = ledger.buckets.get(&key).unwrap();
        assert_eq!(bucket.entries.len(), 1);
        assert_eq!(bucket.entries[0].amount, 500);
    }

    #[test]
    fn try_reserve_within_limit() {
        let policies = make_policies();
        let ledger = BudgetLedger::default();
        let ctx = make_context("alice");
        let reservations = HashMap::new();
        let now = Utc::now();

        let (result, entries) = ledger.try_reserve(&policies, &ctx, 5000, &reservations, now);
        assert!(matches!(result, ReservationResult::Granted));
        assert_eq!(entries.len(), 2); // one for each matching policy
    }

    #[test]
    fn try_reserve_exceeds_limit() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let ctx = make_context("alice");
        let now = Utc::now();

        // Pre-fill with 9000 tokens
        ledger.apply(
            &BudgetEvent::UsageRecorded(UsageRecorded {
                policy_name: "hourly".into(),
                bucket_key: "alice".into(),
                session_id: Uuid::new_v4(),
                call_id: "c1".into(),
                amount: 9000,
                recorded_at: now,
            }),
            1,
        );

        let reservations = HashMap::new();
        let (result, entries) = ledger.try_reserve(&policies, &ctx, 2000, &reservations, now);
        assert!(matches!(result, ReservationResult::Denied { .. }));
        assert!(entries.is_empty(), "atomic rollback on deny");
    }

    #[test]
    fn try_reserve_counts_active_reservations() {
        let policies = make_policies();
        let ledger = BudgetLedger::default();
        let ctx = make_context("alice");
        let now = Utc::now();

        // Simulate existing reservation of 9000
        let mut reservations: HashMap<(Uuid, String), Vec<ReservationEntry>> = HashMap::new();
        reservations.insert(
            (Uuid::new_v4(), "prev-call".into()),
            vec![ReservationEntry {
                composite_key: composite_key("hourly", "alice"),
                amount: 9000,
            }],
        );

        let (result, _) = ledger.try_reserve(&policies, &ctx, 2000, &reservations, now);
        assert!(matches!(result, ReservationResult::Denied { .. }));
    }

    #[test]
    fn windowed_eviction() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();
        let old = now - Duration::hours(2);

        ledger.apply(
            &BudgetEvent::UsageRecorded(UsageRecorded {
                policy_name: "hourly".into(),
                bucket_key: "alice".into(),
                session_id: Uuid::new_v4(),
                call_id: "c1".into(),
                amount: 5000,
                recorded_at: old,
            }),
            1,
        );
        ledger.apply(
            &BudgetEvent::UsageRecorded(UsageRecorded {
                policy_name: "hourly".into(),
                bucket_key: "alice".into(),
                session_id: Uuid::new_v4(),
                call_id: "c2".into(),
                amount: 3000,
                recorded_at: now,
            }),
            2,
        );

        ledger.evict_expired(&policies, now);

        let key = composite_key("hourly", "alice");
        let bucket = ledger.buckets.get(&key).unwrap();
        assert_eq!(bucket.entries.len(), 1, "old entry should be evicted");
        assert_eq!(bucket.entries[0].amount, 3000);
    }

    #[test]
    fn usage_in_window_excludes_old() {
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();

        // Old entry (2 hours ago)
        ledger.apply(
            &BudgetEvent::UsageRecorded(UsageRecorded {
                policy_name: "hourly".into(),
                bucket_key: "alice".into(),
                session_id: Uuid::new_v4(),
                call_id: "c1".into(),
                amount: 8000,
                recorded_at: now - Duration::hours(2),
            }),
            1,
        );
        // Recent entry
        ledger.apply(
            &BudgetEvent::UsageRecorded(UsageRecorded {
                policy_name: "hourly".into(),
                bucket_key: "alice".into(),
                session_id: Uuid::new_v4(),
                call_id: "c2".into(),
                amount: 2000,
                recorded_at: now,
            }),
            2,
        );

        let policies = make_policies();
        let ctx = make_context("alice");
        let reservations = HashMap::new();

        // Should be allowed because old entry is outside window
        let (result, _) = ledger.try_reserve(&policies, &ctx, 5000, &reservations, now);
        assert!(matches!(result, ReservationResult::Granted));
    }

    #[test]
    fn match_conditions_filter() {
        let policies = vec![BudgetPolicyConfig {
            name: "opus_gate".into(),
            group_by: vec!["user_id".into()],
            dimension: super::super::config::BudgetDimension::TotalTokens,
            limit: 1000,
            window: Some("1h".into()),
            strategy: ExhaustionStrategy::Reject,
            match_conditions: Some(HashMap::from([("model".into(), "opus".into())])),
        }];

        let mut ctx = make_context("alice");
        ctx.set("model", "sonnet"); // does NOT match

        let ledger = BudgetLedger::default();
        let reservations = HashMap::new();
        let now = Utc::now();

        // Should be granted because the policy doesn't match (model != opus)
        let (result, entries) = ledger.try_reserve(&policies, &ctx, 5000, &reservations, now);
        assert!(matches!(result, ReservationResult::Granted));
        assert!(entries.is_empty(), "non-matching policy produces no entries");

        // Now with matching model
        ctx.set("model", "opus");
        let (result, _) = ledger.try_reserve(&policies, &ctx, 5000, &reservations, now);
        assert!(matches!(result, ReservationResult::Denied { .. }));
    }

    #[test]
    fn multi_policy_atomic_rollback() {
        // Policy 1: hourly limit 10k (passes)
        // Policy 2: tenant total 5k (fails)
        // Both should fail atomically
        let policies = vec![
            BudgetPolicyConfig {
                name: "hourly".into(),
                group_by: vec!["user_id".into()],
                dimension: super::super::config::BudgetDimension::TotalTokens,
                limit: 10_000,
                window: Some("1h".into()),
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
            },
            BudgetPolicyConfig {
                name: "tenant_total".into(),
                group_by: vec![],
                dimension: super::super::config::BudgetDimension::TotalTokens,
                limit: 5_000,
                window: None,
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
            },
        ];

        let ledger = BudgetLedger::default();
        let ctx = make_context("alice");
        let reservations = HashMap::new();
        let now = Utc::now();

        // 8000 exceeds tenant_total (5000) even though hourly (10000) would pass
        let (result, entries) = ledger.try_reserve(&policies, &ctx, 8000, &reservations, now);
        assert!(matches!(result, ReservationResult::Denied { policy_name, .. } if policy_name == "tenant_total"));
        assert!(entries.is_empty(), "atomic rollback");
    }

    #[test]
    fn bucket_key_empty_group_by() {
        let ctx = make_context("alice");
        assert_eq!(ctx.bucket_key(&[]), Some(String::new()));
    }

    #[test]
    fn bucket_key_missing_field() {
        let ctx = BudgetContext::default();
        assert_eq!(ctx.bucket_key(&["user_id".into()]), None);
    }

    #[test]
    fn bucket_key_multi_field() {
        let mut ctx = BudgetContext::default();
        ctx.set("user_id", "alice");
        ctx.set("model", "opus");
        let key = ctx.bucket_key(&["user_id".into(), "model".into()]);
        assert_eq!(key, Some("alice|opus".into()));
    }

    #[test]
    fn context_matches_empty() {
        let ctx = BudgetContext::default();
        assert!(ctx.matches(&HashMap::new()));
    }

    #[test]
    fn context_matches_mismatch() {
        let mut ctx = BudgetContext::default();
        ctx.set("model", "sonnet");
        let conditions = HashMap::from([("model".into(), "opus".into())]);
        assert!(!ctx.matches(&conditions));
    }
}
