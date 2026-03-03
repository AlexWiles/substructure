pub(crate) mod actor;

pub use actor::{budget_actor_name, spawn_budget_actor};

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::aggregate::{AggregateState, Emit};
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
    #[serde(rename = "budget.reservation_created")]
    ReservationCreated(ReservationCreated),
    #[serde(rename = "budget.reservation_released")]
    ReservationReleased(ReservationReleased),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservationCreated {
    pub session_id: Uuid,
    pub call_id: String,
    pub estimated_tokens: u64,
    pub reserved_at: DateTime<Utc>,
    pub entries: Vec<ReservationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservationReleased {
    pub session_id: Uuid,
    pub call_id: String,
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
            .all(|(k, v)| self.values.get(k) == Some(v))
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

pub enum BudgetCommand {
    Reserve {
        session_id: Uuid,
        call_id: String,
        context: BudgetContext,
        estimated_tokens: u64,
        reserved_at: DateTime<Utc>,
    },
    RecordUsage {
        session_id: Uuid,
        call_id: String,
        total_tokens: u64,
        recorded_at: DateTime<Utc>,
    },
    ReleaseReservation {
        session_id: Uuid,
        call_id: String,
    },
    ReleaseSessionReservations {
        session_id: Uuid,
    },
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum BudgetError {
    Denied {
        policy_name: String,
        strategy: ExhaustionStrategy,
        current: u64,
        limit: u64,
    },
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetError::Denied {
                policy_name,
                current,
                limit,
                ..
            } => write!(f, "budget denied by {policy_name}: {current}/{limit}"),
        }
    }
}

impl std::error::Error for BudgetError {}

// ---------------------------------------------------------------------------
// Aggregate state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetLedger {
    pub buckets: HashMap<String, BucketState>,
    #[serde(default)]
    pub reservations: HashMap<String, Vec<ReservationEntry>>,
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

#[async_trait::async_trait]
impl AggregateState for BudgetLedger {
    type Event = BudgetEvent;
    type Command = BudgetCommand;
    type Error = BudgetError;
    type Context = Vec<BudgetPolicyConfig>;
    type Derived = BudgetDerived;

    fn aggregate_type() -> &'static str {
        "budget"
    }

    fn apply(&mut self, event: &BudgetEvent) {
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
            BudgetEvent::ReservationCreated(e) => {
                let key = reservation_key(e.session_id, &e.call_id);
                self.reservations.insert(key, e.entries.clone());
            }
            BudgetEvent::ReservationReleased(e) => {
                let key = reservation_key(e.session_id, &e.call_id);
                self.reservations.remove(&key);
            }
        }
    }

    fn handle_command(
        &self,
        cmd: Self::Command,
        ctx: &Self::Context,
    ) -> Result<Vec<Emit<Self::Event>>, Self::Error> {
        let policies = ctx;

        match cmd {
            BudgetCommand::Reserve {
                session_id,
                call_id,
                context,
                estimated_tokens,
                reserved_at,
            } => {
                let mut pending_entries = Vec::new();

                for policy in policies {
                    if let Some(ref conditions) = policy.match_conditions {
                        if !context.matches(conditions) {
                            continue;
                        }
                    }

                    let bucket_key = match context.bucket_key(&policy.group_by) {
                        Some(k) => k,
                        None => continue,
                    };

                    let ck = composite_key(&policy.name, &bucket_key);

                    let settled = self
                        .buckets
                        .get(&ck)
                        .map(|b| usage_in_window(b, &policy.window, reserved_at))
                        .unwrap_or(0);

                    let reserved: u64 = self
                        .reservations
                        .values()
                        .flat_map(|entries| entries.iter())
                        .filter(|e| e.composite_key == ck)
                        .map(|e| e.amount)
                        .sum();

                    let current = settled + reserved;
                    if current + estimated_tokens > policy.limit {
                        return Err(BudgetError::Denied {
                            policy_name: policy.name.clone(),
                            strategy: policy.strategy.clone(),
                            current,
                            limit: policy.limit,
                        });
                    }

                    pending_entries.push(ReservationEntry {
                        composite_key: ck,
                        amount: estimated_tokens,
                    });
                }

                if pending_entries.is_empty() {
                    return Ok(vec![]);
                }

                Ok(vec![Emit::new(BudgetEvent::ReservationCreated(
                    ReservationCreated {
                        session_id,
                        call_id,
                        estimated_tokens,
                        reserved_at,
                        entries: pending_entries,
                    },
                ))])
            }

            BudgetCommand::RecordUsage {
                session_id,
                call_id,
                total_tokens,
                recorded_at,
            } => {
                let rkey = reservation_key(session_id, &call_id);
                let mut emits: Vec<Emit<BudgetEvent>> = Vec::new();

                if let Some(entries) = self.reservations.get(&rkey) {
                    if total_tokens > 0 {
                        let mut seen = HashMap::new();
                        for entry in entries {
                            seen.entry(entry.composite_key.as_str()).or_insert(());
                        }
                        for ck in seen.keys() {
                            if let Some((policy_name, bucket_key)) = ck.split_once('|') {
                                emits.push(Emit::new(BudgetEvent::UsageRecorded(
                                    UsageRecorded {
                                        policy_name: policy_name.to_string(),
                                        bucket_key: bucket_key.to_string(),
                                        session_id,
                                        call_id: call_id.clone(),
                                        amount: total_tokens,
                                        recorded_at,
                                    },
                                )));
                            }
                        }
                    }

                    emits.push(Emit::new(BudgetEvent::ReservationReleased(
                        ReservationReleased {
                            session_id,
                            call_id,
                        },
                    )));
                }

                Ok(emits)
            }

            BudgetCommand::ReleaseReservation {
                session_id,
                call_id,
            } => {
                let rkey = reservation_key(session_id, &call_id);
                if self.reservations.contains_key(&rkey) {
                    Ok(vec![Emit::new(BudgetEvent::ReservationReleased(
                        ReservationReleased {
                            session_id,
                            call_id,
                        },
                    ))])
                } else {
                    Ok(vec![])
                }
            }

            BudgetCommand::ReleaseSessionReservations { session_id } => {
                let prefix = format!("{session_id}:");
                let emits: Vec<Emit<BudgetEvent>> = self
                    .reservations
                    .keys()
                    .filter(|k| k.starts_with(&prefix))
                    .filter_map(|k| {
                        let call_id = k.strip_prefix(&prefix)?.to_string();
                        Some(Emit::new(BudgetEvent::ReservationReleased(
                            ReservationReleased {
                                session_id,
                                call_id,
                            },
                        )))
                    })
                    .collect();
                Ok(emits)
            }
        }
    }

    async fn on_event(
        &self,
        _event: &Self::Event,
        _ctx: &Self::Context,
        _span: &crate::runtime::span::SpanContext,
    ) -> Option<Self::Command> {
        None
    }

    fn derived_state(&self) -> BudgetDerived {
        BudgetDerived {}
    }
}

/// Build the composite bucket key: "policy_name|bucket_key".
pub fn composite_key(policy_name: &str, bucket_key: &str) -> String {
    format!("{policy_name}|{bucket_key}")
}

/// Build the reservations HashMap key: "session_id:call_id".
pub fn reservation_key(session_id: Uuid, call_id: &str) -> String {
    format!("{session_id}:{call_id}")
}

// ---------------------------------------------------------------------------
// Reservation types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
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
// BudgetLedger helper methods
// ---------------------------------------------------------------------------

impl BudgetLedger {
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
        .filter(|e| cutoff.is_none_or(|c| e.recorded_at >= c))
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
                dimension: crate::runtime::config::BudgetDimension::TotalTokens,
                limit: 10_000,
                window: Some("1h".into()),
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
            },
            BudgetPolicyConfig {
                name: "tenant_total".into(),
                group_by: vec![],
                dimension: crate::runtime::config::BudgetDimension::TotalTokens,
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

    /// Helper: run a command through handle_command and apply all emitted events.
    fn execute(
        ledger: &mut BudgetLedger,
        cmd: BudgetCommand,
        policies: &[BudgetPolicyConfig],
    ) -> Result<Vec<Emit<BudgetEvent>>, BudgetError> {
        let emits = ledger.handle_command(cmd, &policies.to_vec())?;
        for emit in &emits {
            ledger.apply(&emit.event);
        }
        Ok(emits)
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

        ledger.apply(&event);

        let key = composite_key("hourly", "alice");
        let bucket = ledger.buckets.get(&key).unwrap();
        assert_eq!(bucket.entries.len(), 1);
        assert_eq!(bucket.entries[0].amount, 500);
    }

    #[test]
    fn apply_reservation_lifecycle() {
        let mut ledger = BudgetLedger::default();
        let sid = Uuid::new_v4();
        let now = Utc::now();

        // Create
        ledger.apply(&BudgetEvent::ReservationCreated(ReservationCreated {
            session_id: sid,
            call_id: "c1".into(),
            estimated_tokens: 500,
            reserved_at: now,
            entries: vec![ReservationEntry {
                composite_key: "hourly|alice".into(),
                amount: 500,
            }],
        }));
        assert_eq!(ledger.reservations.len(), 1);

        // Release
        ledger.apply(&BudgetEvent::ReservationReleased(ReservationReleased {
            session_id: sid,
            call_id: "c1".into(),
        }));
        assert!(ledger.reservations.is_empty());
    }

    #[test]
    fn reserve_within_limit() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();

        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c1".into(),
            context: make_context("alice"),
            estimated_tokens: 5000,
            reserved_at: now,
        };
        let emits = execute(&mut ledger, cmd, &policies).unwrap();
        assert_eq!(emits.len(), 1);
        match &emits[0].event {
            BudgetEvent::ReservationCreated(e) => {
                assert_eq!(e.entries.len(), 2); // one per matching policy
            }
            _ => panic!("expected ReservationCreated"),
        }
        assert_eq!(ledger.reservations.len(), 1);
    }

    #[test]
    fn reserve_exceeds_limit() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();

        // Pre-fill with 9000 tokens
        ledger.apply(&BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "hourly".into(),
            bucket_key: "alice".into(),
            session_id: Uuid::new_v4(),
            call_id: "c1".into(),
            amount: 9000,
            recorded_at: now,
        }));

        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            context: make_context("alice"),
            estimated_tokens: 2000,
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(matches!(result, Err(BudgetError::Denied { .. })));
    }

    #[test]
    fn reserve_counts_active_reservations() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();

        // Create an existing reservation of 9000
        ledger.apply(&BudgetEvent::ReservationCreated(ReservationCreated {
            session_id: Uuid::new_v4(),
            call_id: "prev".into(),
            estimated_tokens: 9000,
            reserved_at: now,
            entries: vec![ReservationEntry {
                composite_key: composite_key("hourly", "alice"),
                amount: 9000,
            }],
        }));

        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            context: make_context("alice"),
            estimated_tokens: 2000,
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(matches!(result, Err(BudgetError::Denied { .. })));
    }

    #[test]
    fn record_usage_settles_and_releases() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();
        let sid = Uuid::new_v4();

        // Reserve
        let cmd = BudgetCommand::Reserve {
            session_id: sid,
            call_id: "c1".into(),
            context: make_context("alice"),
            estimated_tokens: 500,
            reserved_at: now,
        };
        execute(&mut ledger, cmd, &policies).unwrap();
        assert_eq!(ledger.reservations.len(), 1);

        // Record actual usage
        let cmd = BudgetCommand::RecordUsage {
            session_id: sid,
            call_id: "c1".into(),
            total_tokens: 300,
            recorded_at: now,
        };
        let emits = execute(&mut ledger, cmd, &policies).unwrap();

        // Should have UsageRecorded events + ReservationReleased
        assert!(emits.iter().any(|e| matches!(e.event, BudgetEvent::UsageRecorded(_))));
        assert!(emits.iter().any(|e| matches!(e.event, BudgetEvent::ReservationReleased(_))));
        assert!(ledger.reservations.is_empty());
        assert!(!ledger.buckets.is_empty());
    }

    #[test]
    fn record_usage_zero_tokens_still_releases() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();
        let sid = Uuid::new_v4();

        // Reserve
        execute(
            &mut ledger,
            BudgetCommand::Reserve {
                session_id: sid,
                call_id: "c1".into(),
                context: make_context("alice"),
                estimated_tokens: 500,
                reserved_at: now,
            },
            &policies,
        )
        .unwrap();

        // Record 0 tokens
        let emits = execute(
            &mut ledger,
            BudgetCommand::RecordUsage {
                session_id: sid,
                call_id: "c1".into(),
                total_tokens: 0,
                recorded_at: now,
            },
            &policies,
        )
        .unwrap();

        // Should release but not record usage
        assert_eq!(emits.len(), 1);
        assert!(matches!(emits[0].event, BudgetEvent::ReservationReleased(_)));
        assert!(ledger.reservations.is_empty());
        assert!(ledger.buckets.is_empty());
    }

    #[test]
    fn release_reservation_on_error() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();
        let sid = Uuid::new_v4();

        execute(
            &mut ledger,
            BudgetCommand::Reserve {
                session_id: sid,
                call_id: "c1".into(),
                context: make_context("alice"),
                estimated_tokens: 500,
                reserved_at: now,
            },
            &policies,
        )
        .unwrap();
        assert_eq!(ledger.reservations.len(), 1);

        execute(
            &mut ledger,
            BudgetCommand::ReleaseReservation {
                session_id: sid,
                call_id: "c1".into(),
            },
            &policies,
        )
        .unwrap();
        assert!(ledger.reservations.is_empty());
    }

    #[test]
    fn release_reservation_idempotent() {
        let policies = make_policies();
        let ledger = BudgetLedger::default();

        let cmd = BudgetCommand::ReleaseReservation {
            session_id: Uuid::new_v4(),
            call_id: "nonexistent".into(),
        };
        let emits = ledger.handle_command(cmd, &policies.to_vec()).unwrap();
        assert!(emits.is_empty());
    }

    #[test]
    fn release_session_reservations() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();
        let sid = Uuid::new_v4();

        for call_id in ["c1", "c2"] {
            execute(
                &mut ledger,
                BudgetCommand::Reserve {
                    session_id: sid,
                    call_id: call_id.into(),
                    context: make_context("alice"),
                    estimated_tokens: 100,
                    reserved_at: now,
                },
                &policies,
            )
            .unwrap();
        }
        assert_eq!(ledger.reservations.len(), 2);

        execute(
            &mut ledger,
            BudgetCommand::ReleaseSessionReservations { session_id: sid },
            &policies,
        )
        .unwrap();
        assert!(ledger.reservations.is_empty());
    }

    #[test]
    fn windowed_eviction() {
        let policies = make_policies();
        let mut ledger = BudgetLedger::default();
        let now = Utc::now();
        let old = now - Duration::hours(2);

        ledger.apply(&BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "hourly".into(),
            bucket_key: "alice".into(),
            session_id: Uuid::new_v4(),
            call_id: "c1".into(),
            amount: 5000,
            recorded_at: old,
        }));
        ledger.apply(&BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "hourly".into(),
            bucket_key: "alice".into(),
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            amount: 3000,
            recorded_at: now,
        }));

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
        let policies = make_policies();

        // Old entry (2 hours ago)
        ledger.apply(&BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "hourly".into(),
            bucket_key: "alice".into(),
            session_id: Uuid::new_v4(),
            call_id: "c1".into(),
            amount: 8000,
            recorded_at: now - Duration::hours(2),
        }));
        // Recent entry
        ledger.apply(&BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "hourly".into(),
            bucket_key: "alice".into(),
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            amount: 2000,
            recorded_at: now,
        }));

        // Should be allowed because old entry is outside window
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c3".into(),
            context: make_context("alice"),
            estimated_tokens: 5000,
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(result.is_ok());
    }

    #[test]
    fn match_conditions_filter() {
        let policies = vec![BudgetPolicyConfig {
            name: "opus_gate".into(),
            group_by: vec!["user_id".into()],
            dimension: crate::runtime::config::BudgetDimension::TotalTokens,
            limit: 1000,
            window: Some("1h".into()),
            strategy: ExhaustionStrategy::Reject,
            match_conditions: Some(HashMap::from([("model".into(), "opus".into())])),
        }];

        let mut ctx = make_context("alice");
        ctx.set("model", "sonnet"); // does NOT match

        let ledger = BudgetLedger::default();
        let now = Utc::now();

        // Should be granted because the policy doesn't match (model != opus)
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c1".into(),
            context: ctx,
            estimated_tokens: 5000,
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty(), "non-matching policy produces no events");

        // Now with matching model
        let mut ctx2 = make_context("alice");
        ctx2.set("model", "opus");
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            context: ctx2,
            estimated_tokens: 5000,
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(matches!(result, Err(BudgetError::Denied { .. })));
    }

    #[test]
    fn multi_policy_atomic_rollback() {
        let policies = vec![
            BudgetPolicyConfig {
                name: "hourly".into(),
                group_by: vec!["user_id".into()],
                dimension: crate::runtime::config::BudgetDimension::TotalTokens,
                limit: 10_000,
                window: Some("1h".into()),
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
            },
            BudgetPolicyConfig {
                name: "tenant_total".into(),
                group_by: vec![],
                dimension: crate::runtime::config::BudgetDimension::TotalTokens,
                limit: 5_000,
                window: None,
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
            },
        ];

        let ledger = BudgetLedger::default();
        let now = Utc::now();

        // 8000 exceeds tenant_total (5000) even though hourly (10000) would pass
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c1".into(),
            context: make_context("alice"),
            estimated_tokens: 8000,
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(
            matches!(&result, Err(BudgetError::Denied { policy_name, .. }) if policy_name == "tenant_total")
        );
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
