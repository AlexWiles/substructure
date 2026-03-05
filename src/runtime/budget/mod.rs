pub(crate) mod actor;

pub use actor::{budget_actor_name, spawn_budget_actor};

use std::collections::{BTreeMap, HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::aggregate::{AggregateState, Emit};
use super::config::{parse_window, BudgetPolicyConfig, ExhaustionStrategy};
use super::span::SpanContext;

/// A bag of named usage scalars. Keys are dimension names like
/// "prompt_tokens", "completion_tokens", "cost".
/// Extensible — new dimensions require no code changes.
pub type UsageBreakdown = BTreeMap<String, u64>;

/// Extract a single dimension's value from a breakdown, defaulting to 0.
fn extract(breakdown: &UsageBreakdown, dimension: &str) -> u64 {
    breakdown.get(dimension).copied().unwrap_or(0)
}

/// Recursively flatten a JSON value into named scalars.
/// Nested objects use dot-separated keys.
/// Integers stored as-is. Floats stored as micro-units (x 1_000_000)
/// to avoid floating point in the budget ledger.
pub fn flatten_usage(value: &serde_json::Value) -> UsageBreakdown {
    let mut out = BTreeMap::new();
    flatten_recursive(value, "", &mut out);
    out
}

fn flatten_recursive(v: &serde_json::Value, prefix: &str, out: &mut BTreeMap<String, u64>) {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(n) = n.as_u64() {
                out.insert(prefix.into(), n);
            } else if let Some(f) = n.as_f64() {
                // Cost fields: store as micro-units (6 decimal places)
                out.insert(prefix.into(), (f * 1_000_000.0) as u64);
            }
        }
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten_recursive(v, &key, out);
            }
        }
        _ => {}
    }
}

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
    pub breakdown: UsageBreakdown,
    pub recorded_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservationCreated {
    pub session_id: Uuid,
    pub call_id: String,
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

    /// Build a context for an LLM call from session fields.
    ///
    /// Populates the standard keys that budget policies can reference in
    /// `group_by` and `match` conditions: `tenant_id`, `user_id` (if present),
    /// `session_id`, `client_id`, `model`, plus any custom `auth.attrs`.
    pub fn for_llm_call(
        session_id: Uuid,
        auth: &super::config::ClientIdentity,
        client_id: &str,
        model: &str,
    ) -> Self {
        let mut ctx = Self::default();
        ctx.set("tenant_id", &auth.tenant_id);
        if let Some(ref sub) = auth.sub {
            ctx.set("user_id", sub);
        }
        for (k, v) in &auth.attrs {
            ctx.set(k.clone(), v.clone());
        }
        ctx.set("session_id", session_id.to_string());
        ctx.set("client_id", client_id);
        ctx.set("model", model);
        ctx
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
        breakdown: UsageBreakdown,
        reserved_at: DateTime<Utc>,
    },
    RecordUsage {
        session_id: Uuid,
        call_id: String,
        breakdown: UsageBreakdown,
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
// Budget status and denial types
// ---------------------------------------------------------------------------

/// Snapshot of a single budget policy's current state.
/// Shared base type for status queries and denial details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    pub policy_name: String,
    pub dimension: String,
    pub strategy: ExhaustionStrategy,
    pub current: u64,
    pub limit: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
}

/// A budget policy denial — a status snapshot plus the estimated usage
/// that triggered the rejection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetDenial {
    #[serde(flatten)]
    pub status: BudgetStatus,
    pub estimated: u64,
}

impl std::fmt::Display for BudgetDenial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "budget denied by {}: {}/{}",
            self.status.policy_name, self.status.current, self.status.limit
        )
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum BudgetError {
    Denied(BudgetDenial),
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetError::Denied(d) => write!(f, "{d}"),
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
    pub breakdown: UsageBreakdown,
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
                    breakdown: e.breakdown.clone(),
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
                breakdown,
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
                        .map(|b| usage_in_window(b, &policy.dimension, &policy.window, reserved_at))
                        .unwrap_or(0);

                    let reserved: u64 = self
                        .reservations
                        .values()
                        .flat_map(|entries| entries.iter())
                        .filter(|e| e.composite_key == ck)
                        .map(|e| extract(&e.breakdown, &policy.dimension))
                        .sum();

                    let estimated = extract(&breakdown, &policy.dimension);
                    let current = settled + reserved;
                    if current + estimated > policy.limit {
                        return Err(BudgetError::Denied(BudgetDenial {
                            status: BudgetStatus {
                                policy_name: policy.name.clone(),
                                dimension: policy.dimension.clone(),
                                strategy: policy.strategy.clone(),
                                current,
                                limit: policy.limit,
                                window: policy.window.clone(),
                            },
                            estimated,
                        }));
                    }

                    pending_entries.push(ReservationEntry {
                        composite_key: ck,
                        breakdown: breakdown.clone(),
                    });
                }

                if pending_entries.is_empty() {
                    return Ok(vec![]);
                }

                Ok(vec![Emit::new(BudgetEvent::ReservationCreated(
                    ReservationCreated {
                        session_id,
                        call_id,
                        reserved_at,
                        entries: pending_entries,
                    },
                ))])
            }

            BudgetCommand::RecordUsage {
                session_id,
                call_id,
                breakdown,
                recorded_at,
            } => {
                let rkey = reservation_key(session_id, &call_id);
                let mut emits: Vec<Emit<BudgetEvent>> = Vec::new();

                if let Some(entries) = self.reservations.get(&rkey) {
                    let has_usage = breakdown.values().any(|&v| v > 0);
                    if has_usage {
                        let mut seen = HashMap::new();
                        for entry in entries {
                            seen.entry(entry.composite_key.as_str()).or_insert(());
                        }
                        for ck in seen.keys() {
                            if let Some((policy_name, bucket_key)) = ck.split_once('|') {
                                emits.push(Emit::new(BudgetEvent::UsageRecorded(UsageRecorded {
                                    policy_name: policy_name.to_string(),
                                    bucket_key: bucket_key.to_string(),
                                    session_id,
                                    call_id: call_id.clone(),
                                    breakdown: breakdown.clone(),
                                    recorded_at,
                                })));
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
        _span: &SpanContext,
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
    pub breakdown: UsageBreakdown,
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

/// Sum usage entries within the policy's window for a specific dimension.
fn usage_in_window(
    bucket: &BucketState,
    dimension: &str,
    window: &Option<String>,
    now: DateTime<Utc>,
) -> u64 {
    let cutoff = window
        .as_ref()
        .and_then(|w| parse_window(w))
        .map(|d| now - d);

    bucket
        .entries
        .iter()
        .filter(|e| cutoff.is_none_or(|c| e.recorded_at >= c))
        .map(|e| extract(&e.breakdown, dimension))
        .sum()
}

// ---------------------------------------------------------------------------
// BudgetActorRef — async reserve method (impl here to keep ractor out of session)
// ---------------------------------------------------------------------------

use super::aggregate::actor::{AggregateActorHandle, AggregateError};
use super::session::BudgetActorRef;

impl BudgetActorRef {
    /// Attempt to reserve budget against policies before an LLM call.
    ///
    /// Returns `Ok(())` if granted or if no matching policies exist.
    /// Returns `Err(BudgetError::Denied(_))` when a policy rejects.
    /// Fails open on infrastructure errors (timeouts, actor crashes).
    pub(crate) async fn reserve(
        &self,
        session_id: Uuid,
        call_id: &str,
        context: BudgetContext,
        breakdown: UsageBreakdown,
        span: &SpanContext,
    ) -> Result<(), BudgetError> {
        let Some(handle) = self
            .inner
            .downcast_ref::<AggregateActorHandle<BudgetLedger>>()
        else {
            tracing::error!("BudgetActorRef contains unexpected type — skipping reservation");
            return Ok(());
        };

        match handle
            .send_command(
                BudgetCommand::Reserve {
                    session_id,
                    call_id: call_id.to_string(),
                    context,
                    breakdown,
                    reserved_at: Utc::now(),
                },
                span.clone(),
                Utc::now(),
            )
            .await
        {
            Ok(_) => Ok(()),
            Err(AggregateError::Command(e)) => Err(e),
            Err(other) => {
                tracing::warn!(error = %other, "budget reserve failed — proceeding without reservation");
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    /// Helper to build a UsageBreakdown with a single "total_tokens" key.
    fn tokens(n: u64) -> UsageBreakdown {
        BTreeMap::from([("total_tokens".into(), n)])
    }

    fn make_policies() -> Vec<BudgetPolicyConfig> {
        vec![
            BudgetPolicyConfig {
                name: "hourly".into(),
                group_by: vec!["user_id".into()],
                dimension: "total_tokens".into(),
                limit: 10_000,
                window: Some("1h".into()),
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
                value_type: Default::default(),
            },
            BudgetPolicyConfig {
                name: "tenant_total".into(),
                group_by: vec![],
                dimension: "total_tokens".into(),
                value_type: Default::default(),
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
            breakdown: tokens(500),
            recorded_at: now,
        });

        ledger.apply(&event);

        let key = composite_key("hourly", "alice");
        let bucket = ledger.buckets.get(&key).unwrap();
        assert_eq!(bucket.entries.len(), 1);
        assert_eq!(extract(&bucket.entries[0].breakdown, "total_tokens"), 500);
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
            reserved_at: now,
            entries: vec![ReservationEntry {
                composite_key: "hourly|alice".into(),
                breakdown: tokens(500),
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
            breakdown: tokens(5000),
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
            breakdown: tokens(9000),
            recorded_at: now,
        }));

        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            context: make_context("alice"),
            breakdown: tokens(2000),
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(matches!(result, Err(BudgetError::Denied(_))));
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
            reserved_at: now,
            entries: vec![ReservationEntry {
                composite_key: composite_key("hourly", "alice"),
                breakdown: tokens(9000),
            }],
        }));

        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            context: make_context("alice"),
            breakdown: tokens(2000),
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(matches!(result, Err(BudgetError::Denied(_))));
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
            breakdown: tokens(500),
            reserved_at: now,
        };
        execute(&mut ledger, cmd, &policies).unwrap();
        assert_eq!(ledger.reservations.len(), 1);

        // Record actual usage
        let cmd = BudgetCommand::RecordUsage {
            session_id: sid,
            call_id: "c1".into(),
            breakdown: tokens(300),
            recorded_at: now,
        };
        let emits = execute(&mut ledger, cmd, &policies).unwrap();

        // Should have UsageRecorded events + ReservationReleased
        assert!(emits
            .iter()
            .any(|e| matches!(e.event, BudgetEvent::UsageRecorded(_))));
        assert!(emits
            .iter()
            .any(|e| matches!(e.event, BudgetEvent::ReservationReleased(_))));
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
                breakdown: tokens(500),
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
                breakdown: BTreeMap::new(),
                recorded_at: now,
            },
            &policies,
        )
        .unwrap();

        // Should release but not record usage
        assert_eq!(emits.len(), 1);
        assert!(matches!(
            emits[0].event,
            BudgetEvent::ReservationReleased(_)
        ));
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
                breakdown: tokens(500),
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
                    breakdown: tokens(100),
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
            breakdown: tokens(5000),
            recorded_at: old,
        }));
        ledger.apply(&BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "hourly".into(),
            bucket_key: "alice".into(),
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            breakdown: tokens(3000),
            recorded_at: now,
        }));

        ledger.evict_expired(&policies, now);

        let key = composite_key("hourly", "alice");
        let bucket = ledger.buckets.get(&key).unwrap();
        assert_eq!(bucket.entries.len(), 1, "old entry should be evicted");
        assert_eq!(extract(&bucket.entries[0].breakdown, "total_tokens"), 3000);
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
            breakdown: tokens(8000),
            recorded_at: now - Duration::hours(2),
        }));
        // Recent entry
        ledger.apply(&BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "hourly".into(),
            bucket_key: "alice".into(),
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            breakdown: tokens(2000),
            recorded_at: now,
        }));

        // Should be allowed because old entry is outside window
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c3".into(),
            context: make_context("alice"),
            breakdown: tokens(5000),
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
            dimension: "total_tokens".into(),
            value_type: Default::default(),
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
            breakdown: tokens(5000),
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_empty(),
            "non-matching policy produces no events"
        );

        // Now with matching model
        let mut ctx2 = make_context("alice");
        ctx2.set("model", "opus");
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            context: ctx2,
            breakdown: tokens(5000),
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(matches!(result, Err(BudgetError::Denied(_))));
    }

    #[test]
    fn multi_policy_atomic_rollback() {
        let policies = vec![
            BudgetPolicyConfig {
                name: "hourly".into(),
                group_by: vec!["user_id".into()],
                dimension: "total_tokens".into(),
                value_type: Default::default(),
                limit: 10_000,
                window: Some("1h".into()),
                strategy: ExhaustionStrategy::Reject,
                match_conditions: None,
            },
            BudgetPolicyConfig {
                name: "tenant_total".into(),
                group_by: vec![],
                dimension: "total_tokens".into(),
                value_type: Default::default(),
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
            breakdown: tokens(8000),
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(
            matches!(&result, Err(BudgetError::Denied(d)) if d.status.policy_name == "tenant_total")
        );
    }

    #[test]
    fn dimension_specific_policy() {
        // Policy on "completion_tokens" should ignore "prompt_tokens" usage
        let policies = vec![BudgetPolicyConfig {
            name: "output_cap".into(),
            group_by: vec!["user_id".into()],
            dimension: "completion_tokens".into(),
            value_type: Default::default(),
            limit: 1000,
            window: None,
            strategy: ExhaustionStrategy::Reject,
            match_conditions: None,
        }];

        let mut ledger = BudgetLedger::default();
        let now = Utc::now();

        // Record large prompt_tokens usage — should not affect completion_tokens policy
        ledger.apply(&BudgetEvent::UsageRecorded(UsageRecorded {
            policy_name: "output_cap".into(),
            bucket_key: "alice".into(),
            session_id: Uuid::new_v4(),
            call_id: "c1".into(),
            breakdown: BTreeMap::from([
                ("prompt_tokens".into(), 50_000),
                ("completion_tokens".into(), 100),
            ]),
            recorded_at: now,
        }));

        // Reserve with large prompt but small completion — should pass
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c2".into(),
            context: make_context("alice"),
            breakdown: BTreeMap::from([
                ("prompt_tokens".into(), 50_000),
                ("completion_tokens".into(), 500),
            ]),
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(result.is_ok(), "should pass: 100 + 500 = 600 < 1000");

        // Reserve that would exceed completion_tokens limit
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c3".into(),
            context: make_context("alice"),
            breakdown: BTreeMap::from([
                ("prompt_tokens".into(), 50_000),
                ("completion_tokens".into(), 1000),
            ]),
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(matches!(result, Err(BudgetError::Denied(_))));
    }

    #[test]
    fn missing_dimension_defaults_to_zero() {
        let policies = vec![BudgetPolicyConfig {
            name: "reasoning_cap".into(),
            group_by: vec!["user_id".into()],
            dimension: "completion_tokens_details.reasoning_tokens".into(),
            value_type: Default::default(),
            limit: 1000,
            window: None,
            strategy: ExhaustionStrategy::Reject,
            match_conditions: None,
        }];

        let ledger = BudgetLedger::default();
        let now = Utc::now();

        // Breakdown without reasoning_tokens — should default to 0 and pass
        let cmd = BudgetCommand::Reserve {
            session_id: Uuid::new_v4(),
            call_id: "c1".into(),
            context: make_context("alice"),
            breakdown: BTreeMap::from([
                ("prompt_tokens".into(), 10_000),
                ("completion_tokens".into(), 5000),
            ]),
            reserved_at: now,
        };
        let result = ledger.handle_command(cmd, &policies.to_vec());
        assert!(result.is_ok());
    }

    #[test]
    fn flatten_usage_basic() {
        let json = serde_json::json!({
            "prompt_tokens": 194,
            "completion_tokens": 2,
            "total_tokens": 196,
        });
        let flat = flatten_usage(&json);
        assert_eq!(flat.get("prompt_tokens"), Some(&194));
        assert_eq!(flat.get("completion_tokens"), Some(&2));
        assert_eq!(flat.get("total_tokens"), Some(&196));
    }

    #[test]
    fn flatten_usage_nested() {
        let json = serde_json::json!({
            "prompt_tokens": 100,
            "prompt_tokens_details": {
                "cached_tokens": 50,
                "cache_write_tokens": 10,
            }
        });
        let flat = flatten_usage(&json);
        assert_eq!(flat.get("prompt_tokens"), Some(&100));
        assert_eq!(flat.get("prompt_tokens_details.cached_tokens"), Some(&50));
        assert_eq!(
            flat.get("prompt_tokens_details.cache_write_tokens"),
            Some(&10)
        );
    }

    #[test]
    fn flatten_usage_float_to_micro_units() {
        let json = serde_json::json!({
            "cost": 0.95,
            "total_tokens": 100,
        });
        let flat = flatten_usage(&json);
        assert_eq!(flat.get("cost"), Some(&950_000));
        assert_eq!(flat.get("total_tokens"), Some(&100));
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
