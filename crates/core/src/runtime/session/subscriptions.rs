use std::sync::Arc;

use tokio::sync::{broadcast, mpsc};

use crate::runtime::event_store::{EventFilter, EventStore, Seq};
use crate::runtime::session::events::EventPayload;
use crate::runtime::Caller;

use super::SessionEvent;

/// Manages event subscriptions for sessions: live streaming, catchup replay,
/// and combined catchup-then-live flows.
pub struct SessionSubscriptions {
    store: Arc<dyn EventStore>,
}

/// Selects one session's events. Streams are strictly single-session — a
/// sub-agent runs as its own session with its own stream, and the parent's
/// log carries only the `sub_agent.*` boundary events. A client that wants
/// child detail opens a second stream with the child's session id (carried
/// by `sub_agent.requested`). This keeps `seq` unique and monotonic per
/// stream, so it doubles as the SSE frame id and resume cursor.
#[derive(Debug, Clone)]
pub struct SessionSubscriptionSpec {
    pub session_id: String,
    pub caller: Caller,
    pub scope: SubscriptionScope,
}

#[derive(Debug, Clone)]
pub enum SubscriptionScope {
    Turn { turn_id: String },
    All,
}

impl SessionSubscriptionSpec {
    fn include(&self, event: &SessionEvent) -> bool {
        match &self.scope {
            SubscriptionScope::Turn { turn_id } => {
                event.meta.turn_id.as_deref() == Some(turn_id.as_str())
            }
            SubscriptionScope::All => true,
        }
    }

    fn is_terminal(&self, event: &SessionEvent) -> bool {
        match &self.scope {
            SubscriptionScope::Turn { turn_id } => {
                event.session_id == self.session_id
                    && matches!(
                        &event.payload,
                        EventPayload::TurnCompleted(tc) if tc.turn_id == *turn_id
                    )
            }
            SubscriptionScope::All => false,
        }
    }
}

impl SessionSubscriptions {
    pub fn new(store: Arc<dyn EventStore>) -> Self {
        Self { store }
    }

    pub fn subscribe(&self, spec: SessionSubscriptionSpec) -> mpsc::Receiver<SessionEvent> {
        let mut rx = self.store.subscribe();
        let (tx, event_rx) = mpsc::channel::<SessionEvent>(64);

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(batch) => {
                        for event in batch.iter() {
                            if event.tenant_id != spec.caller.tenant_id() {
                                continue;
                            }
                            let in_scope = event.session_id == spec.session_id;
                            if in_scope && spec.include(event) {
                                if tx.send(event.clone()).await.is_err() {
                                    return;
                                }
                                if spec.is_terminal(event) {
                                    return;
                                }
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => return,
                }
            }
        });

        event_rx
    }

    /// Unified stream: live subscription, optionally preceded by historical
    /// replay from `after` (a per-stream cursor). Works for both Turn and All specs.
    ///
    /// - If `after` is `None`, yields live events only.
    /// - If `after` is `Some(v)`, replays historical events with
    ///   `seq > v` first, then streams live (deduped against historical).
    /// - For Turn specs, auto-closes when the turn completes. If historical
    ///   replay already contains the TurnCompleted event, live is skipped.
    pub async fn stream(
        &self,
        spec: SessionSubscriptionSpec,
        after: Option<Seq>,
    ) -> mpsc::Receiver<SessionEvent> {
        // Subscribe live FIRST so we don't miss anything between historical
        // load and live attach.
        let live_rx = self.subscribe(spec.clone());

        let historical = match after {
            Some(cursor) => self.load_historical(&spec, cursor).await,
            None => Vec::new(),
        };

        let max_seq = historical.last().map(|e| e.seq).unwrap_or(0);

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            for event in historical {
                let terminal = spec.is_terminal(&event);
                if tx.send(event).await.is_err() {
                    return;
                }
                if terminal {
                    return;
                }
            }
            let mut live = live_rx;
            while let Some(event) = live.recv().await {
                if event.seq <= max_seq {
                    continue;
                }
                if tx.send(event).await.is_err() {
                    return;
                }
            }
        });
        rx
    }

    async fn load_historical(
        &self,
        spec: &SessionSubscriptionSpec,
        after: Seq,
    ) -> Vec<SessionEvent> {
        let events = self
            .store
            .query_events(&EventFilter {
                tenant_id: Some(spec.caller.tenant_id().to_string()),
                session_id: Some(spec.session_id.clone()),
                after_seq: Some(after),
                ..Default::default()
            })
            .await
            .unwrap_or_default();
        filter_by_spec(events, spec)
    }
}

fn filter_by_spec(events: Vec<SessionEvent>, spec: &SessionSubscriptionSpec) -> Vec<SessionEvent> {
    events
        .into_iter()
        .filter(|event| {
            event.tenant_id == spec.caller.tenant_id()
                && event.session_id == spec.session_id
                && spec.include(event)
        })
        .collect()
}
