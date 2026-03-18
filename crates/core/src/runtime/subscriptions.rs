use std::sync::Arc;

use tokio::sync::{broadcast, mpsc};

use super::aggregate::DomainEvent;
use super::event_store::{Event, EventFilter, EventStore};
use super::session::events::EventPayload;
use super::session::state::{DerivedState, SessionState};
use super::RuntimeError;

/// Manages event subscriptions for sessions: live streaming, catchup replay,
/// and combined catchup-then-live flows.
pub struct Subscriptions {
    store: Arc<dyn EventStore>,
}

impl Subscriptions {
    pub fn new(store: Arc<dyn EventStore>) -> Self {
        Self { store }
    }

    /// Subscribe to live events for a session (and descendants), filtered by session ID.
    /// Auto-closes when the given turn completes.
    pub fn subscribe_turn(
        &self,
        session_id: String,
        turn_id: String,
    ) -> mpsc::Receiver<Event> {
        let mut rx = self.store.subscribe();
        let (tx, event_rx) = mpsc::channel::<Event>(64);

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(batch) => {
                        for event in batch.iter() {
                            let is_self = event.aggregate_id == session_id;
                            let is_descendant = !is_self && is_descendant_of(event, &session_id);

                            if !is_self && !is_descendant {
                                continue;
                            }
                            if tx.send(event.clone()).await.is_err() {
                                return;
                            }
                            if is_self {
                                if let Ok(de) = DomainEvent::<SessionState>::from_raw(event) {
                                    if matches!(
                                        &de.payload,
                                        EventPayload::TurnCompleted(tc) if tc.turn_id == turn_id
                                    ) {
                                        return;
                                    }
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

    /// Subscribe to all events for a session (and descendants).
    /// Never auto-closes — stays open until the receiver is dropped.
    pub fn subscribe_all(
        &self,
        session_id: String,
    ) -> mpsc::Receiver<Event> {
        let mut rx = self.store.subscribe();
        let (tx, event_rx) = mpsc::channel::<Event>(64);

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(batch) => {
                        for event in batch.iter() {
                            let is_self = event.aggregate_id == session_id;
                            let is_descendant = !is_self && is_descendant_of(event, &session_id);

                            if !is_self && !is_descendant {
                                continue;
                            }
                            if tx.send(event.clone()).await.is_err() {
                                return;
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

    /// Catchup + live: load historical events for a turn, then chain with
    /// a live subscription. Deduplicates by sequence number since we subscribe
    /// before querying to avoid gaps.
    pub async fn catchup_turn(
        &self,
        session_id: String,
        turn_id: String,
    ) -> Result<(String, mpsc::Receiver<Event>), RuntimeError> {
        // Subscribe FIRST to capture future events, then load historical.
        let live_rx = self.subscribe_turn(session_id.clone(), turn_id.clone());

        let historical = self.load_turn_events(session_id.clone(), &turn_id).await;
        let max_seq = historical.last().map(|e| e.sequence).unwrap_or(0);

        let (tx, rx) = mpsc::channel(64);
        let sid = session_id.clone();
        tokio::spawn(async move {
            for event in historical {
                if tx.send(event).await.is_err() {
                    return;
                }
            }
            let mut live = live_rx;
            while let Some(event) = live.recv().await {
                if event.sequence <= max_seq {
                    continue;
                }
                if tx.send(event).await.is_err() {
                    return;
                }
            }
        });
        Ok((sid, rx))
    }

    /// Replay historical events for a completed turn (no live subscription).
    pub async fn replay_turn(
        &self,
        session_id: String,
        turn_id: String,
    ) -> Result<(String, mpsc::Receiver<Event>), RuntimeError> {
        let sid = session_id.clone();
        let historical = self.load_turn_events(session_id, &turn_id).await;

        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            for event in historical {
                if tx.send(event).await.is_err() {
                    return;
                }
            }
        });
        Ok((sid, rx))
    }

    /// Load events for a session filtered to a specific turn.
    async fn load_turn_events(&self, session_id: String, turn_id: &str) -> Vec<Event> {
        let all_events = self
            .store
            .query_events(&EventFilter {
                aggregate_id: Some(session_id.clone()),
                ..Default::default()
            })
            .await
            .unwrap_or_default();
        filter_by_turn(all_events, turn_id)
    }
}

/// Filter events to those belonging to a specific turn,
/// using the `turn_id` in each event's derived state.
fn filter_by_turn(events: Vec<Event>, turn_id: &str) -> Vec<Event> {
    events
        .into_iter()
        .filter(|e| {
            e.derived
                .as_ref()
                .and_then(|d| serde_json::from_value::<DerivedState>(d.clone()).ok())
                .and_then(|d| d.turn_id)
                .as_deref()
                == Some(turn_id)
        })
        .collect()
}

/// Check if an event belongs to a descendant session of the given session_id.
fn is_descendant_of(event: &Event, session_id: &str) -> bool {
    event
        .derived
        .as_ref()
        .and_then(|d| serde_json::from_value::<DerivedState>(d.clone()).ok())
        .map_or(false, |d| d.ancestry.iter().any(|a| a == session_id))
}
