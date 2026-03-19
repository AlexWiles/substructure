use std::sync::Arc;

use tokio::sync::{broadcast, mpsc};

use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::event_store::{Event, EventFilter, EventStore};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::SessionState;
use crate::runtime::RuntimeError;

/// Manages event subscriptions for sessions: live streaming, catchup replay,
/// and combined catchup-then-live flows.
pub struct SessionSubscriptions {
    store: Arc<dyn EventStore>,
}

#[derive(Debug, Clone)]
pub enum SessionSubscriptionSpec {
    Turn {
        root_session_id: String,
        turn_id: String,
    },
    All {
        root_session_id: String,
    },
}

impl SessionSubscriptionSpec {
    fn root_session_id(&self) -> &str {
        match self {
            Self::Turn {
                root_session_id, ..
            }
            | Self::All { root_session_id } => root_session_id,
        }
    }

    fn include(&self, event: &DomainEvent<SessionState>) -> bool {
        match self {
            Self::Turn { turn_id, .. } => {
                event.derived.as_ref().and_then(|d| d.turn_id.as_deref()) == Some(turn_id.as_str())
            }
            Self::All { .. } => true,
        }
    }

    fn is_terminal(&self, event: &DomainEvent<SessionState>) -> bool {
        match self {
            Self::Turn {
                root_session_id,
                turn_id,
            } => {
                event.aggregate_id == *root_session_id
                    && matches!(
                        &event.payload,
                        EventPayload::TurnCompleted(tc) if tc.turn_id == *turn_id
                    )
            }
            Self::All { .. } => false,
        }
    }

    fn turn_scope(&self) -> Option<(&str, &str)> {
        match self {
            Self::Turn {
                root_session_id,
                turn_id,
            } => Some((root_session_id.as_str(), turn_id.as_str())),
            Self::All { .. } => None,
        }
    }
}

impl SessionSubscriptions {
    pub fn new(store: Arc<dyn EventStore>) -> Self {
        Self { store }
    }

    pub fn subscribe(&self, spec: SessionSubscriptionSpec) -> mpsc::Receiver<Event> {
        let mut rx = self.store.subscribe();
        let (tx, event_rx) = mpsc::channel::<Event>(64);

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(batch) => {
                        for raw in batch.iter() {
                            if let Some(event) = decode_session_event(raw) {
                                let in_scope =
                                    belongs_to_root_session(raw, &event, spec.root_session_id());
                                if in_scope && spec.include(&event) {
                                    if tx.send(raw.clone()).await.is_err() {
                                        return;
                                    }
                                    if spec.is_terminal(&event) {
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

    /// Catchup + live: load historical events for a turn, then chain with
    /// a live subscription. Deduplicates by sequence number since we subscribe
    /// before querying to avoid gaps.
    pub async fn catchup(
        &self,
        spec: SessionSubscriptionSpec,
    ) -> Result<(String, mpsc::Receiver<Event>), RuntimeError> {
        let (session_id, turn_id) = match spec.turn_scope() {
            Some((session_id, turn_id)) => (session_id.to_string(), turn_id.to_string()),
            None => {
                return Err(RuntimeError(
                    "catchup currently requires turn-scoped session subscription".into(),
                ));
            }
        };

        // Subscribe FIRST to capture future events, then load historical.
        let live_rx = self.subscribe(spec.clone());

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
    pub async fn replay(
        &self,
        spec: SessionSubscriptionSpec,
    ) -> Result<(String, mpsc::Receiver<Event>), RuntimeError> {
        let (session_id, turn_id) = match spec.turn_scope() {
            Some((session_id, turn_id)) => (session_id.to_string(), turn_id.to_string()),
            None => {
                return Err(RuntimeError(
                    "replay currently requires turn-scoped session subscription".into(),
                ));
            }
        };

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
        let spec = SessionSubscriptionSpec::Turn {
            root_session_id: session_id,
            turn_id: turn_id.to_string(),
        };
        filter_by_spec(all_events, &spec)
    }
}

fn filter_by_spec(events: Vec<Event>, spec: &SessionSubscriptionSpec) -> Vec<Event> {
    events
        .into_iter()
        .filter(|e| {
            let Some(event) = decode_session_event(e) else {
                return false;
            };
            if !belongs_to_root_session(e, &event, spec.root_session_id()) {
                return false;
            }
            spec.include(&event)
        })
        .collect()
}

fn decode_session_event(raw: &Event) -> Option<DomainEvent<SessionState>> {
    if raw.aggregate_type != SessionState::AGGREGATE_TYPE {
        return None;
    }
    DomainEvent::<SessionState>::from_raw(raw).ok()
}

fn belongs_to_root_session(
    raw: &Event,
    event: &DomainEvent<SessionState>,
    root_session_id: &str,
) -> bool {
    if raw.aggregate_id == root_session_id {
        return true;
    }
    event
        .derived
        .as_ref()
        .is_some_and(|d| d.ancestry.iter().any(|a| a == root_session_id))
}
