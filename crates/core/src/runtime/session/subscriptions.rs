use std::sync::Arc;

use tokio::sync::{broadcast, mpsc};

use crate::runtime::aggregate::{AggregateState, DomainEvent};
use crate::runtime::event_store::{Event, EventFilter, EventStore};
use crate::runtime::session::events::EventPayload;
use crate::runtime::session::state::SessionState;

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

    /// Unified stream: live subscription, optionally preceded by historical
    /// replay from `sequence_after`. Works for both Turn and All specs.
    ///
    /// - If `sequence_after` is `None`, yields live events only.
    /// - If `sequence_after` is `Some(n)`, replays historical events with
    ///   `sequence > n` first, then streams live (deduped against historical).
    /// - For Turn specs, auto-closes when the turn completes. If historical
    ///   replay already contains the TurnCompleted event, live is skipped.
    pub async fn stream(
        &self,
        spec: SessionSubscriptionSpec,
        sequence_after: Option<u64>,
    ) -> mpsc::Receiver<Event> {
        // Subscribe live FIRST so we don't miss anything between historical
        // load and live attach.
        let live_rx = self.subscribe(spec.clone());

        let historical = match sequence_after {
            Some(cursor) => self.load_historical(&spec, cursor).await,
            None => Vec::new(),
        };

        let max_seq = historical.last().map(|e| e.sequence).unwrap_or(0);
        let already_complete = matches!(&spec, SessionSubscriptionSpec::Turn { .. })
            && historical.iter().any(|e| is_turn_completed_for(e, &spec));

        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            for event in historical {
                if tx.send(event).await.is_err() {
                    return;
                }
            }
            if already_complete {
                return;
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
        rx
    }

    async fn load_historical(
        &self,
        spec: &SessionSubscriptionSpec,
        sequence_after: u64,
    ) -> Vec<Event> {
        let events = self
            .store
            .query_events(&EventFilter {
                aggregate_id: Some(spec.root_session_id().to_string()),
                sequence_after: Some(sequence_after),
                ..Default::default()
            })
            .await
            .unwrap_or_default();
        filter_by_spec(events, spec)
    }
}

fn is_turn_completed_for(raw: &Event, spec: &SessionSubscriptionSpec) -> bool {
    let Some(event) = decode_session_event(raw) else {
        return false;
    };
    spec.is_terminal(&event)
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
