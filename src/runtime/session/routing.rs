use std::sync::Arc;

use ractor::ActorRef;
use uuid::Uuid;

use crate::runtime::aggregate::DomainEvent;
use crate::runtime::aggregate::actor::AggregateMessage;
use crate::runtime::types::SessionMessage;
use super::client::Notification;
use super::state::AgentState;

// ---------------------------------------------------------------------------
// Naming conventions for ractor registry / process groups
// ---------------------------------------------------------------------------

pub fn aggregate_actor_name(session_id: Uuid) -> String {
    format!("session-{session_id}")
}

pub fn session_group(session_id: Uuid) -> String {
    format!("session-group-{session_id}")
}

pub fn session_observer_group(session_id: Uuid) -> String {
    format!("session-observers-{session_id}")
}

/// Routing closure for the session aggregate dispatcher.
/// Broadcasts typed events to the session process group and aggregate actor.
pub(crate) fn session_route(aggregate_id: Uuid, events: Vec<Arc<DomainEvent<AgentState>>>) {
    let group = session_group(aggregate_id);
    for cell in ractor::pg::get_members(&group) {
        let actor: ActorRef<SessionMessage> = cell.into();
        let _ = actor.send_message(SessionMessage::Events(events.clone()));
    }

    if let Some(cell) = ractor::registry::where_is(aggregate_actor_name(aggregate_id)) {
        let actor: ActorRef<AggregateMessage<AgentState>> = cell.into();
        let _ = actor.send_message(AggregateMessage::Events(events));
    }
}

/// Broadcast a transient notification to session observers only.
pub fn notify_observers(session_id: Uuid, notification: Arc<Notification>) {
    let group = session_observer_group(session_id);
    for cell in ractor::pg::get_members(&group) {
        let actor: ActorRef<SessionMessage> = cell.into();
        let _ = actor.send_message(SessionMessage::Notify(Arc::clone(&notification)));
    }
}
