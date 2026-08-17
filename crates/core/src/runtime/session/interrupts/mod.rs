//! Interrupts the engine authors, keyed by an id prefix. A new kind is one
//! module and one entry in [`KINDS`]. Ids go on the wire unchanged, so a
//! prefix is a compatibility surface.

pub mod approval;
pub mod auth;

use serde_json::Value;

use super::propose::Proposing;
use crate::protocol::{DecisionAction, DecisionResponse};

pub trait InterruptKind: Sync {
    /// The id namespace, colon included.
    fn prefix(&self) -> &'static str;

    /// What the engine does when this interrupt resumes. `None` falls through
    /// to the generic resume.
    fn resumed(
        &self,
        _tail: &str,
        _payload: &Value,
        _p: &Proposing<'_>,
    ) -> Option<DecisionResponse> {
        None
    }

    /// Actions that go with resolving this interrupt.
    fn on_resolved(&self, _tail: &str) -> Vec<DecisionAction> {
        Vec::new()
    }
}

pub const KINDS: &[&dyn InterruptKind] = &[&approval::Approval, &auth::Auth];

pub fn kind_for(interrupt_id: &str) -> Option<(&'static dyn InterruptKind, &str)> {
    KINDS.iter().find_map(|kind| {
        interrupt_id
            .strip_prefix(kind.prefix())
            .map(|tail| (*kind, tail))
    })
}

/// An id that no kind owns adds nothing.
pub fn resolve_followups(interrupt_id: &str) -> Vec<DecisionAction> {
    kind_for(interrupt_id)
        .map(|(kind, tail)| kind.on_resolved(tail))
        .unwrap_or_default()
}
