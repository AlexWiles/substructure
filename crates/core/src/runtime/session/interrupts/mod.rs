//! Interrupt kinds the engine itself authors, keyed by an id prefix.
//!
//! A kind owns its id namespace, the engine's continuation when its interrupt
//! resumes, and the actions that ride along when a channel resolves one. Ids
//! reach the wire unchanged, so a kind's format is a compatibility surface.
//! Adding a kind is one module plus one entry in [`KINDS`]; `propose` and the
//! channel proposers dispatch through here and name no kind directly.

pub mod approval;
pub mod auth;

use serde_json::Value;

use super::propose::Proposing;
use crate::protocol::{DecisionAction, DecisionResponse};

pub trait InterruptKind: Sync {
    /// The id namespace, colon included.
    fn prefix(&self) -> &'static str;

    /// The engine's continuation when this interrupt resumes. `None` falls
    /// through to the generic resume (re-prompt over the recorded path).
    fn resumed(
        &self,
        _tail: &str,
        _payload: &Value,
        _p: &Proposing<'_>,
    ) -> Option<DecisionResponse> {
        None
    }

    /// Actions a channel adds alongside resolving this interrupt.
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

/// What resolving `interrupt_id` runs besides the resolution itself.
/// Ids no kind owns — a worker's own prompts — add nothing.
pub fn resolve_followups(interrupt_id: &str) -> Vec<DecisionAction> {
    kind_for(interrupt_id)
        .map(|(kind, tail)| kind.on_resolved(tail))
        .unwrap_or_default()
}
