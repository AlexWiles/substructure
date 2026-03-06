//! Worker interface — the contract between the runtime and worker implementations.
//!
//! All types are generated from `proto/worker.proto` (source of truth).
//! Internal code converts to/from these types at the boundary.

pub mod convert;
pub mod default;

// Generated proto types (prost + pbjson serde impls)
include!(concat!(env!("OUT_DIR"), "/worker.rs"));
include!(concat!(env!("OUT_DIR"), "/worker.serde.rs"));

// ---------------------------------------------------------------------------
// Worker trait — the decision-maker
// ---------------------------------------------------------------------------

/// A worker is a pure decision-maker.
///
/// Given a trigger (what happened), its opaque state, and a context snapshot,
/// produce actions for the runtime to execute and an updated state.
///
/// Implementations can be in-process (Rust) or remote (gRPC, Kafka).
pub trait Worker: Send + Sync + std::fmt::Debug {
    fn decide(
        &self,
        trigger: &DecisionTrigger,
        state: &[u8],
        ctx: &WorkerCtx,
    ) -> WorkerDecision;
}

