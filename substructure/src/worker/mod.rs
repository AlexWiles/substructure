//! Worker interface — the contract between the runtime and worker implementations.
//!
//! All types are generated from `proto/worker.proto` (source of truth).
//! Internal code converts to/from these types at the boundary.

pub mod default;

// Generated proto types (prost + pbjson serde impls)
#[allow(clippy::large_enum_variant)]
mod proto_gen {
    include!(concat!(env!("OUT_DIR"), "/worker.rs"));
    include!(concat!(env!("OUT_DIR"), "/worker.serde.rs"));
}
pub use proto_gen::*;

// Generated tonic gRPC stubs for WorkerGateway service
include!(concat!(env!("OUT_DIR"), "/worker.WorkerGateway.rs"));

// ---------------------------------------------------------------------------
// Worker trait — the decision-maker and tool executor
// ---------------------------------------------------------------------------

/// A worker is a decision-maker and tool executor.
///
/// Given a trigger (what happened), its opaque state, and a context snapshot,
/// produce actions for the runtime to execute and an updated state.
///
/// Workers also execute tool calls — the runtime dispatches tool calls to the
/// worker, which handles them using its own tool implementations (MCP, HTTP, etc.).
///
/// Implementations can be in-process (Rust) or remote (gRPC).
#[async_trait::async_trait]
pub trait Worker: Send + Sync + std::fmt::Debug {
    /// Return the agent names this worker handles.
    /// The runtime uses this to validate session creation and sub-agent routing.
    fn agent_names(&self) -> Vec<String>;

    fn decide(&self, trigger: &DecisionTrigger, state: &[u8], ctx: &WorkerCtx) -> WorkerDecision;

    async fn execute_tool_call(&self, dispatch: &ToolCallDispatch) -> ToolCallResult;
}
