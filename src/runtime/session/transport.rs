use ractor::ActorRef;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::runtime::session::command::CommandPayload;
use crate::runtime::span::SpanContext;
use crate::runtime::types::RuntimeMessage;

use super::default_strategy::DefaultStrategy;
use super::strategy::{DecisionTrigger, Strategy, StrategyCtx};

// ---------------------------------------------------------------------------
// Strategy dispatch — the wire message
// ---------------------------------------------------------------------------

/// Serializable request sent to a strategy executor (local or remote).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDispatch {
    pub session_id: Uuid,
    pub decision_id: String,
    pub trigger: DecisionTrigger,
    pub strategy_state: serde_json::Value,
    pub ctx: StrategyCtx,
    pub span: SpanContext,
}

// ---------------------------------------------------------------------------
// Strategy transport — abstracts local vs remote strategy execution
// ---------------------------------------------------------------------------

/// Transport abstraction for strategy execution.
///
/// The session aggregate calls `dispatch()` and returns — fire-and-forget.
/// The transport handles execution and delivers the result back via
/// `RuntimeMessage::DeliverToSession`.
pub trait StrategyTransport: Send + Sync {
    fn dispatch(&self, request: StrategyDispatch);
}

/// Executes strategies in-process and delivers results back via the runtime actor.
pub struct LocalStrategyTransport {
    pub runtime: ActorRef<RuntimeMessage>,
}

impl StrategyTransport for LocalStrategyTransport {
    fn dispatch(&self, request: StrategyDispatch) {
        let strategy = resolve_strategy(&request.ctx.agent.strategy);
        let decision = strategy.decide(
            &request.trigger,
            &request.strategy_state,
            &request.ctx,
        );
        let _ = self.runtime.send_message(RuntimeMessage::DeliverToSession {
            session_id: request.session_id,
            payload: CommandPayload::SubmitStrategyDecision {
                decision_id: request.decision_id,
                actions: decision.actions,
                state: decision.state,
            },
            span: request.span,
        });
    }
}

// ---------------------------------------------------------------------------
// Strategy resolution
// ---------------------------------------------------------------------------

use crate::runtime::config::StrategyConfig;

/// Resolve a strategy implementation from agent config.
pub fn resolve_strategy(_config: &StrategyConfig) -> Box<dyn Strategy> {
    // V1: only the default strategy. Future: match on config.kind.
    Box::new(DefaultStrategy)
}
