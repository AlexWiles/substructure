use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent_state::AgentState;
use super::call_tracker::CallTracker;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub core: AgentState,
    pub tracker: CallTracker,
    pub strategy_state: Value,
}
