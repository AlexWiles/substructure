use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent_state::AgentState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub core: AgentState,
    pub strategy_state: Value,
}
