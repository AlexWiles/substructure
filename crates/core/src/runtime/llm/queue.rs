use crate::runtime::llm::LlmRequest;
use crate::runtime::owner::SessionOwner;
use crate::runtime::span::SpanContext;

#[derive(Debug, Clone)]
pub struct LlmTask {
    pub session_id: String,
    pub tenant_id: String,
    pub agent_id: String,
    pub call_id: String,
    pub attempt: u32,
    pub request: LlmRequest,
    pub stream: bool,
    pub owner: SessionOwner,
    /// Parent chain, root-first. Empty for top-level sessions.
    pub ancestry: Vec<String>,
    /// Tagged on emitted token deltas so Turn-scoped subscribers can filter.
    pub turn_id: Option<String>,
    pub span: SpanContext,
}

impl LlmTask {
    pub fn dedupe_key(&self) -> String {
        format!("llm:{}:{}", self.session_id, self.call_id)
    }
}
