use crate::runtime::identity::ClientIdentity;
use crate::runtime::llm::LlmRequest;
use crate::runtime::span::SpanContext;

#[derive(Debug, Clone)]
pub struct LlmTask {
    pub session_id: String,
    pub tenant_id: String,
    pub agent_id: String,
    pub call_id: String,
    pub attempt: u32,
    pub request: LlmRequest,
    pub identity: ClientIdentity,
    /// Parent chain, root-last. Empty for top-level sessions.
    pub ancestry: Vec<String>,
    pub span: SpanContext,
}

impl LlmTask {
    pub fn dedupe_key(&self) -> String {
        format!("llm:{}:{}", self.session_id, self.call_id)
    }
}
