use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::protocol::{
    DeferToolsStrategy, ErrorCode, ErrorInfo, LlmResponse, LlmTool, PromptRequest, SessionOwner,
    StreamDelta,
};

impl LlmTool {
    /// The schema providers receive: the declared `input`, or the empty
    /// object schema for a no-argument tool.
    pub fn input_schema(&self) -> serde_json::Value {
        self.input
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
    }
}

/// Trait for LLM client providers (resolved by the runtime).
#[async_trait]
pub trait LlmProviderTrait: Send + Sync {
    async fn resolve(&self, owner: &SessionOwner) -> Result<Arc<dyn LlmCallable>, String>;
}

/// What the executor asks for a client: the block a call names, resolved for
/// the session that made it.
///
/// A deployment serving one declaration answers from a static registry; one
/// serving many resolves per owner, because the same block name means a
/// different credential for each.
#[async_trait]
pub trait LlmResolver: Send + Sync {
    async fn resolve(
        &self,
        llm: &str,
        owner: &SessionOwner,
    ) -> Result<Arc<dyn LlmCallable>, String>;

    /// `true` ⇒ nothing can ever resolve, so the engine-side LLM subsystem is
    /// not started. A resolver that only learns its blocks at call time says
    /// `false`.
    fn is_empty(&self) -> bool {
        false
    }
}

/// The engine-executed half of the declared `[llm.*]` blocks: one client per
/// block the engine calls itself, keyed by the name a call references.
///
/// Blocks the agent's worker runs are absent by construction — they never need
/// a credential where the engine runs — so a name missing here is either
/// undeclared or worker-run, and either way not the engine's call to make.
#[derive(Default)]
pub struct LlmProviderRegistry {
    providers: BTreeMap<String, Arc<dyn LlmProviderTrait>>,
}

impl LlmProviderRegistry {
    pub fn new(providers: BTreeMap<String, Arc<dyn LlmProviderTrait>>) -> Self {
        Self { providers }
    }
}

#[async_trait]
impl LlmResolver for LlmProviderRegistry {
    fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// The client for one call. There is no default provider and no fallback
    /// chain: an unknown name is an error naming what was declared.
    async fn resolve(
        &self,
        llm: &str,
        owner: &SessionOwner,
    ) -> Result<Arc<dyn LlmCallable>, String> {
        match self.providers.get(llm) {
            Some(p) => p.resolve(owner).await,
            None => Err(format!(
                "no engine-run llm block `{llm}` — declared: {}",
                match self.providers.is_empty() {
                    true => "none".to_string(),
                    false => self
                        .providers
                        .keys()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        .join(", "),
                }
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CallContext<'a> {
    pub session_id: &'a str,
    pub tenant_id: &'a str,
    pub agent_id: &'a str,
    pub call_id: &'a str,
    pub attempt: u32,
    pub owner: &'a SessionOwner,
    /// Parent chain, root-last. Empty for top-level sessions.
    pub ancestry: &'a [String],
    /// How this call lowers a deferred tool.
    pub defer_tools_strategy: DeferToolsStrategy,
}

impl CallContext<'_> {
    /// The root of this session's chain, or this session where it is the root.
    ///
    /// A router that pins a session to one endpoint forgets it once it goes
    /// quiet, and a parent makes no calls while its delegation runs. One name
    /// for the whole tree holds the endpoint for all of it.
    pub fn root_session_id(&self) -> &str {
        self.ancestry
            .last()
            .map(String::as_str)
            .unwrap_or(self.session_id)
    }
}

/// Trait for calling an LLM (single call or streaming).
#[async_trait]
pub trait LlmCallable: Send + Sync + 'static {
    async fn call(
        &self,
        request: &PromptRequest,
        ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError>;

    /// Streaming variant — sends deltas through `tx` while the call is
    /// in progress, then returns the final assembled response.
    /// Default implementation ignores the channel and delegates to `call()`.
    async fn call_streaming(
        &self,
        request: &PromptRequest,
        ctx: &CallContext<'_>,
        _tx: tokio::sync::mpsc::UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        self.call(request, ctx).await
    }
}

/// A model call that failed: the failure, plus whether this attempt is worth
/// repeating.
#[derive(Debug, Clone)]
pub struct LlmCallError {
    pub error: ErrorInfo,
    pub retryable: bool,
}

impl LlmCallError {
    pub fn new(code: ErrorCode, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            error: ErrorInfo::new(code, message),
            retryable,
        }
    }
}

/// A provider reply the engine could not read. Never retryable: the same bytes
/// parse the same way, and a provider that changed its shape needs a fix here,
/// not another call.
impl From<crate::json::JsonParseError> for LlmCallError {
    fn from(e: crate::json::JsonParseError) -> Self {
        Self {
            error: ErrorInfo::from(e),
            retryable: false,
        }
    }
}
