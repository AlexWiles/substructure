use async_trait::async_trait;

use crate::json::JsonParseError;
use crate::protocol::{DecisionResponse, ErrorCode, ErrorInfo};
use crate::runtime::llm::TokenDeltaTransport;

use super::WorkerDecisionRequest;

#[async_trait]
pub trait Decider: Send + Sync {
    async fn push(
        &self,
        decision: &WorkerDecisionRequest,
        token_delta_transport: std::sync::Arc<dyn TokenDeltaTransport>,
    ) -> Result<DecisionResponse, PushError>;
}

/// Why a decision could not be delivered or its answer understood.
///
/// Shaped after a Stripe API error: `code` classifies, `message` is one
/// engine-authored sentence, `param` names the one input to go and fix, and
/// `detail` carries small structured particulars. Nothing here is a raw body —
/// an unbounded document from a worker belongs in the log, not in an event that
/// is persisted and streamed to whoever is watching the session.
#[derive(Debug)]
pub struct PushError {
    pub error: ErrorInfo,
    pub retryable: bool,
}

impl PushError {
    /// A failure a second attempt could plausibly survive: a timeout, a 5xx,
    /// a stream cut short.
    pub fn retryable(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            error: ErrorInfo::new(code, message),
            retryable: true,
        }
    }

    /// A failure that will repeat verbatim: a malformed answer, a 4xx, a
    /// decision nothing can route.
    pub fn fatal(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            error: ErrorInfo::new(code, message),
            retryable: false,
        }
    }

    /// Name the one input to go and fix.
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.error = self.error.with_param(param);
        self
    }

    pub fn with_detail(mut self, detail: serde_json::Value) -> Self {
        self.error = self.error.with_detail(detail);
        self
    }

    pub fn retryable_if(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }
}

/// A document the worker sent that the engine could not read. Fatal by
/// construction: the same bytes parse the same way on every attempt.
impl From<JsonParseError> for PushError {
    fn from(e: JsonParseError) -> Self {
        PushError {
            error: ErrorInfo::from(e),
            retryable: false,
        }
    }
}

impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}
