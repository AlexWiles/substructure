use std::sync::Arc;

use async_trait::async_trait;

use crate::protocol::DecisionResponse;
use crate::runtime::llm::TokenDeltaTransport;

use super::WorkerDecisionRequest;

#[async_trait]
pub trait PushTransport: Send + Sync {
    async fn push(
        &self,
        decision: &WorkerDecisionRequest,
        token_delta_transport: std::sync::Arc<dyn TokenDeltaTransport>,
    ) -> Result<DecisionResponse, PushError>;
}

#[derive(Debug)]
pub struct PushError {
    pub message: String,
    pub retryable: bool,
}

impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub type TransportConstructor =
    Box<dyn Fn(serde_json::Value) -> Result<Arc<dyn PushTransport>, String> + Send + Sync>;

pub struct TransportRegistry {
    constructors: std::collections::HashMap<String, TransportConstructor>,
}

impl TransportRegistry {
    pub fn new(entries: Vec<(&str, TransportConstructor)>) -> Self {
        Self {
            constructors: entries
                .into_iter()
                .map(|(name, c)| (name.to_string(), c))
                .collect(),
        }
    }

    pub fn create(
        &self,
        transport_type: &str,
        config: serde_json::Value,
    ) -> Result<Arc<dyn PushTransport>, String> {
        let constructor = self
            .constructors
            .get(transport_type)
            .ok_or_else(|| format!("unknown transport type: {transport_type}"))?;
        constructor(config)
    }
}
