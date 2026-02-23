use async_trait::async_trait;
use serde_json::Value;

use crate::domain::agent::AgentConfig;
use crate::domain::event::SessionAuth;
use crate::domain::session::{Strategy, StrategyKind, ReactStrategy};

#[derive(Debug, thiserror::Error)]
pub enum StrategyProviderError {
    #[error("unknown strategy: {0:?}")]
    UnknownStrategy(StrategyKind),
}

#[async_trait]
pub trait StrategyProvider: Send + Sync + 'static {
    async fn resolve(
        &self,
        kind: &StrategyKind,
        agent: &AgentConfig,
        auth: &SessionAuth,
    ) -> Result<(Box<dyn Strategy>, Value), StrategyProviderError>;
}

pub struct StaticStrategyProvider;

#[async_trait]
impl StrategyProvider for StaticStrategyProvider {
    async fn resolve(
        &self,
        kind: &StrategyKind,
        _agent: &AgentConfig,
        _auth: &SessionAuth,
    ) -> Result<(Box<dyn Strategy>, Value), StrategyProviderError> {
        match kind {
            StrategyKind::React => Ok((Box::new(ReactStrategy::new()), Value::Null)),
        }
    }
}
