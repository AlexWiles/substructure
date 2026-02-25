use std::sync::Arc;

use async_trait::async_trait;

use crate::domain::agent::AgentConfig;
use crate::domain::event::SessionAuth;
use crate::domain::session::{DefaultStrategy, Strategy};

#[derive(Debug, thiserror::Error)]
pub enum StrategyProviderError {
    #[error("unknown strategy: {0}")]
    UnknownStrategy(String),
    #[error("invalid strategy config: {0}")]
    InvalidConfig(String),
}

#[async_trait]
pub trait StrategyProvider: Send + Sync + 'static {
    async fn resolve(
        &self,
        agent: &AgentConfig,
        auth: &SessionAuth,
    ) -> Result<Arc<dyn Strategy>, StrategyProviderError>;
}

pub struct StaticStrategyProvider;

#[async_trait]
impl StrategyProvider for StaticStrategyProvider {
    async fn resolve(
        &self,
        agent: &AgentConfig,
        _auth: &SessionAuth,
    ) -> Result<Arc<dyn Strategy>, StrategyProviderError> {
        match agent.strategy.kind.as_str() {
            "default" => {
                let strategy = DefaultStrategy::init(&agent.strategy.params)
                    .map_err(StrategyProviderError::InvalidConfig)?;
                Ok(Arc::from(strategy))
            }
            other => Err(StrategyProviderError::UnknownStrategy(other.to_string())),
        }
    }
}
