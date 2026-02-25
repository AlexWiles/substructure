use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::client::LlmClient;
use crate::domain::config::LlmClientConfig;
use crate::domain::event::SessionAuth;

pub type LlmClientFactory = Box<
    dyn Fn(&serde_json::Map<String, serde_json::Value>) -> Result<Arc<dyn LlmClient>, String>
        + Send
        + Sync,
>;

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("unknown LLM client: {0}")]
    UnknownClient(String),
}

#[async_trait]
pub trait LlmClientProvider: Send + Sync + 'static {
    async fn resolve(
        &self,
        client_id: &str,
        auth: &SessionAuth,
    ) -> Result<Arc<dyn LlmClient>, ProviderError>;
}

pub struct StaticLlmClientProvider {
    clients: HashMap<String, Arc<dyn LlmClient>>,
}

impl StaticLlmClientProvider {
    pub fn new(clients: HashMap<String, Arc<dyn LlmClient>>) -> Self {
        Self { clients }
    }

    pub fn from_config(
        configs: &HashMap<String, LlmClientConfig>,
        factories: &HashMap<String, LlmClientFactory>,
    ) -> Result<Self, String> {
        let mut clients = HashMap::new();
        for (id, config) in configs {
            let factory = factories
                .get(config.client_type.as_str())
                .ok_or_else(|| format!("unknown LLM client type: {}", config.client_type))?;
            clients.insert(id.clone(), factory(&config.settings)?);
        }
        Ok(Self { clients })
    }
}

#[async_trait]
impl LlmClientProvider for StaticLlmClientProvider {
    async fn resolve(
        &self,
        client_id: &str,
        _auth: &SessionAuth,
    ) -> Result<Arc<dyn LlmClient>, ProviderError> {
        self.clients
            .get(client_id)
            .cloned()
            .ok_or_else(|| ProviderError::UnknownClient(client_id.to_string()))
    }
}
