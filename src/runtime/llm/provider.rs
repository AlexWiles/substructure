use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::{LlmCallable, LlmProviderTrait};
use crate::runtime::config::LlmClientConfig;
use crate::runtime::config::ClientIdentity;

pub type LlmClientFactory = Box<
    dyn Fn(&serde_json::Map<String, serde_json::Value>) -> Result<Arc<dyn LlmCallable>, String>
        + Send
        + Sync,
>;

pub struct StaticLlmClientProvider {
    clients: HashMap<String, Arc<dyn LlmCallable>>,
}

impl StaticLlmClientProvider {
    pub fn new(clients: HashMap<String, Arc<dyn LlmCallable>>) -> Self {
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
impl LlmProviderTrait for StaticLlmClientProvider {
    async fn resolve(
        &self,
        client_id: &str,
        _auth: &ClientIdentity,
    ) -> Result<Arc<dyn LlmCallable>, String> {
        self.clients
            .get(client_id)
            .cloned()
            .ok_or_else(|| format!("unknown LLM client: {client_id}"))
    }
}
