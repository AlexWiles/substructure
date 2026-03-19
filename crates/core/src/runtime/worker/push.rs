use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::WorkerDecisionRequest;

#[async_trait]
pub trait PushTransport: Send + Sync {
    async fn push(&self, decision: &WorkerDecisionRequest) -> Result<PushResponse, PushError>;
}

pub struct PushResponse {
    pub actions: Vec<crate::runtime::session::decision::WorkerAction>,
    pub state: Vec<u8>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushRegistrationRecord {
    pub tenant_id: String,
    pub agent_id: String,
    pub transport_type: String,
    pub config: serde_json::Value,
}

#[async_trait]
pub trait PushRegistrationStore: Send + Sync {
    async fn save(&self, record: &PushRegistrationRecord) -> Result<(), String>;
    async fn remove(&self, tenant_id: &str, agent_id: &str) -> Result<(), String>;
    async fn get(
        &self,
        tenant_id: &str,
        agent_id: &str,
    ) -> Result<Option<PushRegistrationRecord>, String>;
    async fn list_tenants(&self) -> Result<HashMap<String, Vec<String>>, String>;
}

pub type TransportConstructor =
    Box<dyn Fn(serde_json::Value) -> Result<Arc<dyn PushTransport>, String> + Send + Sync>;

pub struct TransportRegistry {
    constructors: HashMap<String, TransportConstructor>,
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

pub struct PushRegistry {
    store: Arc<dyn PushRegistrationStore>,
    transports: TransportRegistry,
}

impl PushRegistry {
    pub fn new(store: Arc<dyn PushRegistrationStore>, transports: TransportRegistry) -> Self {
        Self { store, transports }
    }

    pub async fn register(&self, record: PushRegistrationRecord) -> Result<(), String> {
        self.transports
            .create(&record.transport_type, record.config.clone())?;
        self.store.save(&record).await
    }

    pub async fn unregister(&self, tenant_id: &str, agent_id: &str) -> Result<(), String> {
        self.store.remove(tenant_id, agent_id).await
    }

    pub async fn lookup(&self, tenant_id: &str, agent_id: &str) -> Option<Arc<dyn PushTransport>> {
        let record = self.store.get(tenant_id, agent_id).await.ok()??;
        self.transports
            .create(&record.transport_type, record.config)
            .ok()
    }

    pub async fn tenants(&self) -> HashMap<String, Vec<String>> {
        self.store.list_tenants().await.unwrap_or_default()
    }
}
