use std::sync::Arc;

use async_trait::async_trait;

use crate::domain::event::{McpServerConfig, McpTransportConfig, SessionAuth};
use super::client::{McpClient, McpError};
use super::stdio::StdioMcpClient;

#[async_trait]
pub trait McpClientProvider: Send + Sync + 'static {
    async fn start_server(
        &self,
        config: &McpServerConfig,
        auth: &SessionAuth,
    ) -> Result<Arc<dyn McpClient>, McpError>;
}

pub struct StaticMcpClientProvider;

#[async_trait]
impl McpClientProvider for StaticMcpClientProvider {
    async fn start_server(
        &self,
        config: &McpServerConfig,
        _auth: &SessionAuth,
    ) -> Result<Arc<dyn McpClient>, McpError> {
        match &config.transport {
            McpTransportConfig::Stdio { command, args } => {
                let client = StdioMcpClient::new(command, args).await?;
                Ok(Arc::new(client))
            }
        }
    }
}
