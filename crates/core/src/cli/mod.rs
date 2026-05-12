pub mod auth;
pub mod env;

use crate::transport::push::PushAdapter;
use crate::worker::push::PushRegistrationRecord;

pub(crate) const DEFAULT_TENANT: &str = "default";

pub async fn register_startup_worker(
    adapter: &PushAdapter,
    url: &str,
    signing_secret: Option<String>,
) -> anyhow::Result<()> {
    let secret = signing_secret.unwrap_or_else(|| hex::encode(rand::random::<[u8; 32]>()));
    adapter
        .register(PushRegistrationRecord {
            tenant_id: DEFAULT_TENANT.into(),
            transport_type: "http".into(),
            config: serde_json::json!({
                "endpoint_url": url,
                "signing_secret": secret,
            }),
        })
        .await
        .map_err(|e| anyhow::anyhow!("failed to register startup worker: {e}"))?;
    tracing::info!(url, "startup worker registered (signing enabled)");
    Ok(())
}
