//! Where a person gives consent for a connection this deployment dials.
//!
//! A deployment decides this, not a channel: the cloud mints a link, an engine
//! hosting the flow mints its own, and a laptop can only name a command. The
//! channel renders whatever comes back, so a second channel adds no cases.

use std::sync::Arc;

use crate::connectors::Requester;
use crate::runtime::session::interrupts::auth::Authorize;

/// How one person authorizes one connection.
pub enum WayIn {
    Link {
        url: String,
        label: String,
    },
    /// Nothing a browser can reach from here.
    Command(String),
}

#[async_trait::async_trait]
pub trait Consent: Send + Sync {
    /// `None` where this person has no way in at all.
    async fn way_in(&self, tenant_id: &str, authorize: &Authorize) -> Option<WayIn>;
}

/// A page that works out who is asking from their session, so the link carries
/// no requester.
pub struct DashboardConsent(pub String);

#[async_trait::async_trait]
impl Consent for DashboardConsent {
    async fn way_in(&self, _tenant_id: &str, _authorize: &Authorize) -> Option<WayIn> {
        Some(WayIn::Link {
            url: self.0.clone(),
            label: "Authorize it in the dashboard".to_string(),
        })
    }
}

/// This engine hosts the flow, minting one link per person per prompt.
pub struct EngineConsent(pub Arc<crate::transport::mcp_auth::AuthorizeLinks>);

#[async_trait::async_trait]
impl Consent for EngineConsent {
    async fn way_in(&self, tenant_id: &str, authorize: &Authorize) -> Option<WayIn> {
        let minted = self
            .0
            .mint_for(
                tenant_id,
                &authorize.connection,
                &authorize.requester,
                chrono::Utc::now(),
            )
            .await;
        match minted {
            // A requester the call would refuse gets no link.
            Ok(None) => Some(cli(&authorize.connection)),
            Ok(Some(url)) => Some(WayIn::Link {
                url,
                label: format!("Authorize {}", authorize.connection),
            }),
            Err(e) => {
                tracing::warn!(error = %e, connection = %authorize.connection, "minting an authorize link failed");
                Some(cli(&authorize.connection))
            }
        }
    }
}

/// Nobody else can open a browser flow that lands on this machine.
#[derive(Default)]
pub struct CliConsent;

#[async_trait::async_trait]
impl Consent for CliConsent {
    async fn way_in(&self, _tenant_id: &str, authorize: &Authorize) -> Option<WayIn> {
        Some(cli(&authorize.connection))
    }
}

fn cli(connection: &str) -> WayIn {
    WayIn::Command(format!(
        "Run `subs mcp login {connection}` to authorize it."
    ))
}

/// Kept out of `Requester`'s own file: this is what a way in is asked about,
/// not what a requester is.
pub fn refuses(requester: &Requester) -> bool {
    requester.subject.is_none()
}
