//! The browser half of an MCP authorization.
//!
//! [`AuthorizeLinks::mint`] takes a connection and a subject and returns a
//! URL. It reads no session and no interrupt, so any caller can use it.
//!
//! The link is a random token naming a stored row. The database holds only
//! its hash.
//!
//! Nothing resumes the waiting session. The person asks the agent again and
//! the retried fetch finds the credential.
//!
//! Both routes are unauthenticated: the link is its own proof, and the
//! callback names a pending flow's OAuth state.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::Router;
use base64::Engine as _;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::connectors::credential::{Credential, CredentialStore};
use crate::connectors::registry::{ConnectionSpec, Connections};
use crate::connectors::{oauth, Requester, Slot};
use crate::providers::sqlite::{Flow, Link, SqliteAuthFlows, SqliteCredentialStore};
use crate::runtime::secret::{SecretRef, SecretStore};

const CLIENT_NAME: &str = "Substructure";

/// How long a link stays good.
pub const LINK_TTL: chrono::Duration = chrono::Duration::hours(4);

/// How often the sweeper looks.
const SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

/// Handles are stored hashed: the database holds no value a browser could
/// present.
fn hash(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}

/// Authorize links for the connections this engine dials.
pub struct AuthorizeLinks {
    base_url: String,
    flows: Arc<SqliteAuthFlows>,
    connections: std::collections::BTreeMap<String, ConnectionSpec>,
}

impl AuthorizeLinks {
    pub fn new(
        base_url: &str,
        flows: Arc<SqliteAuthFlows>,
        connections: std::collections::BTreeMap<String, ConnectionSpec>,
    ) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            flows,
            connections,
        }
    }

    pub fn spec(&self, connection_id: &str) -> Option<&ConnectionSpec> {
        self.connections.get(connection_id)
    }

    /// Where every flow's redirect lands.
    pub fn callback_url(&self) -> String {
        format!("{}/mcp/callback", self.base_url)
    }

    /// A link with the slot resolved by the same decision the call makes.
    /// `None` when that decision refuses.
    pub async fn mint_for(
        &self,
        tenant_id: &str,
        connection_id: &str,
        requester: &Requester,
        now: DateTime<Utc>,
    ) -> Result<Option<String>, String> {
        let Some(spec) = self.connections.get(connection_id) else {
            return Ok(None);
        };
        let Ok(subject) = Connections::slot_for(connection_id, spec, requester) else {
            return Ok(None);
        };
        self.mint(tenant_id, connection_id, subject, now).await
    }

    /// A link that fills `subject`'s slot on `connection_id`. `None` for a
    /// connection this engine does not declare.
    pub async fn mint(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: Slot,
        now: DateTime<Utc>,
    ) -> Result<Option<String>, String> {
        if !self.connections.contains_key(connection_id) {
            return Ok(None);
        }
        let mut raw = [0u8; 32];
        rand::Rng::fill(&mut rand::rng(), &mut raw);
        let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw);

        self.flows
            .mint(
                Link {
                    link_hash: hash(&token),
                    tenant_id: tenant_id.to_string(),
                    connection_id: connection_id.to_string(),
                    subject,
                },
                now,
                now + LINK_TTL,
            )
            .await
            .map_err(|e| e.to_string())?;
        Ok(Some(format!("{}/mcp/authorize/{token}", self.base_url)))
    }
}

/// The OAuth half a flow holds between the click and the redirect.
#[derive(Serialize, Deserialize)]
struct PendingFlow {
    pending: oauth::Pending,
    client: oauth::ClientId,
    token_endpoint: String,
}

pub struct McpAuthDeps {
    pub links: Arc<AuthorizeLinks>,
    pub flows: Arc<SqliteAuthFlows>,
    pub secrets: Arc<dyn SecretStore>,
    pub credentials: Arc<SqliteCredentialStore>,
    pub http: reqwest::Client,
}

pub fn router(deps: Arc<McpAuthDeps>) -> Router {
    Router::new()
        .route("/mcp/authorize/{token}", get(authorize))
        .route("/mcp/callback", get(callback))
        .with_state(deps)
}

fn page(text: &str) -> Response {
    Html(format!(
        "<!doctype html><meta charset=utf-8><title>Substructure</title>\
         <body style=\"font:16px system-ui;padding:3rem\"><p>{text}</p>"
    ))
    .into_response()
}

/// A link that names no live row. Every reason reads the same, so a probe
/// cannot tell them apart.
const SPENT: &str = "This link is finished. Ask the agent again.";

/// Look the link up and send the browser to the provider.
async fn authorize(State(deps): State<Arc<McpAuthDeps>>, Path(token): Path<String>) -> Response {
    match begin(&deps, &token).await {
        Ok(url) => Redirect::temporary(&url).into_response(),
        Err(why) => page(&why),
    }
}

async fn begin(deps: &McpAuthDeps, token: &str) -> Result<String, String> {
    let now = Utc::now();
    let link = deps
        .flows
        .resolve(&hash(token), now)
        .await
        .map_err(|e| e.to_string())?
        .ok_or(SPENT)?;
    let spec = deps.links.spec(&link.connection_id).ok_or(SPENT)?;

    let discovered = oauth::discover(&deps.http, &spec.url)
        .await
        .map_err(|e| format!("discovery failed for `{}`: {e}", link.connection_id))?;
    let redirect_uri = deps.links.callback_url();
    let client = oauth::client_id(
        &deps.http,
        &discovered.server,
        None,
        &redirect_uri,
        CLIENT_NAME,
    )
    .await
    .map_err(|e| e.to_string())?;
    let pending =
        oauth::authorize(&discovered, &client, &redirect_uri, &[]).map_err(|e| e.to_string())?;

    let held = PendingFlow {
        pending: pending.clone(),
        client,
        token_endpoint: discovered.server.token_endpoint.clone(),
    };
    let secret_ref = SecretRef::mint();
    deps.secrets
        .put(
            &link.tenant_id,
            &secret_ref,
            &serde_json::to_vec(&held).map_err(|e| e.to_string())?,
        )
        .await
        .map_err(|e| e.to_string())?;
    let displaced = deps
        .flows
        .attach(&link.link_hash, &hash(&pending.state), &secret_ref, now)
        .await
        .map_err(|e| e.to_string())?;
    if let Some(stale) = displaced {
        let _ = deps.secrets.delete(&link.tenant_id, &stale).await;
    }
    Ok(pending.url)
}

#[derive(Debug, Default, Deserialize)]
struct Returned {
    code: Option<String>,
    state: Option<String>,
    iss: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

/// The provider's redirect. Claim first, so a replayed redirect stops before
/// anything is exchanged.
async fn callback(State(deps): State<Arc<McpAuthDeps>>, Query(ret): Query<Returned>) -> Response {
    let now = Utc::now();
    let Some(state) = &ret.state else {
        return page("the redirect carried no state");
    };
    let flow = match deps.flows.claim(&hash(state), now).await {
        Ok(Some(flow)) => flow,
        Ok(None) => return page(SPENT),
        Err(e) => {
            tracing::error!(error = %e, "auth flow claim failed");
            return page("Something went wrong on this side. Try again.");
        }
    };

    let outcome = redeem(&deps, &flow, &ret).await;
    // A link is spent by authorizing, not by clicking. A failed handshake
    // leaves it usable.
    let settled = match outcome.is_ok() {
        true => deps.flows.complete(&flow).await,
        false => deps.flows.release(&flow).await,
    };
    if let Err(e) = settled {
        tracing::error!(error = %e, connection = %flow.connection_id, "auth flow settle failed");
    }
    let _ = deps
        .secrets
        .delete(&flow.tenant_id, &flow.pending_secret)
        .await;

    match outcome {
        Ok(()) => page(
            "Authorized. Close this tab, go back to the conversation, and ask the agent to \
             try again.",
        ),
        Err(why) => page(&format!("Authorization did not complete: {why}")),
    }
}

async fn redeem(deps: &McpAuthDeps, flow: &Flow, ret: &Returned) -> Result<(), String> {
    if let Some(error) = &ret.error {
        let detail = ret
            .error_description
            .as_ref()
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        return Err(format!("the provider declined: {error}{detail}"));
    }
    let code = ret.code.as_ref().ok_or("the redirect carried no code")?;
    let held: PendingFlow = deps
        .secrets
        .get(&flow.tenant_id, &flow.pending_secret)
        .await
        .map_err(|e| e.to_string())?
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .ok_or("this flow's state is gone; ask the agent again")?;

    let tokens = oauth::redeem(
        &deps.http,
        &held.pending,
        &held.client,
        &held.token_endpoint,
        code,
        ret.iss.as_deref(),
    )
    .await
    .map_err(|e| e.to_string())?;

    deps.credentials
        .put(
            &flow.tenant_id,
            &flow.connection_id,
            &flow.subject,
            Credential::Oauth(Box::new(tokens)),
        )
        .await
        .map_err(|e| format!("storing the credential: {e}"))
}

/// Delete expired links and dead claims, and their pending secrets.
pub fn spawn_sweeper(
    deps: Arc<McpAuthDeps>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(SWEEP_INTERVAL) => {}
                _ = cancel.cancelled() => break,
            }
            match deps.flows.sweep(Utc::now()).await {
                Ok(orphaned) => {
                    for (tenant_id, secret_ref) in &orphaned {
                        // A failed delete leaves an unreferenced secret.
                        // Nothing can read it back.
                        let _ = deps.secrets.delete(tenant_id, secret_ref).await;
                    }
                }
                Err(e) => tracing::warn!(error = %e, "auth flow sweep failed"),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::registry::CredentialScope;
    use crate::protocol::{ConnectorProtocol, Visibility};
    use crate::protocol::{Issuer, Subject};
    use crate::providers::sqlite::SqliteDb;

    fn spec(credential: Option<CredentialScope>) -> ConnectionSpec {
        ConnectionSpec {
            url: "https://mcp.gmail.test/mcp".into(),
            protocol: ConnectorProtocol::Mcp,
            auth: None,
            header: None,
            credential,
            prefix_tools: true,
        }
    }

    fn links(
        credential: Option<CredentialScope>,
    ) -> (AuthorizeLinks, Arc<SqliteAuthFlows>, std::path::PathBuf) {
        let path =
            std::env::temp_dir().join(format!("core-authorize-links-{}.db", uuid::Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), std::time::Duration::from_secs(5)).unwrap();
        let flows = Arc::new(SqliteAuthFlows::new(db));
        let links = AuthorizeLinks::new(
            "https://agent.test/",
            flows.clone(),
            [("gmail".to_string(), spec(credential))].into(),
        );
        (links, flows, path)
    }

    fn cleanup(path: &std::path::Path) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    fn token_of(url: &str) -> &str {
        url.rsplit('/').next().unwrap()
    }

    /// The URL carries no identifiers. The row holds them.
    #[tokio::test]
    async fn a_link_discloses_nothing_and_resolves_to_its_slot() {
        let (links, flows, path) = links(Some(CredentialScope::User));
        let now = Utc::now();
        let url = links
            .mint_for(
                "default",
                "gmail",
                &Requester::new(Subject::new(Issuer::slack(), "T1:U1"), Visibility::Private),
                now,
            )
            .await
            .unwrap()
            .expect("a private person may authorize");

        let token = token_of(&url);
        assert!(!url.contains("default"), "no tenant in the URL: {url}");
        assert!(!url.contains("slack:U1"), "no subject in the URL: {url}");
        assert!(!token.contains("gmail"), "no connection in the token");

        let link = flows.resolve(&hash(token), now).await.unwrap().unwrap();
        assert_eq!(
            link.subject,
            Slot::Of(Subject::new(Issuer::slack(), "T1:U1".to_string()))
        );
        assert_eq!(link.connection_id, "gmail");
        assert_eq!(link.tenant_id, "default");
        cleanup(&path);
    }

    /// Whoever could not read the credential gets no link to store one.
    #[tokio::test]
    async fn a_requester_the_call_would_refuse_gets_no_link() {
        let (links, _, path) = links(Some(CredentialScope::User));
        for refused in [
            Requester::machine(),
            Requester::new(Subject::new(Issuer::slack(), "T1:U1"), Visibility::Shared),
        ] {
            assert!(links
                .mint_for("default", "gmail", &refused, Utc::now())
                .await
                .unwrap()
                .is_none());
        }
        cleanup(&path);
    }

    /// A shared connection has one slot and anybody may fill it.
    #[tokio::test]
    async fn a_shared_connection_answers_for_a_machine() {
        let (links, flows, path) = links(None);
        let now = Utc::now();
        let url = links
            .mint_for("default", "gmail", &Requester::machine(), now)
            .await
            .unwrap()
            .expect("a shared connection has a slot for anybody");
        let link = flows
            .resolve(&hash(token_of(&url)), now)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(link.subject, Slot::Shared);
        cleanup(&path);
    }

    #[tokio::test]
    async fn an_undeclared_connection_mints_nothing() {
        let (links, _, path) = links(None);
        assert!(links
            .mint_for("default", "nope", &Requester::machine(), Utc::now())
            .await
            .unwrap()
            .is_none());
        cleanup(&path);
    }

    /// Two mints of one slot are two links.
    #[tokio::test]
    async fn every_mint_is_its_own_token() {
        let (links, _, path) = links(None);
        let now = Utc::now();
        let one = links
            .mint_for("default", "gmail", &Requester::machine(), now)
            .await
            .unwrap()
            .unwrap();
        let two = links
            .mint_for("default", "gmail", &Requester::machine(), now)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(one, two);
        cleanup(&path);
    }
}
