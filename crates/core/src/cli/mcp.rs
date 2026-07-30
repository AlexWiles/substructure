//! Authorizing the connections `substructure.toml` declares.
//!
//! Consent is always a human in a browser; what differs is where the credential
//! lands. A local environment authorizes over a loopback redirect and stores the
//! credential in its own database, beside the sessions that use it. A remote one
//! asks the server to start the flow, and the credential never touches this
//! machine.

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use axum::extract::Query;
use axum::response::Html;
use axum::routing::get;
use axum::Router;
use clap::Subcommand;
use serde::Deserialize;
use tokio::sync::{mpsc, Mutex, Notify};

use super::cloud::context::Context as CloudContext;
use super::cloud::project_config::{self, EnvConfig, LocalEnv, RemoteEnv};
use super::cloud::{AppScope, CloudGlobals};
use super::DEFAULT_TENANT;
use crate::api::v1::{McpAuthorizeRequest, McpAuthorizeResponse, McpConnection, McpGrantRequest};
use crate::connectors::oauth;
use crate::providers::sqlite::{SqliteDb, SqliteTokenStore};

/// How long the listener waits for the browser before giving up.
const CONSENT_TIMEOUT: Duration = Duration::from_secs(300);

/// How long a remote login waits for consent to land on the server.
const REMOTE_TIMEOUT: Duration = Duration::from_secs(300);
const POLL_INTERVAL: Duration = Duration::from_secs(3);

const CLIENT_NAME: &str = "Substructure";

#[derive(Subcommand)]
pub enum McpCommand {
    /// Authorize a connection, opening a browser for consent.
    Login {
        /// Connection id from `substructure.toml`. Optional when the file
        /// declares exactly one.
        id: Option<String>,
        /// Print the authorization URL instead of opening a browser.
        #[arg(long)]
        no_browser: bool,
        /// Don't grant the connection to the pinned app. Remote only.
        #[arg(long)]
        no_grant: bool,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Forget a connection's stored credential.
    Logout {
        id: Option<String>,
        #[command(flatten)]
        scope: AppScope,
    },
    /// Show declared connections and whether each one is authorized.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: AppScope,
    },
}

pub async fn run(command: McpCommand) -> Result<()> {
    match command {
        McpCommand::Login {
            id,
            no_browser,
            no_grant,
            scope,
        } => match environment(&scope.globals)? {
            EnvConfig::Local(cfg) => login_local(id, no_browser, cfg).await,
            EnvConfig::Remote(cfg) => login_remote(id, no_browser, no_grant, scope, cfg).await,
        },
        McpCommand::Logout { id, scope } => match environment(&scope.globals)? {
            EnvConfig::Local(cfg) => logout_local(id, cfg).await,
            EnvConfig::Remote(cfg) => logout_remote(id, scope, cfg).await,
        },
        McpCommand::List { scope } => match environment(&scope.globals)? {
            EnvConfig::Local(cfg) => list_local(cfg).await,
            EnvConfig::Remote(cfg) => list_remote(scope, cfg).await,
        },
    }
}

/// The environment these commands act on. An absent file is a configuration
/// problem, not an empty list: nothing here can succeed without one.
fn environment(globals: &CloudGlobals) -> Result<EnvConfig> {
    let found = project_config::resolve(globals.config.as_deref())?
        .context("no substructure.toml found; connections are declared under `[mcp.<id>]`")?;
    Ok(found.config)
}

/// Resolve the id to act on, allowing it to be omitted where there is no
/// ambiguity.
fn pick<T: Clone>(connections: &BTreeMap<String, T>, id: Option<String>) -> Result<(String, T)> {
    if connections.is_empty() {
        bail!("substructure.toml declares no connections under `[mcp.<id>]`");
    }
    let id = match id {
        Some(id) => id,
        None if connections.len() == 1 => connections.keys().next().expect("one").clone(),
        None => bail!(
            "name a connection: {}",
            connections.keys().cloned().collect::<Vec<_>>().join(", ")
        ),
    };
    let spec = connections
        .get(&id)
        .cloned()
        .with_context(|| format!("`{id}` is not declared in substructure.toml"))?;
    Ok((id, spec))
}

// ── Local: loopback consent, credential in the environment's database ────────

/// The environment's database, which is also its credential store. `subs run`
/// creates it the same way, so a login before the first run is not special.
fn open_db(cfg: &LocalEnv) -> Result<SqliteDb> {
    let path = cfg.db_path();
    if let Some(parent) = Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(SqliteDb::open(&path, Duration::from_secs(5))?)
}

/// The store, or nothing when no engine has ever run here — a database is not
/// worth creating just to report that nothing is authorized.
fn open_existing_db(cfg: &LocalEnv) -> Result<Option<SqliteDb>> {
    let path = cfg.db_path();
    if !Path::new(&path).exists() {
        return Ok(None);
    }
    Ok(Some(SqliteDb::open(&path, Duration::from_secs(5))?))
}

async fn login_local(id: Option<String>, no_browser: bool, cfg: LocalEnv) -> Result<()> {
    let (id, spec) = pick(&cfg.connections(), id)?;

    if let Some(auth) = &spec.auth {
        bail!(
            "`{id}` takes its credential from ${}; set that variable instead of logging in",
            auth.token_env
        );
    }

    let http = reqwest::Client::new();

    println!("Discovering {}...", spec.url);
    let discovered = oauth::discover(&http, &spec.url).await?;

    // Bound before registering: the redirect URI must name the port we will
    // actually be listening on, and dynamic registration records it.
    let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .context("binding a loopback port for the redirect")?;
    let port = listener.local_addr()?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    // No metadata document: this is a laptop, with no stable HTTPS address to
    // serve one from. A deployment that has one passes it here instead.
    let client =
        oauth::client_id(&http, &discovered.server, None, &redirect_uri, CLIENT_NAME).await?;

    let pending = oauth::authorize(&discovered, &client, &redirect_uri, &[])?;

    if let Some(scope) = &pending.scope {
        println!("Requesting: {scope}");
    }
    if no_browser || webbrowser::open(&pending.url).is_err() {
        println!("Open this URL to authorize:\n  {}", pending.url);
    } else {
        println!("Opened your browser to authorize.");
    }

    let returned = wait_for_redirect(listener).await?;
    if let Some(error) = returned.error {
        bail!(
            "authorization declined: {error}{}",
            returned
                .error_description
                .map(|d| format!(" ({d})"))
                .unwrap_or_default()
        );
    }
    // Compared before anything else is trusted: a redirect carrying another
    // flow's state is not this flow's.
    if returned.state.as_deref() != Some(pending.state.as_str()) {
        bail!("the redirect did not carry this login's state");
    }
    let code = returned.code.context("the redirect carried no code")?;

    let tokens = oauth::redeem(
        &http,
        &pending,
        &client,
        &discovered.server.token_endpoint,
        &code,
        returned.iss.as_deref(),
    )
    .await?;

    let store = SqliteTokenStore::new(open_db(&cfg)?)?;
    let refreshable = tokens.refreshable();
    // Keyed by URL, not by id: the credential is bound to the server, so two
    // ids naming one server share it.
    oauth::TokenStore::put(&store, DEFAULT_TENANT, &spec.url, tokens)
        .await
        .map_err(|e| anyhow::anyhow!("storing the credential: {e}"))?;

    println!("Authorized `{id}` in {}.", cfg.db_path());
    if !refreshable {
        println!(
            "Note: the server issued no refresh token, so this expires and needs `subs mcp login {id}` again."
        );
    }
    Ok(())
}

async fn logout_local(id: Option<String>, cfg: LocalEnv) -> Result<()> {
    let (id, spec) = pick(&cfg.connections(), id)?;
    let Some(db) = open_existing_db(&cfg)? else {
        println!("`{id}` was not authorized.");
        return Ok(());
    };
    let store = SqliteTokenStore::new(db)?;
    if store.delete(DEFAULT_TENANT, &spec.url).await? {
        println!("Forgot the credential for `{id}`.");
    } else {
        println!("`{id}` was not authorized.");
    }
    Ok(())
}

async fn list_local(cfg: LocalEnv) -> Result<()> {
    let connections = cfg.connections();
    if connections.is_empty() {
        bail!("substructure.toml declares no connections under `[mcp.<id>]`");
    }
    let store = match open_existing_db(&cfg)? {
        Some(db) => Some(SqliteTokenStore::new(db)?),
        None => None,
    };

    for (id, spec) in &connections {
        let status = match &spec.auth {
            Some(auth) if super::env_value(&auth.token_env).is_some() => {
                format!("${}", auth.token_env)
            }
            Some(auth) => format!("${} is unset", auth.token_env),
            None => match &store {
                Some(store) => {
                    match oauth::TokenStore::get(store, DEFAULT_TENANT, &spec.url).await {
                        Some(tokens) => describe(&tokens),
                        None => "not authorized".to_string(),
                    }
                }
                None => "not authorized".to_string(),
            },
        };
        println!("{id}\t{}\t{status}", spec.url);
    }
    Ok(())
}

fn describe(tokens: &oauth::Tokens) -> String {
    match tokens.expires_at {
        Some(at) if tokens.stale() && !tokens.refreshable() => {
            format!("expired {}", at.format("%Y-%m-%d %H:%M UTC"))
        }
        Some(at) => format!("authorized, expires {}", at.format("%Y-%m-%d %H:%M UTC")),
        None => "authorized".to_string(),
    }
}

// ── Remote: the server runs the flow, the credential stays there ─────────────

/// Consent may land in the dashboard instead of here, so this polls for the
/// outcome rather than owning it.
async fn login_remote(
    id: Option<String>,
    no_browser: bool,
    no_grant: bool,
    scope: AppScope,
    cfg: RemoteEnv,
) -> Result<()> {
    let (id, spec) = pick(&cfg.mcp, id)?;
    let ctx = CloudContext::load(&scope.globals)?;
    let org = ctx.require_org(scope.org.as_deref()).await?;

    let started: McpAuthorizeResponse = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/mcp/authorize"),
            &McpAuthorizeRequest {
                url: spec.url.clone(),
                connection_id: Some(id.clone()),
            },
        )
        .await?;

    if let Some(requested) = &started.scope {
        println!("Requesting: {requested}");
    }
    if no_browser || webbrowser::open(&started.authorize_url).is_err() {
        println!("Open this URL to authorize:\n  {}", started.authorize_url);
    } else {
        println!("Opened your browser to authorize.");
    }

    let connection = match wait_for_connection(&ctx, &org, &id).await? {
        Some(connection) => connection,
        None => {
            println!(
                "Still waiting on consent. Finish in the browser; the connection appears \
                 in `subs mcp list` once it lands."
            );
            return Ok(());
        }
    };
    println!("Authorized `{id}` for {org}.");

    if no_grant {
        return Ok(());
    }
    let app = ctx
        .pinned_app(scope.app.as_deref())
        .await
        .context("no app to grant the connection to. Pass --app <id>, or --no-grant.")?;
    ctx.client
        .post_json_discard(
            &format!(
                "/api/v1/orgs/{org}/mcp/connections/{}/grants",
                connection.id
            ),
            &McpGrantRequest {
                app_id: app.clone(),
            },
        )
        .await?;
    println!("Granted it to {app}.");
    Ok(())
}

/// Poll until the callback has stored the connection. Bounded: the person may
/// have closed the tab, and a CLI that never returns is worse than one that
/// says where to look.
async fn wait_for_connection(
    ctx: &CloudContext,
    org: &str,
    id: &str,
) -> Result<Option<McpConnection>> {
    let deadline = std::time::Instant::now() + REMOTE_TIMEOUT;
    loop {
        let connections: Vec<McpConnection> = ctx
            .client
            .get(&format!("/api/v1/orgs/{org}/mcp/connections"))
            .await?;
        if let Some(found) = connections
            .into_iter()
            .find(|c| c.connection_id == id && c.status == "active")
        {
            return Ok(Some(found));
        }
        if std::time::Instant::now() + POLL_INTERVAL > deadline {
            return Ok(None);
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn logout_remote(id: Option<String>, scope: AppScope, cfg: RemoteEnv) -> Result<()> {
    let (id, _) = pick(&cfg.mcp, id)?;
    let ctx = CloudContext::load(&scope.globals)?;
    let org = ctx.require_org(scope.org.as_deref()).await?;

    let connections: Vec<McpConnection> = ctx
        .client
        .get(&format!("/api/v1/orgs/{org}/mcp/connections"))
        .await?;
    let Some(found) = connections.into_iter().find(|c| c.connection_id == id) else {
        println!("`{id}` was not authorized.");
        return Ok(());
    };
    ctx.client
        .delete_discard(&format!("/api/v1/orgs/{org}/mcp/connections/{}", found.id))
        .await?;
    println!("Disconnected `{id}`.");
    Ok(())
}

async fn list_remote(scope: AppScope, cfg: RemoteEnv) -> Result<()> {
    if cfg.mcp.is_empty() {
        bail!("substructure.toml declares no connections under `[mcp.<id>]`");
    }
    let ctx = CloudContext::load(&scope.globals)?;
    let org = ctx.require_org(scope.org.as_deref()).await?;
    let connections: Vec<McpConnection> = ctx
        .client
        .get(&format!("/api/v1/orgs/{org}/mcp/connections"))
        .await?;

    for (id, spec) in &cfg.mcp {
        let status = match connections.iter().find(|c| &c.connection_id == id) {
            Some(found) if found.granted_apps.is_empty() => {
                format!("{}, granted to no app", found.status)
            }
            Some(found) => format!(
                "{}, granted to {}",
                found.status,
                found.granted_apps.join(", ")
            ),
            None => "not authorized".to_string(),
        };
        println!("{id}\t{}\t{status}", spec.url);
    }
    Ok(())
}

#[derive(Debug, Default, Deserialize)]
struct Returned {
    code: Option<String>,
    state: Option<String>,
    iss: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

/// Serve the redirect on the bound port until the browser arrives, then stop.
///
/// Graceful shutdown is what lets the page render: the signal fires once the
/// handler has produced a response, and the server drains it before closing.
async fn wait_for_redirect(listener: tokio::net::TcpListener) -> Result<Returned> {
    let (tx, mut rx) = mpsc::channel::<Returned>(1);
    let tx = Arc::new(Mutex::new(Some(tx)));
    let stop = Arc::new(Notify::new());

    let app = Router::new().route(
        "/callback",
        get(move |Query(returned): Query<Returned>| {
            let tx = tx.clone();
            async move {
                if let Some(tx) = tx.lock().await.take() {
                    let _ = tx.send(returned).await;
                }
                Html(PAGE)
            }
        }),
    );

    let serving = stop.clone();
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move { serving.notified().await })
            .await;
    });

    let returned = tokio::time::timeout(CONSENT_TIMEOUT, rx.recv()).await;
    stop.notify_one();
    let _ = server.await;

    match returned {
        Ok(Some(returned)) => Ok(returned),
        Ok(None) => bail!("the redirect listener closed before the browser returned"),
        Err(_) => bail!("timed out waiting for authorization"),
    }
}

const PAGE: &str = "<!doctype html><meta charset=utf-8><title>Substructure</title>\
<body style=\"font:16px system-ui;padding:3rem\">\
<p>Authorized. You can close this tab and return to your terminal.</p>";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::registry::{AuthSpec, ConnectionSpec};
    use crate::protocol::ConnectorProtocol;
    use std::path::PathBuf;

    fn spec(token_env: Option<&str>) -> ConnectionSpec {
        ConnectionSpec {
            url: "https://mcp.example.test/mcp".into(),
            protocol: ConnectorProtocol::Mcp,
            auth: token_env.map(|v| AuthSpec {
                header: None,
                token_env: v.into(),
            }),
            prefix_tools: true,
        }
    }

    fn tmpdir() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("subs-mcp-add-test-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_id_may_be_omitted_only_when_there_is_no_ambiguity() {
        let one = BTreeMap::from([("linear".to_string(), spec(None))]);
        assert_eq!(pick(&one, None).unwrap().0, "linear");

        let two = BTreeMap::from([
            ("linear".to_string(), spec(None)),
            ("sentry".to_string(), spec(None)),
        ]);
        let err = pick(&two, None).unwrap_err().to_string();
        assert!(err.contains("linear") && err.contains("sentry"), "{err}");

        let err = pick(&one, Some("nope".into())).unwrap_err().to_string();
        assert!(err.contains("not declared"), "{err}");

        let none: BTreeMap<String, ConnectionSpec> = BTreeMap::new();
        let err = pick(&none, None).unwrap_err().to_string();
        assert!(err.contains("declares no connections"), "{err}");
    }

    #[tokio::test]
    async fn a_token_backed_connection_is_not_logged_in_to() {
        let cfg = project_config::load_explicit(&{
            let path = tmpdir().join(project_config::FILENAME);
            std::fs::write(
                &path,
                "target = \"local\"\n[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n\
                 auth = { token_env = \"SENTRY_TOKEN\" }\n",
            )
            .unwrap();
            path
        })
        .unwrap()
        .into_local("x")
        .unwrap();

        let err = login_local(Some("sentry".into()), true, cfg)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("SENTRY_TOKEN"), "{err}");
    }

    #[tokio::test]
    async fn a_credential_lands_in_the_environments_database() {
        let dir = tmpdir();
        let db_path = dir.join("engine.db");
        let cfg: LocalEnv = toml::from_str(&format!(
            "target = \"local\"\ndb = {:?}\n[mcp.linear]\nurl = \"https://mcp.linear.app/mcp\"\n",
            db_path.to_str().unwrap()
        ))
        .unwrap();

        let store = SqliteTokenStore::new(open_db(&cfg).unwrap()).unwrap();
        let url = "https://mcp.linear.app/mcp";
        oauth::TokenStore::put(&store, DEFAULT_TENANT, url, tokens())
            .await
            .unwrap();

        // What `list` reads, and what `logout` removes.
        assert!(oauth::TokenStore::get(&store, DEFAULT_TENANT, url)
            .await
            .is_some());
        logout_local(Some("linear".into()), cfg.clone())
            .await
            .unwrap();
        assert!(oauth::TokenStore::get(&store, DEFAULT_TENANT, url)
            .await
            .is_none());

        // A second logout is a statement, not a failure.
        logout_local(Some("linear".into()), cfg).await.unwrap();
    }

    #[tokio::test]
    async fn listing_does_not_create_a_database() {
        let dir = tmpdir();
        let db_path = dir.join("absent.db");
        let cfg: LocalEnv = toml::from_str(&format!(
            "target = \"local\"\ndb = {:?}\n[mcp.linear]\nurl = \"https://mcp.linear.app/mcp\"\n",
            db_path.to_str().unwrap()
        ))
        .unwrap();

        list_local(cfg).await.unwrap();
        assert!(
            !db_path.exists(),
            "reporting a status must not create state"
        );
    }

    fn tokens() -> oauth::Tokens {
        oauth::Tokens {
            access_token: "a".into(),
            refresh_token: Some("r".into()),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            scope: None,
            issuer: "https://mcp.linear.app".into(),
            token_endpoint: "https://mcp.linear.app/token".into(),
            resource: "https://mcp.linear.app/mcp".into(),
            client: oauth::ClientId::Metadata {
                url: "https://app.test/client.json".into(),
            },
        }
    }

    #[test]
    fn status_distinguishes_expired_from_live() {
        let mut tokens = tokens();
        assert!(describe(&tokens).starts_with("authorized, expires"));

        tokens.refresh_token = None;
        tokens.expires_at = Some(chrono::Utc::now() - chrono::Duration::hours(1));
        assert!(describe(&tokens).starts_with("expired"));

        // Refreshable and stale is not expired: the resolver renews it.
        tokens.refresh_token = Some("r".into());
        assert!(describe(&tokens).starts_with("authorized, expires"));

        tokens.expires_at = None;
        assert_eq!(describe(&tokens), "authorized");
    }
}
