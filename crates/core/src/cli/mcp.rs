//! Authorizing the connections `substructure.toml` declares.
//!
//! Consent is always a human in a browser; what differs is where the credential
//! lands, and the file says which. A file naming a `[remote]` asks that
//! server to run the flow, and the credential never touches this machine.
//! Otherwise the engine here is the one that will dial the connection, so
//! consent comes back over a loopback redirect into that engine's database,
//! beside the sessions that use it.

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
use super::cloud::project_config::{self, ProjectConfig};
use super::cloud::{CloudGlobals, ProjectScope};
use super::DEFAULT_TENANT;
use crate::api::v1::{McpAuthorizeResponse, McpConnection, McpDeclareRequest};
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
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Show declared connections and whether each one is authorized.
    #[command(name = "list", visible_alias = "ls")]
    List {
        #[command(flatten)]
        scope: ProjectScope,
    },
}

pub async fn run(command: McpCommand) -> Result<()> {
    match command {
        McpCommand::Login {
            id,
            no_browser,
            scope,
        } => match environment(&scope.globals)? {
            cfg if cfg.remote.is_none() => login_local(id, no_browser, cfg).await,
            cfg => login_remote(id, no_browser, scope, cfg).await,
        },
        McpCommand::List { scope } => match environment(&scope.globals)? {
            cfg if cfg.remote.is_none() => list_local(cfg).await,
            cfg => list_remote(scope, cfg).await,
        },
    }
}

/// The environment these commands act on. An absent file is a configuration
/// problem, not an empty list: nothing here can succeed without one.
fn environment(globals: &CloudGlobals) -> Result<ProjectConfig> {
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
fn open_db(cfg: &ProjectConfig) -> Result<SqliteDb> {
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
fn open_existing_db(cfg: &ProjectConfig) -> Result<Option<SqliteDb>> {
    let path = cfg.db_path();
    if !Path::new(&path).exists() {
        return Ok(None);
    }
    Ok(Some(SqliteDb::open(&path, Duration::from_secs(5))?))
}

async fn login_local(id: Option<String>, no_browser: bool, cfg: ProjectConfig) -> Result<()> {
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
    println!("Open this URL to authorize:\n  {}", pending.url);
    if !no_browser && webbrowser::open(&pending.url).is_ok() {
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
    // Keyed by id, not by URL: two ids naming one server are two accounts.
    oauth::TokenStore::put(&store, DEFAULT_TENANT, &id, tokens)
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

async fn list_local(cfg: ProjectConfig) -> Result<()> {
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
                Some(store) => match oauth::TokenStore::get(store, DEFAULT_TENANT, id).await {
                    Some(tokens) => describe(&tokens, &spec.url),
                    None => "not authorized".to_string(),
                },
                None => "not authorized".to_string(),
            },
        };
        println!("{id}\t{}\t{status}", spec.url);
    }
    Ok(())
}

fn describe(tokens: &oauth::Tokens, url: &str) -> String {
    // The credential is keyed by id, so an edited `url` leaves one that the
    // resolver will refuse to send. Saying so beats reading as authorized.
    if !oauth::same_origin(&tokens.resource, url) {
        return format!("authorized for {}; log in again", tokens.resource);
    }
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
    scope: ProjectScope,
    cfg: ProjectConfig,
) -> Result<()> {
    let (id, spec) = pick(&cfg.mcp, id)?;
    if spec.auth.is_some() {
        bail!(
            "`{id}` names `token_env`, which the deployment cannot read: it holds its own \
             credential. Drop `auth` to authorize it there."
        );
    }
    let ctx = CloudContext::load(&scope.globals)?;
    let org = ctx.require_org(scope.org.as_deref()).await?;

    // Resolved before declaring, not after: a connection belongs to a project,
    // so the project is what decides which credential this consent replaces.
    let project = ctx
        .pinned_project(scope.project.as_deref())
        .await?
        .context("no project to connect for. Pin one with `subs apply`, or pass --project <id>.")?;

    // Declaring first is idempotent and inert: `subs apply` may already have
    // done it, and either way consent is the step that matters.
    let declared: McpConnection = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/mcp/connections"),
            &McpDeclareRequest {
                url: spec.url.clone(),
                connection_id: Some(id.clone()),
                project_id: project.clone(),
            },
        )
        .await?;
    let started: McpAuthorizeResponse = ctx
        .client
        .post_empty(&format!(
            "/api/v1/orgs/{org}/mcp/connections/{}/authorize",
            declared.id
        ))
        .await?;

    if let Some(requested) = &started.scope {
        println!("Requesting: {requested}");
    }
    println!("Open this URL to authorize:\n  {}", started.authorize_url);
    if !no_browser && webbrowser::open(&started.authorize_url).is_ok() {
        println!("Opened your browser to authorize.");
    }

    if wait_for_connection(&ctx, &org, &declared.id)
        .await?
        .is_none()
    {
        println!(
            "Still waiting on consent. Finish in the browser; the connection appears \
             in `subs mcp list` once it lands."
        );
        return Ok(());
    }
    println!("Authorized `{id}` for {project}.");
    Ok(())
}

/// Poll until the callback has stored the credential. Watches the row that was
/// declared rather than the id: two projects can hold a connection of one name,
/// and only this one's consent is being waited on.
///
/// Bounded: the person may have closed the tab, and a CLI that never returns is
/// worse than one that says where to look.
async fn wait_for_connection(
    ctx: &CloudContext,
    org: &str,
    row_id: &str,
) -> Result<Option<McpConnection>> {
    let deadline = std::time::Instant::now() + REMOTE_TIMEOUT;
    loop {
        let connections: Vec<McpConnection> = ctx
            .client
            .get(&format!("/api/v1/orgs/{org}/mcp/connections"))
            .await?;
        if let Some(found) = connections
            .into_iter()
            .find(|c| c.id == row_id && c.status == "active")
        {
            return Ok(Some(found));
        }
        if std::time::Instant::now() + POLL_INTERVAL > deadline {
            return Ok(None);
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// The connections a project holds, which is where a name resolves to exactly
/// one. `project` is passed as a filter the deployment applies; an older one
/// ignores it and answers with the org's, which is what it used to hold anyway.
async fn connections_for(
    ctx: &CloudContext,
    org: &str,
    project: Option<&str>,
) -> Result<Vec<McpConnection>> {
    let path = match project {
        Some(project) => format!("/api/v1/orgs/{org}/mcp/connections?project={project}"),
        None => format!("/api/v1/orgs/{org}/mcp/connections"),
    };
    ctx.client.get(&path).await
}

async fn list_remote(scope: ProjectScope, cfg: ProjectConfig) -> Result<()> {
    if cfg.mcp.is_empty() {
        bail!("substructure.toml declares no connections under `[mcp.<id>]`");
    }
    let ctx = CloudContext::load(&scope.globals)?;
    let org = ctx.require_org(scope.org.as_deref()).await?;
    let project = ctx.pinned_project(scope.project.as_deref()).await?;
    let connections = connections_for(&ctx, &org, project.as_deref()).await?;

    for (id, spec) in &cfg.mcp {
        let status = match connections.iter().find(|c| &c.connection_id == id) {
            Some(found) => found.status.clone(),
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
                "[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n\
                 auth = { token_env = \"SENTRY_TOKEN\" }\n",
            )
            .unwrap();
            path
        })
        .unwrap()
        .config;

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
        let cfg: ProjectConfig = toml::from_str(&format!(
            "db = {:?}\n[mcp.linear]\nurl = \"https://mcp.linear.app/mcp\"\n",
            db_path.to_str().unwrap()
        ))
        .unwrap();

        let store = SqliteTokenStore::new(open_db(&cfg).unwrap()).unwrap();
        oauth::TokenStore::put(&store, DEFAULT_TENANT, "linear", tokens())
            .await
            .unwrap();

        // What `list` reads, and what the engine keeps: the file declares it.
        assert!(oauth::TokenStore::get(&store, DEFAULT_TENANT, "linear")
            .await
            .is_some());
        let declared: Vec<String> = cfg.connections().keys().cloned().collect();
        assert!(store
            .retain(DEFAULT_TENANT, &declared)
            .await
            .unwrap()
            .is_empty());
        assert!(oauth::TokenStore::get(&store, DEFAULT_TENANT, "linear")
            .await
            .is_some());
    }

    /// Two ids naming one server are two accounts: dropping one from the file
    /// must not disconnect the other.
    #[tokio::test]
    async fn one_server_declared_twice_holds_two_credentials() {
        let dir = tmpdir();
        let db_path = dir.join("engine.db");
        let cfg: ProjectConfig = toml::from_str(&format!(
            "db = {:?}\n\
             [mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n\
             [mcp.sentry2]\nurl = \"https://mcp.sentry.dev/mcp\"\n",
            db_path.to_str().unwrap()
        ))
        .unwrap();

        let store = SqliteTokenStore::new(open_db(&cfg).unwrap()).unwrap();
        let sentry = oauth::Tokens {
            access_token: "first".into(),
            issuer: "https://mcp.sentry.dev".into(),
            token_endpoint: "https://mcp.sentry.dev/token".into(),
            resource: "https://mcp.sentry.dev/mcp".into(),
            ..tokens()
        };
        let sentry2 = oauth::Tokens {
            access_token: "second".into(),
            ..sentry.clone()
        };
        oauth::TokenStore::put(&store, DEFAULT_TENANT, "sentry", sentry)
            .await
            .unwrap();
        oauth::TokenStore::put(&store, DEFAULT_TENANT, "sentry2", sentry2)
            .await
            .unwrap();

        // `[mcp.sentry2]` taken out of the file: the engine forgets that one
        // credential the next time it starts, and only that one.
        let without: ProjectConfig = toml::from_str(&format!(
            "db = {:?}\n[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n",
            db_path.to_str().unwrap()
        ))
        .unwrap();
        let declared: Vec<String> = without.connections().keys().cloned().collect();
        assert_eq!(
            store.retain(DEFAULT_TENANT, &declared).await.unwrap(),
            ["sentry2"]
        );

        assert_eq!(
            oauth::TokenStore::get(&store, DEFAULT_TENANT, "sentry")
                .await
                .unwrap()
                .access_token,
            "first"
        );
        assert!(oauth::TokenStore::get(&store, DEFAULT_TENANT, "sentry2")
            .await
            .is_none());

        // And starting again on the same file has nothing left to forget.
        assert!(store
            .retain(DEFAULT_TENANT, &declared)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn listing_does_not_create_a_database() {
        let dir = tmpdir();
        let db_path = dir.join("absent.db");
        let cfg: ProjectConfig = toml::from_str(&format!(
            "db = {:?}\n[mcp.linear]\nurl = \"https://mcp.linear.app/mcp\"\n",
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
        let url = "https://mcp.linear.app/mcp";
        let mut tokens = tokens();
        assert!(describe(&tokens, url).starts_with("authorized, expires"));

        tokens.refresh_token = None;
        tokens.expires_at = Some(chrono::Utc::now() - chrono::Duration::hours(1));
        assert!(describe(&tokens, url).starts_with("expired"));

        // Refreshable and stale is not expired: the resolver renews it.
        tokens.refresh_token = Some("r".into());
        assert!(describe(&tokens, url).starts_with("authorized, expires"));

        tokens.expires_at = None;
        assert_eq!(describe(&tokens, url), "authorized");
    }

    /// A `url` edited after the login reads as what it is, rather than as a
    /// credential the resolver would refuse to send.
    #[test]
    fn a_credential_for_another_server_does_not_read_as_authorized() {
        let status = describe(&tokens(), "https://mcp.sentry.dev/mcp");
        assert!(status.contains("mcp.linear.app"), "{status}");
        assert!(status.contains("log in again"), "{status}");
    }
}
