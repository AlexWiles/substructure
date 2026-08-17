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
use super::cloud::{notices, pickers, CloudGlobals, ProjectScope};
use super::{DEFAULT_TENANT, LOCAL_SUBJECT};
use crate::api::v1::{McpAuthorizeResponse, McpConnection, McpDeclareRequest, McpTokenRequest};
use crate::connectors::credential::{Credential, CredentialStore};
use crate::connectors::oauth;
use crate::connectors::registry::{AuthKind, ConnectionSpec, CredentialScope};
use crate::connectors::Slot;
use crate::providers::sqlite::{SqliteCredentialStore, SqliteDb, SqliteSecretStore};
use crate::runtime::secret::SecretCipher;

/// How long the listener waits for the browser before giving up.
const CONSENT_TIMEOUT: Duration = Duration::from_secs(300);

/// How long a remote login waits for consent to land on the server.
const REMOTE_TIMEOUT: Duration = Duration::from_secs(300);
const POLL_INTERVAL: Duration = Duration::from_secs(3);

const CLIENT_NAME: &str = "Substructure";

#[derive(Subcommand)]
pub enum McpCommand {
    /// Authorize an `auth = "oauth"` connection, opening a browser for consent.
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
    /// Set the token for an `auth = "token"` connection. Typed at a prompt,
    /// piped in on stdin, or read from the environment variable `--env` names.
    SetToken {
        id: Option<String>,
        /// Read the token from this environment variable instead of stdin.
        #[arg(long, value_name = "VAR")]
        env: Option<String>,
        #[command(flatten)]
        scope: ProjectScope,
    },
    /// Forget a connection's credentials — every holder's, whichever way they
    /// were obtained. Calls on it fail until somebody connects again.
    #[command(name = "logout")]
    Logout {
        id: Option<String>,
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
        McpCommand::SetToken { id, env, scope } => match environment(&scope.globals)? {
            cfg if cfg.remote.is_none() => set_token_local(id, env, &scope.globals, cfg).await,
            cfg => set_token_remote(id, env, scope, cfg).await,
        },
        McpCommand::Logout { id, scope } => match environment(&scope.globals)? {
            cfg if cfg.remote.is_none() => delete_token_local(id, cfg).await,
            cfg => delete_token_remote(id, scope, cfg).await,
        },
        McpCommand::List { scope } => match environment(&scope.globals)? {
            cfg if cfg.remote.is_none() => list_local(cfg).await,
            cfg => list_remote(scope, cfg).await,
        },
    }
}

/// The cipher the environment configures, with a set-and-wrong variable
/// reported rather than read as "no cipher".
fn cipher() -> Result<Option<Arc<SecretCipher>>> {
    Ok(SecretCipher::from_env()
        .map_err(|e| anyhow::anyhow!(e))?
        .map(Arc::new))
}

/// The two stores every local command reads and writes through: references in
/// `connector_credentials`, material in `secrets`.
fn open_store(db: SqliteDb) -> Result<SqliteCredentialStore> {
    let secrets = Arc::new(SqliteSecretStore::new(db.clone(), cipher()?));
    Ok(SqliteCredentialStore::new(db, secrets)?)
}

/// Checked before a command that only makes sense under one method. A file that
/// declares nothing is the usual case and passes: the server is asked instead.
fn require_auth(id: &str, spec: &ConnectionSpec, want: AuthKind) -> Result<()> {
    let declared = match spec.auth {
        // Nothing declared: discovery decides, and either command may be the
        // one that fits. `login` finds out; `set-token` is what a file has to
        // ask for, since nothing on the wire announces a static token.
        None if want == AuthKind::Oauth => return Ok(()),
        None => bail!(
            "`{id}` declares no `auth`, so its token would never be sent. \
             Write `auth = \"token\"` on [mcp.{id}] first."
        ),
        Some(declared) if declared == want => return Ok(()),
        Some(declared) => declared,
    };
    let fix = match declared {
        AuthKind::Oauth => format!("authorize it with `subs mcp login {id}`"),
        AuthKind::Token => format!("set its token with `subs mcp set-token {id}`"),
        AuthKind::None => "it declares that it needs no credential".to_string(),
    };
    bail!("`{id}` declares `auth = \"{}\"`: {fix}", declared.as_str())
}

/// Whose slot a local command reads or writes. Consent given here is
/// whoever ran the command, so no other person's slot is reachable.
fn local_slot(spec: &ConnectionSpec) -> Slot {
    match spec.effective_scope() {
        CredentialScope::Shared => Slot::Shared,
        CredentialScope::User => Slot::Of(crate::protocol::Subject::new(
            crate::protocol::Issuer::cli(),
            LOCAL_SUBJECT,
        )),
    }
}

/// A server whose OAuth needs an app somebody registered by hand.
///
/// The flow is a dead end here, so the message leads with that in the reader's
/// terms and spends its last line on the way through rather than on the
/// mechanism they cannot reach.
fn unregistrable(id: &str, err: oauth::OauthError) -> anyhow::Error {
    if !matches!(err, oauth::OauthError::Registration(_)) {
        return err.into();
    }
    anyhow::anyhow!(
        "{err}.\n\n\
         If the server accepts a static token, add `auth = \"token\"` to [mcp.{id}], \
         then:\n\n  subs mcp set-token {id}"
    )
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
    let (id, spec) = pick(&cfg.resolved_connections()?, id)?;
    require_auth(&id, &spec, AuthKind::Oauth)?;
    let subject = local_slot(&spec);

    let http = reqwest::Client::new();

    println!("Discovering {}...", spec.url);
    // What the server says it wants, where the file did not. A file that
    // declared OAuth is taken at its word: it is there to override exactly this.
    if spec.auth.is_none() {
        match oauth::sniff(&http, &spec.url).await? {
            oauth::Probed::NoChallenge => {
                println!("{} did not ask for a credential. Nothing to do.", spec.url);
                return Ok(());
            }
            oauth::Probed::Protected => bail!(
                "{} wants a credential but publishes no way to get one. If it takes a \
                 static token, write `auth = \"token\"` on [mcp.{id}] and run \
                 `subs mcp set-token {id}`.",
                spec.url
            ),
            oauth::Probed::Oauth => {}
        }
    }
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
    let client = oauth::client_id(&http, &discovered.server, None, &redirect_uri, CLIENT_NAME)
        .await
        .map_err(|e| unregistrable(&id, e))?;

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

    let refreshable = tokens.refreshable();
    // Keyed by id, not by URL: two ids naming one server are two accounts.
    store_credential(&cfg, &id, &subject, Credential::Oauth(Box::new(tokens))).await?;

    println!("Authorized `{id}` in {}.", cfg.db_path());
    if !refreshable {
        println!(
            "Note: the server issued no refresh token, so this expires and needs `subs mcp login {id}` again."
        );
    }
    Ok(())
}

/// The token, typed or piped, the way `subs llm set-key` takes one: a static
/// credential is configuration, and it never appears in argv where a shell
/// history would keep it.
async fn set_token_local(
    id: Option<String>,
    env: Option<String>,
    globals: &CloudGlobals,
    cfg: ProjectConfig,
) -> Result<()> {
    let (id, spec) = pick(&cfg.resolved_connections()?, id)?;
    require_auth(&id, &spec, AuthKind::Token)?;
    let subject = local_slot(&spec);

    let token = match &env {
        Some(var) => super::env_value(var)
            .with_context(|| format!("${var} is not set"))?
            .trim()
            .to_string(),
        None => pickers::read_secret(globals, "Paste the token")?,
    };
    if token.is_empty() {
        bail!("no token given. Pipe it in, or pass --env <VAR>.");
    }

    store_credential(&cfg, &id, &subject, Credential::Static { token }).await?;
    println!("Token set for `{id}` in {}.", cfg.db_path());
    Ok(())
}

async fn delete_token_local(id: Option<String>, cfg: ProjectConfig) -> Result<()> {
    let (id, _) = pick(&cfg.resolved_connections()?, id)?;
    let Some(db) = open_existing_db(&cfg)? else {
        println!("`{id}` holds no credential.");
        return Ok(());
    };
    let store = open_store(db)?;
    match store.delete(DEFAULT_TENANT, &id).await? {
        true => println!("Forgot the credential for `{id}`."),
        false => println!("`{id}` holds no credential."),
    }
    Ok(())
}

async fn store_credential(
    cfg: &ProjectConfig,
    id: &str,
    subject: &Slot,
    credential: Credential,
) -> Result<()> {
    let store = open_store(open_db(cfg)?)?;
    CredentialStore::put(&store, DEFAULT_TENANT, id, subject, credential)
        .await
        .map_err(|e| anyhow::anyhow!("storing the credential: {e}"))
}

async fn list_local(cfg: ProjectConfig) -> Result<()> {
    let connections = cfg.resolved_connections()?;
    if connections.is_empty() {
        bail!("substructure.toml declares no connections under `[mcp.<id>]`");
    }
    let store = match open_existing_db(&cfg)? {
        Some(db) => Some(open_store(db)?),
        None => None,
    };

    for (id, spec) in &connections {
        if spec.effective_scope() == CredentialScope::Shared || spec.auth == Some(AuthKind::None) {
            let held = match &store {
                Some(store) => CredentialStore::get(store, DEFAULT_TENANT, id, &Slot::Shared).await,
                None => None,
            };
            println!("{id}\t{}\t{}", spec.url, describe(spec, held.as_ref()));
            continue;
        }

        let holders = match &store {
            Some(store) => store.holders(DEFAULT_TENANT, id).await?,
            None => Vec::new(),
        };
        println!("{id}\t{}\t{}", spec.url, connected(holders.len()));
    }
    Ok(())
}

fn connected(holders: usize) -> String {
    match holders {
        0 => "nobody connected".to_string(),
        n => format!("{n} connected"),
    }
}

/// What one connection's slot holds, read against what the file declares.
fn describe(spec: &ConnectionSpec, held: Option<&Credential>) -> String {
    if spec.auth == Some(AuthKind::None) {
        return "no credential needed".to_string();
    }
    let Some(held) = held else {
        // Nothing declared and nothing held is not a problem yet: the server
        // has not been asked, and may want nothing.
        return match spec.auth {
            Some(AuthKind::Token) => "no token set".to_string(),
            Some(_) => "not authorized".to_string(),
            None => "not connected".to_string(),
        };
    };
    if spec.auth.is_some_and(|declared| declared != held.kind()) {
        return format!("holds {}; not authorized", held.kind().as_str());
    }
    match held {
        Credential::Static { .. } => "token set".to_string(),
        Credential::Oauth(tokens) => {
            // The credential is keyed by id, so an edited `url` leaves one that
            // the resolver will refuse to send. Saying so beats reading as
            // authorized.
            if !oauth::same_origin(&tokens.resource, &spec.url) {
                return format!("authorized for {}; log in again", tokens.resource);
            }
            match tokens.expires_at {
                Some(at) if expired(tokens) => {
                    format!("expired {}", at.format("%Y-%m-%d %H:%M UTC"))
                }
                Some(at) => format!("authorized, expires {}", at.format("%Y-%m-%d %H:%M UTC")),
                None => "authorized".to_string(),
            }
        }
    }
}

/// Run out and unable to renew, which the resolver cannot send.
fn expired(tokens: &oauth::Tokens) -> bool {
    tokens.stale() && !tokens.refreshable()
}

/// What a connection still needs before an engine here could dial it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Needs {
    /// A consent nobody has given, or one that has run out.
    Login,
    /// A token the file asked for and nobody set.
    Token,
    /// A credential the server wants and publishes no way to obtain, so the
    /// file has to say how.
    Declaration,
}

/// Declared connections an engine here could not dial, each with what it wants.
///
/// A connection the file describes is answered from the store alone. One it
/// says nothing about is asked of the server, because nothing else can tell a
/// server that wants no credential from one nobody has authorized — and
/// reporting the first as unauthorized would be a permanent false alarm about a
/// connection that works.
pub(crate) async fn unauthorized_local(cfg: &ProjectConfig) -> Result<Vec<(String, Needs)>> {
    let declared: Vec<(String, ConnectionSpec)> = cfg
        .resolved_connections()?
        .into_iter()
        .filter(|(_, spec)| spec.auth != Some(AuthKind::None))
        .collect();
    let store = match open_existing_db(cfg)? {
        Some(db) => Some(open_store(db)?),
        // No engine has ever run here, so nothing is authorized — and a
        // database is not worth creating to say so.
        None => None,
    };

    let http = reqwest::Client::new();
    let mut out = Vec::new();
    for (id, spec) in declared {
        let slot = local_slot(&spec);
        let held = match &store {
            Some(store) => CredentialStore::get(store, DEFAULT_TENANT, &id, &slot).await,
            None => None,
        };
        let usable = match &held {
            Some(Credential::Static { .. }) => spec.auth == Some(AuthKind::Token),
            Some(Credential::Oauth(tokens)) => {
                spec.auth != Some(AuthKind::Token)
                    && oauth::same_origin(&tokens.resource, &spec.url)
                    && !expired(tokens)
            }
            None => false,
        };
        if usable {
            continue;
        }
        let needs = match spec.auth {
            Some(AuthKind::Token) => Needs::Token,
            Some(_) => Needs::Login,
            // Unreachable is not unauthorized: saying what a server wants means
            // having asked it.
            None => match oauth::sniff(&http, &spec.url).await {
                Ok(oauth::Probed::Oauth) => Needs::Login,
                Ok(oauth::Probed::Protected) => Needs::Declaration,
                Ok(oauth::Probed::NoChallenge) | Err(_) => continue,
            },
        };
        out.push((id, needs));
    }
    Ok(out)
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
    require_auth(&id, &spec, AuthKind::Oauth)?;
    let (ctx, org, project) = remote_target(&scope).await?;

    // Declaring first is idempotent and inert: `subs apply` may already have
    // done it, and either way consent is the step that matters.
    let declared: McpConnection = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/mcp/connections"),
            &declare(&id, &spec, &project),
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
    notices::remaining(&ctx, &project, &scope.globals).await;
    Ok(())
}

/// What a declare says: where the connection points and how it authenticates.
/// The deployment holds the credential, so this is the whole of what it needs
/// from the file.
fn declare(id: &str, spec: &ConnectionSpec, project: &str) -> McpDeclareRequest {
    McpDeclareRequest {
        url: spec.url.clone(),
        connection_id: Some(id.to_string()),
        project_id: project.to_string(),
        auth: spec.auth.map(|a| a.as_str().to_string()),
        header: spec.header.clone(),
    }
}

/// The deployment holds the token and the engine there sends it; nothing is
/// written on this machine. Declaring first is idempotent, so a token may be
/// set before the first `subs apply`.
async fn set_token_remote(
    id: Option<String>,
    env: Option<String>,
    scope: ProjectScope,
    cfg: ProjectConfig,
) -> Result<()> {
    let (id, spec) = pick(&cfg.mcp, id)?;
    require_auth(&id, &spec, AuthKind::Token)?;

    let token = match &env {
        Some(var) => super::env_value(var)
            .with_context(|| format!("${var} is not set"))?
            .trim()
            .to_string(),
        None => pickers::read_secret(&scope.globals, "Paste the token")?,
    };
    if token.is_empty() {
        bail!("no token given. Pipe it in, or pass --env <VAR>.");
    }

    let (ctx, org, project) = remote_target(&scope).await?;
    let declared: McpConnection = ctx
        .client
        .post_json(
            &format!("/api/v1/orgs/{org}/mcp/connections"),
            &declare(&id, &spec, &project),
        )
        .await?;
    ctx.client
        .put_discard(
            &format!("/api/v1/orgs/{org}/mcp/connections/{}/token", declared.id),
            &McpTokenRequest { token },
        )
        .await?;

    println!("Token set for `{id}` in {project}.");
    notices::remaining(&ctx, &project, &scope.globals).await;
    Ok(())
}

async fn delete_token_remote(
    id: Option<String>,
    scope: ProjectScope,
    cfg: ProjectConfig,
) -> Result<()> {
    let (id, _) = pick(&cfg.mcp, id)?;
    let (ctx, org, project) = remote_target(&scope).await?;
    let found = connections_for(&ctx, &org, Some(&project))
        .await?
        .into_iter()
        .find(|c| c.connection_id == id);
    let Some(found) = found else {
        println!("`{id}` holds no credential.");
        return Ok(());
    };
    ctx.client
        .delete_discard(&format!(
            "/api/v1/orgs/{org}/mcp/connections/{}/token",
            found.id
        ))
        .await?;
    println!("Forgot the credential for `{id}`.");
    Ok(())
}

/// The org and project a remote connection command acts on. A connection
/// belongs to one project, so the project is resolved before anything is
/// declared: it decides which credential this replaces.
async fn remote_target(scope: &ProjectScope) -> Result<(CloudContext, String, String)> {
    let ctx = CloudContext::load(&scope.globals)?;
    let org = ctx.require_org(scope.org.as_deref()).await?;
    let project = ctx
        .pinned_project(scope.project.as_deref())
        .await?
        .context("no project to connect for. Pin one with `subs apply`, or pass --project <id>.")?;
    Ok((ctx, org, project))
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
    use crate::protocol::ConnectorProtocol;
    use std::path::PathBuf;

    fn spec(auth: Option<AuthKind>) -> ConnectionSpec {
        ConnectionSpec {
            url: "https://mcp.linear.app/mcp".into(),
            protocol: ConnectorProtocol::Mcp,
            auth,
            header: None,
            credential: None,
            prefix_tools: true,
        }
    }

    fn globals() -> CloudGlobals {
        CloudGlobals {
            no_interaction: true,
            ..Default::default()
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
        let one = BTreeMap::from([("linear".to_string(), spec(Some(AuthKind::Oauth)))]);
        assert_eq!(pick(&one, None).unwrap().0, "linear");

        let two = BTreeMap::from([
            ("linear".to_string(), spec(Some(AuthKind::Oauth))),
            ("sentry".to_string(), spec(Some(AuthKind::Oauth))),
        ]);
        let err = pick(&two, None).unwrap_err().to_string();
        assert!(err.contains("linear") && err.contains("sentry"), "{err}");

        let err = pick(&one, Some("nope".into())).unwrap_err().to_string();
        assert!(err.contains("not declared"), "{err}");

        let none: BTreeMap<String, ConnectionSpec> = BTreeMap::new();
        let err = pick(&none, None).unwrap_err().to_string();
        assert!(err.contains("declares no connections"), "{err}");
    }

    fn written(body: &str) -> ProjectConfig {
        let path = tmpdir().join(project_config::FILENAME);
        std::fs::write(&path, body).unwrap();
        project_config::load_explicit(&path).unwrap().config
    }

    /// Each command belongs to one declared method, and says which the other is
    /// rather than running a flow the connection does not use.
    #[tokio::test]
    async fn a_command_refuses_a_connection_declaring_the_other_method() {
        let token =
            written("[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\nauth = \"token\"\n");
        let err = login_local(Some("sentry".into()), true, token)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("subs mcp set-token sentry"), "{err}");

        let oauth =
            written("[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\nauth = \"oauth\"\n");
        let err = set_token_local(
            Some("sentry".into()),
            Some("PATH".into()),
            &globals(),
            oauth,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("subs mcp login sentry"), "{err}");

        // And a connection that declares nothing: a token it never sends is
        // worse than a refusal, since nothing would say why.
        let bare = written("[mcp.sentry]\nurl = \"https://mcp.sentry.dev/mcp\"\n");
        let err = set_token_local(Some("sentry".into()), Some("PATH".into()), &globals(), bare)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("declares no `auth`"), "{err}");
    }

    /// The token reaches the store the engine reads, and never argv.
    #[tokio::test]
    async fn a_token_set_here_is_what_the_resolver_would_send() {
        let dir = tmpdir();
        let db_path = dir.join("engine.db");
        let cfg: ProjectConfig = toml::from_str(&format!(
            "db = {:?}\n[mcp.github]\nurl = \"https://api.github.test/mcp\"\nauth = \"token\"\n",
            db_path.to_str().unwrap()
        ))
        .unwrap();

        std::env::set_var("SUBS_TEST_MCP_TOKEN", "ghp_written");
        set_token_local(
            Some("github".into()),
            Some("SUBS_TEST_MCP_TOKEN".into()),
            &globals(),
            cfg.clone(),
        )
        .await
        .unwrap();

        let store = open_store(open_db(&cfg).unwrap()).unwrap();
        assert_eq!(
            CredentialStore::get(&store, DEFAULT_TENANT, "github", &Slot::Shared).await,
            Some(Credential::Static {
                token: "ghp_written".into()
            })
        );

        // And forgetting it empties the slot rather than the file.
        delete_token_local(Some("github".into()), cfg)
            .await
            .unwrap();
        assert!(
            CredentialStore::get(&store, DEFAULT_TENANT, "github", &Slot::Shared)
                .await
                .is_none()
        );
    }

    /// A declared connection is answered from the file and the store alone.
    /// `auth = "none"` says there is nothing to do, so it is never reported;
    /// the other two are reported with the command that fills them.
    ///
    /// A connection declaring nothing is left out here: it is the one case that
    /// asks the server, which a test has no business doing.
    #[tokio::test]
    async fn what_a_declared_connection_still_needs() {
        let cfg = written(
            "[mcp.open]\nurl = \"https://tools.test/mcp\"\nauth = \"none\"\n\
             [mcp.linear]\nurl = \"https://mcp.linear.app/mcp\"\nauth = \"oauth\"\n\
             [mcp.github]\nurl = \"https://api.github.test/mcp\"\nauth = \"token\"\n",
        );
        assert_eq!(
            unauthorized_local(&cfg).await.unwrap(),
            [
                ("github".to_string(), Needs::Token),
                ("linear".to_string(), Needs::Login)
            ]
        );
    }

    fn oauth_of(tokens: oauth::Tokens) -> Credential {
        Credential::Oauth(Box::new(tokens))
    }

    fn granted() -> Credential {
        oauth_of(tokens())
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
        let spec = spec(Some(AuthKind::Oauth));
        let mut tokens = tokens();
        let of = |t: &oauth::Tokens| describe(&spec, Some(&oauth_of(t.clone())));
        assert!(of(&tokens).starts_with("authorized, expires"));

        tokens.refresh_token = None;
        tokens.expires_at = Some(chrono::Utc::now() - chrono::Duration::hours(1));
        assert!(of(&tokens).starts_with("expired"));

        // Refreshable and stale is not expired: the resolver renews it.
        tokens.refresh_token = Some("r".into());
        assert!(of(&tokens).starts_with("authorized, expires"));

        tokens.expires_at = None;
        assert_eq!(of(&tokens), "authorized");
    }

    /// A `url` edited after the login reads as what it is, rather than as a
    /// credential the resolver would refuse to send.
    #[test]
    fn a_credential_for_another_server_does_not_read_as_authorized() {
        let mut spec = spec(Some(AuthKind::Oauth));
        spec.url = "https://mcp.sentry.dev/mcp".into();
        let status = describe(&spec, Some(&granted()));
        assert!(status.contains("mcp.linear.app"), "{status}");
        assert!(status.contains("log in again"), "{status}");
    }

    /// The slot is shared, so what it holds is read against what the file
    /// declares rather than reported as whatever landed there.
    #[test]
    fn a_slot_holding_the_other_kind_does_not_read_as_authorized() {
        let held = Credential::Static { token: "t".into() };
        assert_eq!(
            describe(&spec(Some(AuthKind::Token)), Some(&held)),
            "token set"
        );
        assert_eq!(
            describe(&spec(Some(AuthKind::Oauth)), Some(&held)),
            "holds token; not authorized"
        );
        assert_eq!(
            describe(&spec(Some(AuthKind::None)), None),
            "no credential needed"
        );
        assert_eq!(describe(&spec(Some(AuthKind::Token)), None), "no token set");
        // Nothing declared and nothing held is not yet a problem: the server
        // has not been asked.
        assert_eq!(describe(&spec(None), None), "not connected");
    }
}
