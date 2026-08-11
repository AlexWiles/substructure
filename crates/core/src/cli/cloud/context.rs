use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use reqwest::StatusCode;

use crate::api::v1::Meta;

use super::credentials;
use super::http::{self, CloudClient};
use super::login;
use super::pickers;
use super::project_config::{self, ProjectConfig};
use super::{CloudGlobals, OrgScope, ProjectScope};

/// The server a credential command targets, without building a [`Context`]:
/// flag > the environment file's `[remote].url` > `$SUBS_API_URL` >
/// default. Same order [`Context::with_config`] applies, so `subs login -c
/// f.toml` writes the token under the URL every other command run with `-c
/// f.toml` reads.
pub fn api_url(globals: &CloudGlobals) -> Result<String> {
    let project = project_config::load(globals.config.as_deref())?;
    Ok(credentials::resolve_api_url(
        globals.url.as_deref().or(project.remote_url()),
    ))
}

/// What every cloud command does about a credential the deployment refuses:
/// log in and carry on, rather than stop at a 401 whose only advice is to run
/// `subs login` and start over.
fn reauth(globals: &CloudGlobals, api_url: String) -> http::Reauth {
    let globals = globals.clone();
    Box::new(move || {
        let globals = globals.clone();
        let api_url = api_url.clone();
        Box::pin(async move { login::refresh(&globals, &api_url).await })
    })
}

pub struct Context {
    /// The file this invocation read, if it read one — not to be confused with
    /// the project it pins.
    pub config: Option<ProjectConfig>,
    pub client: CloudClient,
    pub globals: CloudGlobals,
    /// Why the defaults probe learned nothing, kept so a caller that finds no
    /// default can say the server was unreachable instead of blaming the file.
    probe_error: Mutex<Option<String>>,
}

impl Context {
    pub fn load(globals: &CloudGlobals) -> Result<Self> {
        let config = project_config::resolve(globals.config.as_deref())?.map(|found| found.config);
        Self::with_config(globals, config)
    }

    /// A context for a command that reports rather than acts: a refusal is the
    /// answer it prints, not a reason to log in. `subs whoami` says who this
    /// machine is, and a login would make that true instead of reporting it.
    pub fn reporting(globals: &CloudGlobals) -> Result<Self> {
        let config = project_config::resolve(globals.config.as_deref())?.map(|found| found.config);
        Self::build(globals, config, false)
    }

    /// A context over a file the caller resolved itself, for `subs link` —
    /// which may be creating the file every other command reads.
    pub fn with_config(globals: &CloudGlobals, config: Option<ProjectConfig>) -> Result<Self> {
        Self::build(globals, config, true)
    }

    fn build(
        globals: &CloudGlobals,
        config: Option<ProjectConfig>,
        login_on_refusal: bool,
    ) -> Result<Self> {
        let credentials_path = credentials::resolve_path(globals.credentials.clone())?;
        let creds = credentials::load(&credentials_path)?;
        // Precedence: --url flag > the file's [remote].url > $SUBS_API_URL > default.
        let url_override = globals
            .url
            .as_deref()
            .or_else(|| config.as_ref().and_then(|p| p.remote_url()));
        let api_url = credentials::resolve_api_url(url_override);
        let token = credentials::resolve_token(&creds, &api_url);
        let client = CloudClient::new(api_url.clone(), token);
        let client = match login_on_refusal {
            true => client.with_reauth(reauth(globals, api_url)),
            false => client,
        };
        Ok(Self {
            config,
            client,
            globals: globals.clone(),
            probe_error: Mutex::new(None),
        })
    }

    pub async fn from_org(scope: &OrgScope) -> Result<(Self, String)> {
        let ctx = Self::load(&scope.globals)?;
        let org = ctx.require_org(scope.org.as_deref()).await?;
        Ok((ctx, org))
    }

    /// A context over the project a deployment holds.
    ///
    /// Every command that acts on one comes through here, so this is where the
    /// rule is enforced: a file naming no `[remote]` describes an engine you
    /// run here, and a command that only a deployment can answer says so rather
    /// than reaching for the default server and asking you to log in to it. A
    /// command with something to say about both branches before this.
    pub async fn from_project(scope: &ProjectScope) -> Result<(Self, String)> {
        crate::cli::target::require_deployment(&scope.globals, "this command")?;
        let ctx = Self::load(&scope.globals)?;

        // If we already have a project id (flag or pinned), skip org resolution
        // entirely — project-scoped routes don't need it.
        if let Some(project) = scope
            .project
            .clone()
            .or_else(|| ctx.pinned(ProjectConfig::project))
        {
            return Ok((ctx, project));
        }

        // A single-tenant server (e.g. a local server) advertises its org/project
        // in response headers; adopt the project and skip the picker entirely.
        if let Some(project) = ctx.server_default_project().await? {
            return Ok((ctx, project));
        }

        // Need to pick. Picker enumerates projects via /orgs/:org/projects, so it
        // still needs an org — but only for the picker, not for the URL we
        // act on afterwards.
        if !pickers::interactive(&ctx.globals) {
            bail!("no project selected. Pass --project <id>.");
        }
        let org = ctx.require_org(scope.org.as_deref()).await?;
        let project = match pickers::pick_project(&ctx, &org).await? {
            Some(a) => a,
            None => bail!("no project selected"),
        };
        Ok((ctx, project))
    }

    /// The project to act on without a picker: flag, then the file's pin, then
    /// what a single-tenant server advertises. For callers that can carry on
    /// without one.
    pub async fn pinned_project(&self, flag: Option<&str>) -> Result<Option<String>> {
        if let Some(project) = flag {
            return Ok(Some(project.to_string()));
        }
        if let Some(project) = self.pinned(ProjectConfig::project) {
            return Ok(Some(project));
        }
        self.server_default_project().await
    }

    /// What the file pins, if this invocation read one.
    fn pinned(&self, key: fn(&ProjectConfig) -> Option<&str>) -> Option<String> {
        key(self.config.as_ref()?).map(str::to_string)
    }

    pub async fn require_org(&self, flag: Option<&str>) -> Result<String> {
        if let Some(s) = flag {
            return Ok(s.to_string());
        }
        if let Some(org) = self.pinned(ProjectConfig::org) {
            return Ok(org);
        }
        if let Some(org) = self.server_default_org().await? {
            return Ok(org);
        }
        if pickers::interactive(&self.globals) {
            return pickers::pick_org(self).await;
        }
        bail!("no org selected. Pass --org <id>.")
    }

    /// What the deployment says it is and what it offers.
    ///
    /// A deployment that answers 404 from the API predates this endpoint, and
    /// reads as one that advertises nothing. A 404 the API did not write is a
    /// proxy or the wrong URL answering, and every other failure is itself —
    /// neither says anything about the deployment, so both are reported.
    pub async fn meta(&self) -> Result<Meta> {
        match self.client.get::<Meta>("/api/v1/meta").await {
            Ok(meta) => Ok(meta),
            Err(e) if http::api_status_of(&e) == Some(StatusCode::NOT_FOUND) => Ok(Meta::default()),
            Err(e) => Err(e),
        }
    }

    /// The org a single-tenant server advertises, or None against the cloud.
    pub async fn server_default_org(&self) -> Result<Option<String>> {
        self.probe_server_defaults().await?;
        Ok(self.client.default_org())
    }

    /// The project a single-tenant server advertises, or None against the cloud.
    pub async fn server_default_project(&self) -> Result<Option<String>> {
        self.probe_server_defaults().await?;
        Ok(self.client.default_project())
    }

    // Defaults ride on response headers (captured in CloudClient), so one
    // throwaway request is enough to learn them; the cloud sends none. A probe
    // that answered headers succeeded whatever its status said — only a probe
    // that taught us nothing reports why.
    async fn probe_server_defaults(&self) -> Result<()> {
        if self.client.needs_default_probe() {
            if let Err(e) = self.client.get::<serde_json::Value>("/api/v1/orgs").await {
                *self.probe_error.lock().unwrap() = Some(format!("{e:#}"));
            }
        }
        if self.client.default_org().is_some() || self.client.default_project().is_some() {
            return Ok(());
        }
        match &*self.probe_error.lock().unwrap() {
            Some(e) => Err(anyhow!("{e}")),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_config(body: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("subs-context-test-{nanos}-{seq}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(project_config::FILENAME);
        std::fs::write(&path, body).unwrap();
        path
    }

    fn globals(url: Option<&str>, config: Option<PathBuf>) -> CloudGlobals {
        CloudGlobals {
            url: url.map(str::to_string),
            config,
            ..Default::default()
        }
    }

    #[test]
    fn the_deployment_names_the_server_and_the_flag_still_wins() {
        let path = write_config("[remote]\nurl = \"https://self.example\"\n");

        assert_eq!(
            api_url(&globals(None, Some(path.clone()))).unwrap(),
            "https://self.example"
        );
        assert_eq!(
            api_url(&globals(Some("https://flag.example"), Some(path))).unwrap(),
            "https://flag.example"
        );
    }

    /// The failure a caller must not read as "you pinned nothing".
    #[tokio::test]
    async fn a_server_that_never_answers_is_reported_as_such_not_as_a_missing_pin() {
        let globals = globals(Some("http://127.0.0.1:1"), None);
        let ctx = Context::with_config(&globals, None).unwrap();

        let err = ctx.require_org(None).await.unwrap_err();
        let chain = format!("{err:#}");
        assert!(
            chain.starts_with("could not connect to http://127.0.0.1:1"),
            "{chain}"
        );
        assert!(!chain.contains("no org selected"), "{chain}");

        // The probe runs once; a second caller gets the same reason, not silence.
        let err = ctx.server_default_project().await.unwrap_err();
        assert!(format!("{err:#}").contains("could not connect to"));
    }

    /// A server that answers 404 to everything, with the body given.
    async fn serve_404(body: &'static str) -> String {
        let app =
            axum::Router::new().fallback(move || async move { (StatusCode::NOT_FOUND, body) });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    /// Only the API's own 404 means a deployment that predates `/meta`. A proxy,
    /// a dev server, or the wrong port answering says nothing about what the
    /// deployment offers, so it must not read as one that offers nothing.
    #[tokio::test]
    async fn a_404_the_api_did_not_write_is_not_a_deployment_that_advertises_nothing() {
        let url = serve_404("<html>404 not found</html>").await;
        let ctx = Context::with_config(&globals(Some(&url), None), None).unwrap();
        let err = format!("{:#}", ctx.meta().await.unwrap_err());
        assert!(err.contains("not an API response"), "{err}");

        let url = serve_404(r#"{"error":{"code":"not_found","message":"No such endpoint"}}"#).await;
        let ctx = Context::with_config(&globals(Some(&url), None), None).unwrap();
        let meta = ctx
            .meta()
            .await
            .expect("a 404 from the API predates `/meta`");
        assert!(meta.features.is_empty());
    }

    #[test]
    fn a_file_that_names_no_deployment_leaves_the_server_alone() {
        let path = write_config("db = \"dev.db\"\n");
        // Whatever the env/default resolves to — the file must not change it.
        assert_eq!(
            api_url(&globals(None, Some(path))).unwrap(),
            credentials::resolve_api_url(None)
        );
    }
}
