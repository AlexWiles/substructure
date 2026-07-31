use anyhow::{bail, Result};

use super::credentials;
use super::http::CloudClient;
use super::pickers;
use super::project_config::{self, EnvConfig};
use super::{AppScope, CloudGlobals, OrgScope};

/// The server a credential command targets, without building a [`Context`]:
/// flag > the environment file's `[deployment].url` > `$SUBS_API_URL` >
/// default. Same order [`Context::with_project`] applies, so `subs login -c
/// f.toml` writes the token under the URL every other command run with `-c
/// f.toml` reads.
pub fn api_url(globals: &CloudGlobals) -> Result<String> {
    let project = project_config::load(globals.config.as_deref())?;
    Ok(credentials::resolve_api_url(
        globals.url.as_deref().or(project.deployment_url()),
    ))
}

pub struct Context {
    pub project: Option<EnvConfig>,
    pub client: CloudClient,
    pub globals: CloudGlobals,
}

impl Context {
    pub fn load(globals: &CloudGlobals) -> Result<Self> {
        let project = project_config::resolve(globals.config.as_deref())?.map(|found| found.config);
        Self::with_project(globals, project)
    }

    /// A context over an environment the caller resolved itself, for `subs
    /// link` — which may be creating the file every other command reads.
    pub fn with_project(globals: &CloudGlobals, project: Option<EnvConfig>) -> Result<Self> {
        let credentials_path = credentials::resolve_path(globals.credentials.clone())?;
        let creds = credentials::load(&credentials_path)?;
        // Precedence: --url flag > project substructure.toml url > $SUBS_API_URL > default.
        let url_override = globals
            .url
            .as_deref()
            .or_else(|| project.as_ref().and_then(|p| p.deployment_url()));
        let api_url = credentials::resolve_api_url(url_override);
        let token = credentials::resolve_token(&creds, &api_url);
        let client = CloudClient::new(api_url, token);
        Ok(Self {
            project,
            client,
            globals: globals.clone(),
        })
    }

    pub async fn from_org(scope: &OrgScope) -> Result<(Self, String)> {
        let ctx = Self::load(&scope.globals)?;
        let org = ctx.require_org(scope.org.as_deref()).await?;
        Ok((ctx, org))
    }

    pub async fn from_app(scope: &AppScope) -> Result<(Self, String)> {
        let ctx = Self::load(&scope.globals)?;

        // If we already have an app id (flag or pinned), skip org resolution
        // entirely — app-scoped routes don't need it.
        if let Some(app) = scope.app.clone().or_else(|| ctx.pinned(EnvConfig::app)) {
            return Ok((ctx, app));
        }

        // A single-tenant server (e.g. a local server) advertises its org/app
        // in response headers; adopt the app and skip the picker entirely.
        if let Some(app) = ctx.server_default_app().await {
            return Ok((ctx, app));
        }

        // Need to pick. Picker enumerates apps via /orgs/:org/apps, so it
        // still needs an org — but only for the picker, not for the URL we
        // act on afterwards.
        if !pickers::interactive(&ctx.globals) {
            bail!("no app selected. Pass --app <id>.");
        }
        let org = ctx.require_org(scope.org.as_deref()).await?;
        let app = match pickers::pick_app(&ctx, &org).await? {
            Some(a) => a,
            None => bail!("no app selected"),
        };
        Ok((ctx, app))
    }

    /// The app to act on without a picker: flag, then the file's pin, then
    /// what a single-tenant server advertises. For callers that can carry on
    /// without one.
    pub async fn pinned_app(&self, flag: Option<&str>) -> Option<String> {
        if let Some(app) = flag {
            return Some(app.to_string());
        }
        if let Some(app) = self.pinned(EnvConfig::app) {
            return Some(app);
        }
        self.server_default_app().await
    }

    /// What the file pins, if this invocation read one.
    fn pinned(&self, key: fn(&EnvConfig) -> Option<&str>) -> Option<String> {
        key(self.project.as_ref()?).map(str::to_string)
    }

    pub async fn require_org(&self, flag: Option<&str>) -> Result<String> {
        if let Some(s) = flag {
            return Ok(s.to_string());
        }
        if let Some(org) = self.pinned(EnvConfig::org) {
            return Ok(org);
        }
        if let Some(org) = self.server_default_org().await {
            return Ok(org);
        }
        if pickers::interactive(&self.globals) {
            return pickers::pick_org(self).await;
        }
        bail!("no org selected. Pass --org <id>.")
    }

    /// The org a single-tenant server advertises, or None against the cloud.
    pub async fn server_default_org(&self) -> Option<String> {
        self.probe_server_defaults().await;
        self.client.default_org()
    }

    /// The app a single-tenant server advertises, or None against the cloud.
    pub async fn server_default_app(&self) -> Option<String> {
        self.probe_server_defaults().await;
        self.client.default_app()
    }

    // Defaults ride on response headers (captured in CloudClient), so one
    // throwaway request is enough to learn them; the cloud sends none.
    async fn probe_server_defaults(&self) {
        if self.client.needs_default_probe() {
            let _ = self.client.get::<serde_json::Value>("/api/v1/orgs").await;
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
        let path = write_config("[deployment]\nurl = \"https://self.example\"\n");

        assert_eq!(
            api_url(&globals(None, Some(path.clone()))).unwrap(),
            "https://self.example"
        );
        assert_eq!(
            api_url(&globals(Some("https://flag.example"), Some(path))).unwrap(),
            "https://flag.example"
        );
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
