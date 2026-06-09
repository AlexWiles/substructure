use anyhow::{bail, Result};

use crate::api::v1::App;

use super::credentials;
use super::http::CloudClient;
use super::pickers;
use super::print;
use super::project_config::{self, ProjectConfig};
use super::{AppScope, CloudGlobals, OrgScope};

pub struct Context {
    pub project: Option<ProjectConfig>,
    pub client: CloudClient,
    pub globals: CloudGlobals,
}

impl Context {
    pub fn load(globals: &CloudGlobals) -> Result<Self> {
        let credentials_path = credentials::resolve_path(globals.credentials.clone())?;
        let creds = credentials::load(&credentials_path)?;
        let project = match globals.config.as_deref() {
            Some(p) => Some(project_config::load_explicit(p)?.config),
            None => project_config::find()?.map(|f| f.config),
        };
        // Precedence: --url flag > project substructure.toml url > $SUBS_API_URL > default.
        let url_override = globals
            .url
            .as_deref()
            .or_else(|| project.as_ref().and_then(|p| p.url.as_deref()));
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
        if let Some(app) = scope
            .app
            .clone()
            .or_else(|| ctx.project.as_ref().and_then(|p| p.app.clone()))
        {
            ctx.maybe_warn_zero_balance(&app).await;
            return Ok((ctx, app));
        }

        // A single-tenant server (e.g. a local server) advertises its org/app
        // in response headers; adopt the app and skip the picker entirely.
        if let Some(app) = ctx.server_default_app().await {
            ctx.maybe_warn_zero_balance(&app).await;
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
        ctx.maybe_warn_zero_balance(&app).await;
        Ok((ctx, app))
    }

    async fn maybe_warn_zero_balance(&self, app_id: &str) {
        let Ok(a) = self
            .client
            .get::<App>(&format!("/api/v1/apps/{app_id}"))
            .await
        else {
            return;
        };
        // A missing balance is not a zero balance: local servers omit it.
        let Some(raw) = a.balance_usd.as_deref() else {
            return;
        };
        if print::is_zero_usd(raw) {
            print::warn_zero_balance(&a.name, self.client.base_url(), app_id);
        }
    }

    pub async fn require_org(&self, flag: Option<&str>) -> Result<String> {
        if let Some(s) = flag {
            return Ok(s.to_string());
        }
        if let Some(org) = self.project.as_ref().and_then(|p| p.org.clone()) {
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

    // Probe the server once for an advertised default org/app. Single-tenant
    // servers set them as response headers; the cloud sends nothing, leaving
    // the picker as the fallback.
    async fn probe_server_defaults(&self) {
        if self.client.needs_default_probe() {
            let _ = self.client.get::<serde_json::Value>("/api/v1/orgs").await;
        }
    }
}
