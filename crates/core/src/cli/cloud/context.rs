use anyhow::{bail, Result};
use serde::Deserialize;

use super::credentials;
use super::http::CloudClient;
use super::pickers;
use super::print;
use super::project_config::{self, ProjectConfig};
use super::{AppScope, CloudGlobals, OrgScope};

#[derive(Debug, Deserialize)]
struct AppBalance {
    name: String,
    #[serde(default)]
    balance_usd: Option<String>,
}

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
        // Precedence: --url flag > project subs.toml url > $SUBS_API_URL > default.
        let url_override = globals
            .url
            .as_deref()
            .or_else(|| project.as_ref().and_then(|p| p.url.as_deref()));
        let api_url = credentials::resolve_api_url(url_override);
        let token = creds.require_token()?.to_string();
        let client = CloudClient::new(api_url, Some(token));
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
            .get::<AppBalance>(&format!("/api/v1/apps/{app_id}"))
            .await
        else {
            return;
        };
        let raw = a.balance_usd.as_deref().unwrap_or("0");
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
        if pickers::interactive(&self.globals) {
            return pickers::pick_org(self).await;
        }
        bail!("no org selected. Pass --org <id>.")
    }
}
