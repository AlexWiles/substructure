// Per-command setup: load the credentials file and (optionally) the
// project-local `subs.toml`, build the HTTP client with the bearer token,
// and provide org/app resolution. Every resource command starts by
// calling `Context::load`.

use anyhow::{bail, Result};

use super::config;
use super::http::CloudClient;
use super::project_config::{self, ProjectConfig};
use super::{AppScope, CloudGlobals, OrgScope};

pub struct Context {
    /// Project-local `subs.toml`, either pointed at explicitly with
    /// `-c/--config` or discovered by walking up from cwd. Holds the
    /// org/app pointers for this directory tree.
    pub project: Option<ProjectConfig>,
    pub client: CloudClient,
}

impl Context {
    pub fn load(globals: &CloudGlobals) -> Result<Self> {
        let credentials_path = config::resolve_path(globals.credentials.clone())?;
        let cfg = config::load(&credentials_path)?;
        let api_url = cfg.resolve_api_url(globals.url.as_deref());
        let token = cfg.require_token()?.to_string();
        let client = CloudClient::new(api_url, Some(token));
        let project = match globals.config.as_deref() {
            Some(p) => Some(project_config::load_explicit(p)?.config),
            None => project_config::find()?.map(|f| f.config),
        };
        Ok(Self { project, client })
    }

    /// Load + resolve an org id from an `OrgScope` in one call.
    pub fn from_org(scope: &OrgScope) -> Result<(Self, String)> {
        let ctx = Self::load(&scope.globals)?;
        let org = ctx.require_org(scope.org.as_deref())?;
        Ok((ctx, org))
    }

    /// Load + resolve org and app ids from an `AppScope` in one call.
    pub fn from_app(scope: &AppScope) -> Result<(Self, String, String)> {
        let ctx = Self::load(&scope.globals)?;
        let org = ctx.require_org(scope.org.as_deref())?;
        let app = ctx.require_app(&org, scope.app.as_deref())?;
        Ok((ctx, org, app))
    }

    /// Resolved org id: flag > project subs.toml. There is no user-level
    /// default. To pin an org for a directory tree, run `subs cloud init`.
    pub fn require_org(&self, flag: Option<&str>) -> Result<String> {
        if let Some(s) = flag {
            return Ok(s.to_string());
        }
        if let Some(org) = self.project.as_ref().and_then(|p| p.org.clone()) {
            return Ok(org);
        }
        bail!(
            "no org selected. Pass --org <id>, or run `subs cloud init` to pin one for this directory."
        )
    }

    /// Resolved app id: flag > project subs.toml.
    pub fn require_app(&self, _org_id: &str, flag: Option<&str>) -> Result<String> {
        if let Some(s) = flag {
            return Ok(s.to_string());
        }
        if let Some(app) = self.project.as_ref().and_then(|p| p.app.clone()) {
            return Ok(app);
        }
        bail!(
            "no app selected. Pass --app <id>, or run `subs cloud init` to pin one for this directory."
        )
    }
}
