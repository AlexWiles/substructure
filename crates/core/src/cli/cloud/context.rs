// Per-command setup: load the config from the resolved path, build the
// HTTP client with the bearer token, and provide org/app resolution
// helpers. Every resource command starts by calling `Context::load`.

use std::path::PathBuf;

use anyhow::{bail, Result};

use super::config::{self, Config};
use super::http::CloudClient;
use super::CloudGlobals;

pub struct Context {
    pub config_path: PathBuf,
    pub config: Config,
    pub client: CloudClient,
}

impl Context {
    pub fn load(globals: &CloudGlobals) -> Result<Self> {
        let config_path = config::resolve_path(globals.config.clone())?;
        let config = config::load(&config_path)?;
        let api_url = config.resolve_api_url(globals.url.as_deref());
        let token = config.require_token()?.to_string();
        let client = CloudClient::new(api_url, Some(token));
        Ok(Self {
            config_path,
            config,
            client,
        })
    }

    pub fn save(&self) -> Result<()> {
        config::save(&self.config_path, &self.config)
    }

    pub fn require_org(&self, flag: Option<&str>) -> Result<String> {
        self.config.resolve_org(flag).ok_or_else(|| {
            anyhow::anyhow!(
                "no org selected — pass --org <id> or run `subs cloud orgs use <id>` to set a default"
            )
        })
    }

    pub fn require_app(&self, org_id: &str, flag: Option<&str>) -> Result<String> {
        match self.config.resolve_app(org_id, flag) {
            Some(id) => Ok(id),
            None => bail!(
                "no app selected for org {org_id} — pass --app <id> or run `subs cloud apps use <id>`"
            ),
        }
    }
}
