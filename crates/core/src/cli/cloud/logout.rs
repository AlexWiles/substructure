// `subs cloud logout`: clears the bearer token from the credentials
// file. Local-only by design: the Better Auth `deviceAuthorization`
// plugin doesn't document a server-side revocation API for device-flow
// sessions. To kill the session before it expires (7 days idle), revoke
// it from the web UI's active-sessions list.

use std::path::PathBuf;

use anyhow::Result;

use super::config;

pub async fn run(_url_flag: Option<String>, credentials_path: Option<PathBuf>) -> Result<()> {
    let path = config::resolve_path(credentials_path)?;
    let mut cfg = config::load(&path)?;
    cfg.token = None;
    config::save(&path, &cfg)?;
    println!("Logged out. Token cleared from {}", path.display());
    println!("Note: the server-side session remains valid until idle expiry.");
    Ok(())
}
