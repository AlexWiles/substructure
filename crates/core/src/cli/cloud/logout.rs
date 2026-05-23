// `subs cloud logout` — clears the local token. Local-only by design: the
// Better Auth `deviceAuthorization` plugin doesn't document a server-side
// revocation API for device-flow sessions (see
// better-auth.com/docs/plugins/device-authorization). If you need to kill
// the session before it expires naturally (7 days idle), revoke it from
// the web UI's active-sessions list.

use std::path::PathBuf;

use anyhow::Result;

use super::config;

pub async fn run(_url_flag: Option<String>, config_path: Option<PathBuf>) -> Result<()> {
    let path = config::resolve_path(config_path)?;
    let mut cfg = config::load(&path)?;

    cfg.token = None;
    // Drop default_org and per-org defaults — they belong to the previous user.
    cfg.default_org = None;
    cfg.orgs.clear();

    config::save(&path, &cfg)?;
    println!("Logged out. Token cleared from {}", path.display());
    println!("Note: the server-side session remains valid until idle expiry.");
    Ok(())
}
