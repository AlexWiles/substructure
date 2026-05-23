// `subs cloud logout` — clears the local token and best-effort tells the
// server to invalidate the session. We don't fail logout on server errors;
// the local credentials are gone either way.

use std::path::PathBuf;

use anyhow::Result;

use super::config;
use super::http::CloudClient;

pub async fn run(url_flag: Option<String>, config_path: Option<PathBuf>) -> Result<()> {
    let path = config::resolve_path(config_path)?;
    let mut cfg = config::load(&path)?;

    if let Some(token) = cfg.token.take() {
        let api_url = cfg.resolve_api_url(url_flag.as_deref());
        let client = CloudClient::new(api_url, Some(token));
        // Best-effort: invalidate server-side. Better Auth's sign-out endpoint
        // accepts the bearer token and revokes the session. Ignore failures —
        // we still want to clear local state.
        let _ = client.post_json::<serde_json::Value, serde_json::Value>("/api/auth/sign-out", &serde_json::json!({})).await;
    }

    // Also drop default_org/default_app — they belong to the previous user.
    cfg.default_org = None;
    cfg.orgs.clear();

    config::save(&path, &cfg)?;
    println!("Logged out. Token cleared from {}", path.display());
    Ok(())
}
