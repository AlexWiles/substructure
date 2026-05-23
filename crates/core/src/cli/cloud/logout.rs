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
