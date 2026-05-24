use std::path::PathBuf;

use anyhow::Result;

use super::credentials;

pub async fn run(_url_flag: Option<String>, credentials_path: Option<PathBuf>) -> Result<()> {
    let path = credentials::resolve_path(credentials_path)?;
    let mut creds = credentials::load(&path)?;
    creds.token = None;
    credentials::save(&path, &creds)?;
    println!("Logged out. Token cleared from {}", path.display());
    Ok(())
}
