use std::path::PathBuf;

use anyhow::Result;

use super::credentials;

pub async fn run(url_flag: Option<String>, credentials_path: Option<PathBuf>) -> Result<()> {
    let path = credentials::resolve_path(credentials_path)?;
    let mut creds = credentials::load(&path)?;
    let api_url = credentials::resolve_api_url(url_flag.as_deref());
    creds.clear_token(&api_url);
    credentials::save(&path, &creds)?;
    println!(
        "Logged out of {api_url}. Token cleared from {}",
        path.display()
    );
    Ok(())
}
