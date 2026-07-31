use anyhow::Result;

use super::context;
use super::credentials;
use super::CloudGlobals;

pub async fn run(globals: &CloudGlobals) -> Result<()> {
    let path = credentials::resolve_path(globals.credentials.clone())?;
    let mut creds = credentials::load(&path)?;
    let api_url = context::api_url(globals)?;
    creds.clear_token(&api_url);
    credentials::save(&path, &creds)?;
    println!(
        "Logged out of {api_url}. Token cleared from {}",
        path.display()
    );
    Ok(())
}
