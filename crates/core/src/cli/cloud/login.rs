// OAuth 2.0 Device Authorization Grant (RFC 8628).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tokio::time::sleep;

use super::credentials;
use super::http::CloudClient;

const CLIENT_ID: &str = "subs-cli";
const MAX_POLL_WINDOW: Duration = Duration::from_secs(15 * 60); // matches deviceAuthorization expiresIn

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    expires_in: u64,
    interval: u64,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
}

#[derive(Debug, Deserialize)]
struct OauthError {
    error: String,
    #[serde(default)]
    error_description: Option<String>,
}

pub async fn run(
    url_flag: Option<String>,
    credentials_path: Option<PathBuf>,
    no_browser: bool,
) -> Result<()> {
    let path = credentials::resolve_path(credentials_path)?;
    let mut creds = credentials::load(&path)?;
    let api_url = credentials::resolve_api_url(url_flag.as_deref());

    let client = CloudClient::new(&api_url, None);

    let res = client
        .post_json_raw(
            "/api/auth/device/code",
            &serde_json::json!({
                "client_id": CLIENT_ID,
                "scope": "openid profile email",
            }),
        )
        .await?;
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        let snippet = body
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(200)
            .collect::<String>();
        bail!(
            "could not request device code (HTTP {}): {snippet}",
            status.as_u16()
        );
    }
    let device: DeviceCodeResponse = res.json().await.context("decoding device-code response")?;

    let open_url = device
        .verification_uri_complete
        .as_deref()
        .unwrap_or(&device.verification_uri);
    println!();
    println!("To authenticate, visit:");
    if let Some(complete) = device.verification_uri_complete.as_deref() {
        println!("  {}", complete);
        println!();
        println!(
            "Or open {} and enter code: {}",
            device.verification_uri, device.user_code
        );
    } else {
        println!("  {}", device.verification_uri);
        println!("Enter code: {}", device.user_code);
    }
    if !no_browser && webbrowser::open(open_url).is_ok() {
        println!();
        println!("Opened in your browser.");
    }
    println!();
    println!("Waiting for approval…");

    let mut interval = Duration::from_secs(device.interval.max(1));
    let started = std::time::Instant::now();
    let total_window = Duration::from_secs(device.expires_in).min(MAX_POLL_WINDOW);

    let token = loop {
        if started.elapsed() > total_window {
            bail!("login expired before approval. Run `subs login` to start over.");
        }
        sleep(interval).await;

        let res = client
            .post_json_raw(
                "/api/auth/device/token",
                &serde_json::json!({
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device.device_code,
                    "client_id": CLIENT_ID,
                }),
            )
            .await?;

        if res.status().is_success() {
            let tok: TokenResponse = res.json().await.context("decoding token response")?;
            break tok.access_token;
        }

        let err: OauthError = res
            .json()
            .await
            .context("decoding OAuth error body during polling")?;
        match err.error.as_str() {
            "authorization_pending" => continue,
            "slow_down" => {
                interval += Duration::from_secs(5);
                continue;
            }
            "access_denied" => bail!("login denied by user"),
            "expired_token" => {
                bail!("login code expired. Run `subs login` to start over.")
            }
            other => bail!(
                "OAuth error `{other}`: {}",
                err.error_description
                    .as_deref()
                    .unwrap_or("(no description)")
            ),
        }
    };

    creds.set_token(&api_url, token);
    credentials::save(&path, &creds)?;

    println!("Logged in to {api_url}. Token saved to {}", path.display());
    Ok(())
}
