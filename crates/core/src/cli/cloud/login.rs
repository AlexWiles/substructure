// OAuth 2.0 Device Authorization Grant (RFC 8628).

use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tokio::time::sleep;

use super::context;
use super::credentials;
use super::http::CloudClient;
use super::pickers;
use super::CloudGlobals;

const CLIENT_ID: &str = "subs-cli";
const MAX_POLL_WINDOW: Duration = Duration::from_secs(15 * 60); // matches deviceAuthorization expiresIn

/// Log in because `api_url` refused what the command sent it, and answer with
/// the credential it issued. Missing and stale look the same from here — only
/// the deployment knows an expired or revoked token, and either way the reader
/// gets the login the 401 would only have named.
///
/// None when there is nobody to run it: a script gets the error it can act on,
/// `--json` keeps its output machine-readable, and $SUBS_API_TOKEN is a
/// credential its operator chose, which a login would not replace.
pub async fn refresh(globals: &CloudGlobals, api_url: &str) -> Result<Option<String>> {
    if globals.json || !pickers::interactive(globals) {
        return Ok(None);
    }
    if std::env::var(credentials::TOKEN_ENV).is_ok_and(|t| !t.is_empty()) {
        return Ok(None);
    }
    let path = credentials::resolve_path(globals.credentials.clone())?;
    println!();
    match credentials::load(&path)?.token_for(api_url).is_some() {
        true => println!("The credential for {api_url} is no longer accepted."),
        false => println!("Not logged in to {api_url}."),
    }
    run(globals, false).await?;
    Ok(credentials::load(&path)?
        .token_for(api_url)
        .map(str::to_string))
}

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

pub async fn run(globals: &CloudGlobals, no_browser: bool) -> Result<()> {
    let path = credentials::resolve_path(globals.credentials.clone())?;
    let mut creds = credentials::load(&path)?;
    let api_url = context::api_url(globals)?;

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

        // A body that is not an OAuth error came from something other than the
        // server — a proxy, or a deployment that went away mid-flow.
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        let err: OauthError = serde_json::from_str(&body).with_context(|| {
            format!(
                "HTTP {} from {api_url} while waiting for approval: {}",
                status.as_u16(),
                body.lines()
                    .next()
                    .unwrap_or("no body")
                    .chars()
                    .take(80)
                    .collect::<String>(),
            )
        })?;
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
