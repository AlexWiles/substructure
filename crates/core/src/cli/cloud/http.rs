// Thin HTTP client wrapper for the cloud API. Adds the bearer token, maps
// status codes to typed errors, and parses JSON bodies. Kept deliberately
// small — no retry, no pagination, no codegen.

use anyhow::{bail, Context, Result};
use reqwest::{header, Method, Response, StatusCode};
use serde::{de::DeserializeOwned, Serialize};

#[derive(Debug)]
pub struct CloudClient {
    base_url: String,
    token: Option<String>,
    http: reqwest::Client,
}

#[derive(Debug, serde::Deserialize)]
struct ErrorBody {
    error: Option<ErrorPayload>,
}

#[derive(Debug, serde::Deserialize)]
struct ErrorPayload {
    code: Option<String>,
    message: Option<String>,
}

impl CloudClient {
    pub fn new(base_url: impl Into<String>, token: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(concat!("subs/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client");
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token,
            http,
        }
    }

    fn url(&self, path: &str) -> String {
        if path.starts_with("http://") || path.starts_with("https://") {
            path.to_string()
        } else {
            format!("{}{}", self.base_url, path)
        }
    }

    fn request(&self, method: Method, path: &str) -> reqwest::RequestBuilder {
        let mut req = self.http.request(method, self.url(path));
        if let Some(t) = &self.token {
            req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        req
    }

    /// GET <path> → T. Errors with the server's error envelope when present.
    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self.request(Method::GET, path).send().await.context("HTTP send")?;
        decode(res).await
    }

    /// POST <path> with JSON body → T.
    pub async fn post_json<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let res = self
            .request(Method::POST, path)
            .json(body)
            .send()
            .await
            .context("HTTP send")?;
        decode(res).await
    }

    /// PATCH <path> with JSON body → T.
    pub async fn patch_json<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let res = self
            .request(Method::PATCH, path)
            .json(body)
            .send()
            .await
            .context("HTTP send")?;
        decode(res).await
    }

    /// PUT <path> with JSON body → T.
    pub async fn put_json<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let res = self
            .request(Method::PUT, path)
            .json(body)
            .send()
            .await
            .context("HTTP send")?;
        decode(res).await
    }

    /// DELETE <path> → T.
    pub async fn delete<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self.request(Method::DELETE, path).send().await.context("HTTP send")?;
        decode(res).await
    }

    /// POST <path> as application/x-www-form-urlencoded → raw Response so the
    /// caller can branch on status (used by the OAuth device-flow polling).
    pub async fn post_form_raw(&self, path: &str, form: &[(&str, &str)]) -> Result<Response> {
        let body = serde_urlencoded::to_string(form).context("encoding form body")?;
        self.request(Method::POST, path)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .context("HTTP send")
    }
}

async fn decode<T: DeserializeOwned>(res: Response) -> Result<T> {
    let status = res.status();
    if status.is_success() {
        return res.json::<T>().await.context("decoding response JSON");
    }

    let body_text = res.text().await.unwrap_or_default();
    let parsed: Option<ErrorBody> = serde_json::from_str(&body_text).ok();
    let (code, message) = parsed
        .and_then(|b| b.error)
        .map(|e| (e.code, e.message))
        .unwrap_or((None, None));

    let hint = match status {
        StatusCode::UNAUTHORIZED => " — run `subs cloud login`",
        StatusCode::FORBIDDEN => " — your account doesn't have access",
        StatusCode::NOT_FOUND => "",
        _ => "",
    };

    let msg = message.unwrap_or_else(|| body_text.lines().next().unwrap_or("").to_string());
    let code_label = code.as_deref().unwrap_or("");

    if code_label.is_empty() {
        bail!("HTTP {}{}: {}", status.as_u16(), hint, msg);
    } else {
        bail!("HTTP {} {}{}: {}", status.as_u16(), code_label, hint, msg);
    }
}
