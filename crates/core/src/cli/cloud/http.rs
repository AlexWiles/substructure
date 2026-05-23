// Thin HTTP client wrapper for the cloud API. Adds the bearer token, maps
// status codes to typed errors, and parses JSON bodies. Kept deliberately
// small: no retry, no pagination, no codegen.

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
        let res = self
            .request(Method::GET, path)
            .send()
            .await?;
        decode(res).await
    }

    /// POST <path> with JSON body → T.
    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .request(Method::POST, path)
            .json(body)
            .send()
            .await?;
        decode(res).await
    }

    /// PATCH <path> with JSON body → T.
    pub async fn patch_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .request(Method::PATCH, path)
            .json(body)
            .send()
            .await?;
        decode(res).await
    }

    /// PUT <path> with JSON body → T.
    pub async fn put_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .request(Method::PUT, path)
            .json(body)
            .send()
            .await?;
        decode(res).await
    }

    /// DELETE <path>, discarding any response body. For endpoints that return
    /// 204 or an envelope the caller doesn't care about.
    pub async fn delete_discard(&self, path: &str) -> Result<()> {
        let res = self.request(Method::DELETE, path).send().await?;
        check_status(res).await?;
        Ok(())
    }

    /// POST <path> with no body → T. For action endpoints with no payload.
    pub async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self
            .request(Method::POST, path)
            .header(header::CONTENT_LENGTH, "0")
            .send()
            .await?;
        decode(res).await
    }

    /// POST <path> with a JSON body → raw Response so the caller can branch
    /// on status (used by the OAuth device-flow polling, where 400 + an
    /// `authorization_pending` body means "keep polling", not "fail").
    /// Better Auth's device endpoints accept JSON only: sending
    /// form-encoded bodies yields `UNSUPPORTED_MEDIA_TYPE`.
    pub async fn post_json_raw<B: Serialize>(&self, path: &str, body: &B) -> Result<Response> {
        Ok(self.request(Method::POST, path).json(body).send().await?)
    }
}

async fn decode<T: DeserializeOwned>(res: Response) -> Result<T> {
    let res = check_status(res).await?;
    res.json::<T>().await.context("decoding response JSON")
}

/// Return `res` unchanged on success, otherwise format the server's error
/// envelope (or fall back to the raw body) into an `anyhow` error.
async fn check_status(res: Response) -> Result<Response> {
    let status = res.status();
    if status.is_success() {
        return Ok(res);
    }

    let body_text = res.text().await.unwrap_or_default();
    let parsed: Option<ErrorBody> = serde_json::from_str(&body_text).ok();
    let (code, message) = parsed
        .and_then(|b| b.error)
        .map(|e| (e.code, e.message))
        .unwrap_or((None, None));

    let raw_msg = message.unwrap_or_else(|| body_text.lines().next().unwrap_or("").to_string());
    let msg = raw_msg.trim_end_matches('.');
    let prefix = match code.as_deref() {
        Some(c) if !c.is_empty() => format!("HTTP {} {}", status.as_u16(), c),
        _ => format!("HTTP {}", status.as_u16()),
    };
    let suffix = match status {
        StatusCode::UNAUTHORIZED => " Run `subs cloud login` to authenticate.",
        StatusCode::FORBIDDEN => " Your account does not have access to this resource.",
        _ => "",
    };

    bail!("{prefix}: {msg}.{suffix}")
}
