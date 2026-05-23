use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
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

    pub fn base_url(&self) -> &str {
        &self.base_url
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

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self.request(Method::GET, path).send().await?;
        decode(res).await
    }

    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self.request(Method::POST, path).json(body).send().await?;
        decode(res).await
    }

    pub async fn patch_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self.request(Method::PATCH, path).json(body).send().await?;
        decode(res).await
    }

    pub async fn put_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self.request(Method::PUT, path).json(body).send().await?;
        decode(res).await
    }

    pub async fn delete_discard(&self, path: &str) -> Result<()> {
        let res = self.request(Method::DELETE, path).send().await?;
        check_status(res).await?;
        Ok(())
    }

    pub async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self
            .request(Method::POST, path)
            .header(header::CONTENT_LENGTH, "0")
            .send()
            .await?;
        decode(res).await
    }

    // Raw Response: OAuth device-flow polling treats 400 `authorization_pending` as "keep polling".
    // Better Auth's device endpoints require JSON (form-encoded yields UNSUPPORTED_MEDIA_TYPE).
    pub async fn post_json_raw<B: Serialize>(&self, path: &str, body: &B) -> Result<Response> {
        Ok(self.request(Method::POST, path).json(body).send().await?)
    }

    /// Open an SSE stream and invoke `on_line` for each non-empty line
    /// (caller decides which lines to keep — e.g. `data:` payloads).
    /// Runs until the server closes the stream or the caller hits Ctrl-C.
    pub async fn stream_sse<F>(&self, path: &str, mut on_line: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        let res = self
            .request(Method::GET, path)
            .header(header::ACCEPT, "text/event-stream")
            .send()
            .await?;
        let res = check_status(res).await?;

        let mut stream = res.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk.context("reading SSE chunk")?);
            // Lines are terminated by \n; \r is tolerated and stripped.
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let s = std::str::from_utf8(&line[..line.len() - 1])
                    .context("non-UTF8 in SSE stream")?
                    .trim_end_matches('\r');
                on_line(s);
            }
        }
        Ok(())
    }
}

async fn decode<T: DeserializeOwned>(res: Response) -> Result<T> {
    let res = check_status(res).await?;
    res.json::<T>().await.context("decoding response JSON")
}

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
