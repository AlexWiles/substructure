use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::Sha256;
use subtle::ConstantTimeEq;

use super::bot::SlackBot;

/// An older timestamp is a replay.
const MAX_SKEW_SECS: i64 = 300;

#[derive(Clone)]
struct WebhookState {
    bot: Arc<SlackBot>,
    signing_secret: Arc<str>,
}

/// Events API routes: `POST {mount}/events` and `POST {mount}/interactions`.
/// Each request is signature-verified. The response is held until the submit
/// is recorded; Slack sends the event again after 3 s and the turn id
/// dedupes it. Call [`SlackBot::start`] first.
pub fn webhook_router(bot: Arc<SlackBot>, signing_secret: String) -> Router {
    Router::new()
        .route("/events", post(events))
        .route("/interactions", post(interactions))
        .with_state(WebhookState {
            bot,
            signing_secret: signing_secret.into(),
        })
}

async fn events(State(state): State<WebhookState>, headers: HeaderMap, body: Bytes) -> Response {
    if !verified(
        &state.signing_secret,
        &headers,
        &body,
        chrono::Utc::now().timestamp(),
    ) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Ok(payload) = serde_json::from_slice::<Value>(&body) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if payload["type"].as_str() == Some("url_verification") {
        return Json(serde_json::json!({ "challenge": payload["challenge"] })).into_response();
    }
    state.bot.handle_event(&payload).await;
    StatusCode::OK.into_response()
}

async fn interactions(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verified(
        &state.signing_secret,
        &headers,
        &body,
        chrono::Utc::now().timestamp(),
    ) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    // Form-encoded; the payload is a JSON field.
    let Some(payload) = serde_urlencoded::from_bytes::<HashMap<String, String>>(&body)
        .ok()
        .and_then(|form| serde_json::from_str::<Value>(form.get("payload")?).ok())
    else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    state.bot.handle_interaction(&payload).await;
    StatusCode::OK.into_response()
}

/// HMAC-SHA256 over `v0:{timestamp}:{raw body}`. Verify the bytes before
/// you parse them.
fn verified(secret: &str, headers: &HeaderMap, body: &[u8], now: i64) -> bool {
    let header = |name| headers.get(name).and_then(|v| v.to_str().ok());
    let (Some(timestamp), Some(signature)) = (
        header("x-slack-request-timestamp"),
        header("x-slack-signature"),
    ) else {
        return false;
    };
    let Ok(ts) = timestamp.parse::<i64>() else {
        return false;
    };
    if (now - ts).abs() > MAX_SKEW_SECS {
        return false;
    }
    let Ok(mut mac) = <Hmac<Sha256>>::new_from_slice(secret.as_bytes()) else {
        return false;
    };
    mac.update(format!("v0:{timestamp}:").as_bytes());
    mac.update(body);
    let expected = format!("v0={}", hex::encode(mac.finalize().into_bytes()));
    expected.as_bytes().ct_eq(signature.as_bytes()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sign(secret: &str, timestamp: &str, body: &[u8]) -> String {
        let mut mac = <Hmac<Sha256>>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(format!("v0:{timestamp}:").as_bytes());
        mac.update(body);
        format!("v0={}", hex::encode(mac.finalize().into_bytes()))
    }

    fn signed_headers(timestamp: &str, signature: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("x-slack-request-timestamp", timestamp.parse().unwrap());
        headers.insert("x-slack-signature", signature.parse().unwrap());
        headers
    }

    #[test]
    fn signature_verifies_and_rejects_tampering() {
        let body = br#"{"type":"event_callback"}"#;
        let headers = signed_headers("1000", &sign("s3cret", "1000", body));
        assert!(verified("s3cret", &headers, body, 1000));
        assert!(verified("s3cret", &headers, body, 1000 + MAX_SKEW_SECS));

        assert!(!verified("s3cret", &headers, b"tampered", 1000));
        assert!(!verified("wrong", &headers, body, 1000));
        // Stale timestamp.
        assert!(!verified(
            "s3cret",
            &headers,
            body,
            1000 + MAX_SKEW_SECS + 1
        ));
    }

    #[test]
    fn missing_or_malformed_headers_fail_closed() {
        let body = b"x";
        assert!(!verified("s", &HeaderMap::new(), body, 0));
        let only_ts = signed_headers("1000", &sign("s", "1000", body));
        let mut no_sig = only_ts.clone();
        no_sig.remove("x-slack-signature");
        assert!(!verified("s", &no_sig, body, 1000));
        let bad_ts = signed_headers("not-a-number", "v0=aa");
        assert!(!verified("s", &bad_ts, body, 1000));
        // A wrong-length signature does not panic.
        let short = signed_headers("1000", "v0=abc");
        assert!(!verified("s", &short, body, 1000));
    }
}
