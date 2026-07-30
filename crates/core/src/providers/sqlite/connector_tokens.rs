use async_trait::async_trait;
use rusqlite::OptionalExtension;

use crate::connectors::oauth::{TokenStore, Tokens};
use crate::event_store::StoreError;

use super::SqliteDb;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS connector_tokens (
    tenant_id   TEXT NOT NULL,
    url         TEXT NOT NULL,
    tokens      TEXT NOT NULL,
    PRIMARY KEY (tenant_id, url)
);
";

/// Authorized connections, in the engine's own database.
///
/// The credential lives beside the sessions that use it: `subs mcp login`
/// writes here and the engine reads here, so one environment is one file and
/// one set of logins. Refresh rotates the credential, so the store has to be
/// writable — a container's config directory usually is not.
pub struct SqliteTokenStore {
    db: SqliteDb,
}

impl SqliteTokenStore {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        db.run_schema(SCHEMA)?;
        Ok(Self { db })
    }

    /// Forget a credential. `false` when there was none to forget.
    pub async fn delete(&self, tenant_id: &str, url: &str) -> Result<bool, StoreError> {
        let (tenant_id, url) = (tenant_id.to_string(), url.to_string());
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let removed = conn
                .execute(
                    "DELETE FROM connector_tokens WHERE tenant_id = ?1 AND url = ?2",
                    rusqlite::params![tenant_id, url],
                )
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok(removed > 0)
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))?
    }
}

#[async_trait]
impl TokenStore for SqliteTokenStore {
    async fn get(&self, tenant_id: &str, url: &str) -> Option<Tokens> {
        let (tenant_id, url) = (tenant_id.to_string(), url.to_string());
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().ok()?;
            let stored: String = conn
                .query_row(
                    "SELECT tokens FROM connector_tokens WHERE tenant_id = ?1 AND url = ?2",
                    rusqlite::params![tenant_id, url],
                    |row| row.get(0),
                )
                .optional()
                .ok()??;
            serde_json::from_str(&stored).ok()
        })
        .await
        .ok()?
    }

    async fn put(&self, tenant_id: &str, url: &str, tokens: Tokens) -> Result<(), String> {
        let (tenant_id, url) = (tenant_id.to_string(), url.to_string());
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let encoded = serde_json::to_string(&tokens).map_err(|e| e.to_string())?;
            let conn = writer.lock().map_err(|e| e.to_string())?;
            conn.execute(
                "INSERT INTO connector_tokens (tenant_id, url, tokens)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(tenant_id, url) DO UPDATE SET tokens = excluded.tokens",
                rusqlite::params![tenant_id, url, encoded],
            )
            .map_err(|e| e.to_string())?;
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::oauth::ClientId;
    use std::time::Duration;
    use uuid::Uuid;

    /// A file per test: `:memory:` opens a shared-cache database, so in-memory
    /// stores would see each other's rows.
    fn temp_store() -> (SqliteTokenStore, std::path::PathBuf) {
        let path =
            std::env::temp_dir().join(format!("core-connector-tokens-{}.db", Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        (SqliteTokenStore::new(db).unwrap(), path)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    fn tokens(access: &str) -> Tokens {
        Tokens {
            access_token: access.into(),
            refresh_token: Some("r".into()),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            scope: Some("read".into()),
            issuer: "https://mcp.linear.app".into(),
            token_endpoint: "https://mcp.linear.app/token".into(),
            resource: "https://mcp.linear.app/mcp".into(),
            client: ClientId::Registered {
                client_id: "c".into(),
                client_secret: None,
            },
        }
    }

    #[tokio::test]
    async fn a_credential_round_trips_and_a_refresh_replaces_it() {
        let (store, path) = temp_store();
        let url = "https://mcp.linear.app/mcp";

        assert!(store.get("default", url).await.is_none());

        let first = tokens("access-1");
        store.put("default", url, first.clone()).await.unwrap();
        assert_eq!(store.get("default", url).await.unwrap(), first);

        // Rotation overwrites in place rather than accumulating rows.
        let second = tokens("access-2");
        store.put("default", url, second.clone()).await.unwrap();
        assert_eq!(store.get("default", url).await.unwrap(), second);
        cleanup(&path);
    }

    #[tokio::test]
    async fn tenants_and_connections_do_not_leak_into_each_other() {
        let (store, path) = temp_store();

        store
            .put("a", "https://mcp.linear.app/mcp", tokens("a-linear"))
            .await
            .unwrap();
        store
            .put("b", "https://mcp.linear.app/mcp", tokens("b-linear"))
            .await
            .unwrap();
        store
            .put("a", "https://mcp.sentry.dev/mcp", tokens("a-sentry"))
            .await
            .unwrap();

        assert_eq!(
            store
                .get("a", "https://mcp.linear.app/mcp")
                .await
                .unwrap()
                .access_token,
            "a-linear"
        );
        assert_eq!(
            store
                .get("b", "https://mcp.linear.app/mcp")
                .await
                .unwrap()
                .access_token,
            "b-linear"
        );
        assert!(store.get("b", "https://mcp.sentry.dev/mcp").await.is_none());
        cleanup(&path);
    }
}
