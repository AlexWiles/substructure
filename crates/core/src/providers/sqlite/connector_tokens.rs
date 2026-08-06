use async_trait::async_trait;
use rusqlite::OptionalExtension;

use crate::connectors::oauth::{TokenStore, Tokens};
use crate::event_store::StoreError;

use super::SqliteDb;

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
        Ok(Self { db })
    }

    /// Forget the credentials for connections `declared` does not name,
    /// answering with the ids forgotten.
    ///
    /// The file is the whole declaration, so a `[mcp.<id>]` taken out of it is
    /// a connection that was removed, and its credential goes with it. Called
    /// as the engine starts, which is when the file is applied.
    pub async fn retain(
        &self,
        tenant_id: &str,
        declared: &[String],
    ) -> Result<Vec<String>, StoreError> {
        let stored = self.ids(tenant_id).await?;
        let mut forgotten = Vec::new();
        for id in stored {
            if declared.iter().any(|d| d == &id) {
                continue;
            }
            self.delete(tenant_id, &id).await?;
            forgotten.push(id);
        }
        Ok(forgotten)
    }

    /// Every connection this tenant holds a credential for.
    async fn ids(&self, tenant_id: &str) -> Result<Vec<String>, StoreError> {
        let tenant_id = tenant_id.to_string();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader
                .open()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let mut stmt = conn
                .prepare("SELECT connection_id FROM connector_credentials WHERE tenant_id = ?1")
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let rows = stmt
                .query_map(rusqlite::params![tenant_id], |row| row.get(0))
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            rows.collect::<rusqlite::Result<Vec<String>>>()
                .map_err(|e| StoreError::Internal(e.to_string()))
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))?
    }

    /// Forget a credential. `false` when there was none to forget.
    pub async fn delete(&self, tenant_id: &str, connection_id: &str) -> Result<bool, StoreError> {
        let (tenant_id, connection_id) = (tenant_id.to_string(), connection_id.to_string());
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let removed = conn
                .execute(
                    "DELETE FROM connector_credentials
                     WHERE tenant_id = ?1 AND connection_id = ?2",
                    rusqlite::params![tenant_id, connection_id],
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
    async fn get(&self, tenant_id: &str, connection_id: &str) -> Option<Tokens> {
        let (tenant_id, connection_id) = (tenant_id.to_string(), connection_id.to_string());
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().ok()?;
            let stored: String = conn
                .query_row(
                    "SELECT tokens FROM connector_credentials
                     WHERE tenant_id = ?1 AND connection_id = ?2",
                    rusqlite::params![tenant_id, connection_id],
                    |row| row.get(0),
                )
                .optional()
                .ok()??;
            serde_json::from_str(&stored).ok()
        })
        .await
        .ok()?
    }

    async fn put(
        &self,
        tenant_id: &str,
        connection_id: &str,
        tokens: Tokens,
    ) -> Result<(), String> {
        let (tenant_id, connection_id) = (tenant_id.to_string(), connection_id.to_string());
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let encoded = serde_json::to_string(&tokens).map_err(|e| e.to_string())?;
            let conn = writer.lock().map_err(|e| e.to_string())?;
            conn.execute(
                "INSERT INTO connector_credentials (tenant_id, connection_id, tokens)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(tenant_id, connection_id) DO UPDATE SET tokens = excluded.tokens",
                rusqlite::params![tenant_id, connection_id, encoded],
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

        assert!(store.get("default", "linear").await.is_none());

        let first = tokens("access-1");
        store.put("default", "linear", first.clone()).await.unwrap();
        assert_eq!(store.get("default", "linear").await.unwrap(), first);

        // Rotation overwrites in place rather than accumulating rows.
        let second = tokens("access-2");
        store
            .put("default", "linear", second.clone())
            .await
            .unwrap();
        assert_eq!(store.get("default", "linear").await.unwrap(), second);
        cleanup(&path);
    }

    #[tokio::test]
    async fn tenants_and_connections_do_not_leak_into_each_other() {
        let (store, path) = temp_store();

        store.put("a", "linear", tokens("a-linear")).await.unwrap();
        store.put("b", "linear", tokens("b-linear")).await.unwrap();
        store.put("a", "sentry", tokens("a-sentry")).await.unwrap();

        assert_eq!(
            store.get("a", "linear").await.unwrap().access_token,
            "a-linear"
        );
        assert_eq!(
            store.get("b", "linear").await.unwrap().access_token,
            "b-linear"
        );
        assert!(store.get("b", "sentry").await.is_none());
        cleanup(&path);
    }

    /// The point of keying by id: `[mcp.sentry]` and `[mcp.sentry2]` name one
    /// server and hold two accounts.
    #[tokio::test]
    async fn two_ids_on_one_server_hold_two_credentials() {
        let (store, path) = temp_store();

        store
            .put("default", "sentry", tokens("first"))
            .await
            .unwrap();
        store
            .put("default", "sentry2", tokens("second"))
            .await
            .unwrap();

        assert_eq!(
            store.get("default", "sentry").await.unwrap().access_token,
            "first"
        );
        assert_eq!(
            store.get("default", "sentry2").await.unwrap().access_token,
            "second"
        );

        // And forgetting one leaves the other authorized.
        assert!(store.delete("default", "sentry2").await.unwrap());
        assert!(store.get("default", "sentry2").await.is_none());
        assert!(store.get("default", "sentry").await.is_some());
        assert!(!store.delete("default", "sentry2").await.unwrap());
        cleanup(&path);
    }
}
