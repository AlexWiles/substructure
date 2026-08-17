//! The secret store, in the engine's own database: one table, sealed at rest
//! when a key is configured, and referenced by id from everywhere else.

use std::sync::Arc;

use async_trait::async_trait;
use rusqlite::OptionalExtension;

use crate::event_store::StoreError;
use crate::runtime::secret::{SecretCipher, SecretRef, SecretStore, KEYS_ENV};

use super::SqliteDb;

pub struct SqliteSecretStore {
    db: SqliteDb,
    /// Set ⇒ every write is sealed. Unset ⇒ rows land as plain JSON. Rows
    /// written before a key appeared stay readable and are sealed whenever
    /// they are next written — a refresh or a new login rewrites them.
    cipher: Option<Arc<SecretCipher>>,
}

impl SqliteSecretStore {
    pub fn new(db: SqliteDb, cipher: Option<Arc<SecretCipher>>) -> Self {
        Self { db, cipher }
    }
}

#[async_trait]
impl SecretStore for SqliteSecretStore {
    async fn put(&self, tenant_id: &str, r: &SecretRef, value: &[u8]) -> Result<(), StoreError> {
        let (tenant_id, id) = (tenant_id.to_string(), r.as_str().to_string());
        let cipher = self.cipher.clone();
        let value = value.to_vec();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let (data, key_version): (String, Option<i64>) = match &cipher {
                Some(c) => {
                    let (version, ciphertext) = c.encrypt(&value);
                    (ciphertext, Some(version as i64))
                }
                None => (
                    String::from_utf8(value)
                        .map_err(|e| StoreError::Internal(format!("secret is not UTF-8: {e}")))?,
                    None,
                ),
            };
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            conn.execute(
                "INSERT INTO secrets (tenant_id, id, data, key_version, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?5)
                 ON CONFLICT(tenant_id, id) DO UPDATE
                     SET data = excluded.data,
                         key_version = excluded.key_version,
                         updated_at = excluded.updated_at",
                rusqlite::params![
                    tenant_id,
                    id,
                    data,
                    key_version,
                    chrono::Utc::now().to_rfc3339()
                ],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))?
    }

    async fn get(&self, tenant_id: &str, r: &SecretRef) -> Result<Option<Vec<u8>>, StoreError> {
        let (tenant_id, id) = (tenant_id.to_string(), r.as_str().to_string());
        let cipher = self.cipher.clone();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader
                .open()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let row: Option<(String, Option<i64>)> = conn
                .query_row(
                    "SELECT data, key_version FROM secrets WHERE tenant_id = ?1 AND id = ?2",
                    rusqlite::params![tenant_id, id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let Some((data, key_version)) = row else {
                return Ok(None);
            };
            match key_version {
                None => Ok(Some(data.into_bytes())),
                Some(_) => match &cipher {
                    Some(c) => c
                        .decrypt(&data)
                        .map(Some)
                        .map_err(|e| StoreError::Internal(format!("secret `{id}`: {e}"))),
                    None => Err(StoreError::Internal(format!(
                        "secret `{id}` is encrypted and no ${KEYS_ENV} is set"
                    ))),
                },
            }
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))?
    }

    async fn delete(&self, tenant_id: &str, r: &SecretRef) -> Result<bool, StoreError> {
        let (tenant_id, id) = (tenant_id.to_string(), r.as_str().to_string());
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let removed = conn
                .execute(
                    "DELETE FROM secrets WHERE tenant_id = ?1 AND id = ?2",
                    rusqlite::params![tenant_id, id],
                )
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok(removed > 0)
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use uuid::Uuid;

    fn temp_db() -> (SqliteDb, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("core-secrets-{}.db", Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        (db, path)
    }

    fn cleanup(path: &std::path::Path) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    fn sref(id: &str) -> SecretRef {
        SecretRef::from_stored(id.to_string())
    }

    fn cipher() -> Arc<SecretCipher> {
        Arc::new(SecretCipher::parse(&"ab".repeat(32)).unwrap())
    }

    fn raw_row(db: &SqliteDb, id: &str) -> (String, Option<i64>) {
        let conn = db.writer.lock().unwrap();
        conn.query_row(
            "SELECT data, key_version FROM secrets WHERE id = ?1",
            [id],
            |row| Ok((row.get(0).unwrap(), row.get(1).unwrap())),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn a_sealed_secret_round_trips_and_never_lands_plain() {
        let (db, path) = temp_db();
        let store = SqliteSecretStore::new(db.clone(), Some(cipher()));

        store.put("t", &sref("s1"), b"xoxb-secret").await.unwrap();
        assert_eq!(
            store.get("t", &sref("s1")).await.unwrap().unwrap(),
            b"xoxb-secret"
        );

        let (data, key_version) = raw_row(&db, "s1");
        assert_eq!(key_version, Some(1));
        assert!(!data.contains("xoxb"), "the table holds ciphertext: {data}");

        assert!(store.delete("t", &sref("s1")).await.unwrap());
        assert!(store.get("t", &sref("s1")).await.unwrap().is_none());
        assert!(!store.delete("t", &sref("s1")).await.unwrap());
        cleanup(&path);
    }

    /// A row written before the key stays readable, and its next write seals
    /// it: sealing happens on write, not by a sweep.
    #[tokio::test]
    async fn a_plaintext_row_reads_under_a_key_and_seals_on_rewrite() {
        let (db, path) = temp_db();
        let plain = SqliteSecretStore::new(db.clone(), None);
        plain.put("t", &sref("s1"), b"ghp_token").await.unwrap();
        assert_eq!(raw_row(&db, "s1").1, None, "plaintext before the key");

        let sealed = SqliteSecretStore::new(db.clone(), Some(cipher()));
        assert_eq!(
            sealed.get("t", &sref("s1")).await.unwrap().unwrap(),
            b"ghp_token"
        );
        sealed.put("t", &sref("s1"), b"ghp_token2").await.unwrap();
        assert_eq!(raw_row(&db, "s1").1, Some(1));
        assert_eq!(
            sealed.get("t", &sref("s1")).await.unwrap().unwrap(),
            b"ghp_token2"
        );
        cleanup(&path);
    }

    /// An encrypted row with no key is an error that names the variable, not a
    /// silent miss.
    #[tokio::test]
    async fn an_encrypted_row_without_a_key_says_what_is_missing() {
        let (db, path) = temp_db();
        let sealed = SqliteSecretStore::new(db.clone(), Some(cipher()));
        sealed.put("t", &sref("s1"), b"secret").await.unwrap();

        let keyless = SqliteSecretStore::new(db.clone(), None);
        let err = keyless.get("t", &sref("s1")).await.unwrap_err();
        assert!(err.to_string().contains(KEYS_ENV), "{err}");
        cleanup(&path);
    }
}
