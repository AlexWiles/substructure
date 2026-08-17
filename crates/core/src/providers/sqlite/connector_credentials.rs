use std::sync::Arc;

use async_trait::async_trait;

use crate::connectors::credential::{Credential, CredentialStore};
use crate::connectors::{Issuer, Slot, Subject};
use crate::event_store::StoreError;
use crate::runtime::secret::{SecretRef, SecretStore};

use super::SqliteDb;

/// Authorized connections, in the engine's own database.
///
/// The rows here hold no token: each one references the secret store, which is
/// the only table that holds secret material and the only place it is
/// encrypted. `subs mcp login` and `subs mcp set-token` write here and the
/// engine reads here, so one environment is one file and one set of
/// credentials.
pub struct SqliteCredentialStore {
    db: SqliteDb,
    secrets: Arc<dyn SecretStore>,
}

/// The column encoding of a subject. SQLite forbids NULL inside a composite
/// primary key, so the shared slot is spelled as a pair of empty strings here
/// and nowhere else.
fn slot_columns(slot: &Slot) -> (&str, &str) {
    match slot {
        Slot::Shared => ("", ""),
        Slot::Of(subject) => (subject.issuer.as_str(), &subject.id),
    }
}

impl SqliteCredentialStore {
    pub fn new(db: SqliteDb, secrets: Arc<dyn SecretStore>) -> Result<Self, StoreError> {
        Ok(Self { db, secrets })
    }

    /// Forget the credentials for connections `declared` does not name,
    /// answering with the ids forgotten.
    ///
    /// The file is the whole declaration, so a `[mcp.<id>]` taken out of it is
    /// a connection that was removed, and its credentials go with it — every
    /// holder's, not one slot's. Called as the engine starts, which is when
    /// the file is applied.
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

    /// Every connection this tenant holds a credential for, whoever holds it.
    async fn ids(&self, tenant_id: &str) -> Result<Vec<String>, StoreError> {
        let tenant_id = tenant_id.to_string();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader
                .open()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let mut stmt = conn
                .prepare(
                    "SELECT DISTINCT connection_id FROM connector_credentials
                     WHERE tenant_id = ?1",
                )
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

    /// Everyone holding a credential for one connection.
    pub async fn holders(
        &self,
        tenant_id: &str,
        connection_id: &str,
    ) -> Result<Vec<Slot>, StoreError> {
        let (tenant_id, connection_id) = (tenant_id.to_string(), connection_id.to_string());
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader
                .open()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let mut stmt = conn
                .prepare(
                    "SELECT issuer, subject FROM connector_credentials
                     WHERE tenant_id = ?1 AND connection_id = ?2",
                )
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let rows = stmt
                .query_map(rusqlite::params![tenant_id, connection_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            rows.map(|held| {
                held.map(|(issuer, id)| match issuer.is_empty() {
                    true => Slot::Shared,
                    false => Slot::Of(Subject::new(Issuer::new(issuer), id)),
                })
                .map_err(|e| StoreError::Internal(e.to_string()))
            })
            .collect()
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))?
    }

    /// The secret each matching row references. `subject` narrows to one
    /// holder; `None` is every holder of the connection.
    async fn secret_refs(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: Option<&Slot>,
    ) -> Result<Vec<SecretRef>, StoreError> {
        let (tenant_id, connection_id) = (tenant_id.to_string(), connection_id.to_string());
        let slot = subject.map(|s| {
            let (issuer, id) = slot_columns(s);
            (issuer.to_string(), id.to_string())
        });
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader
                .open()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let (sql, params): (&str, Vec<String>) = match slot {
                Some((issuer, id)) => (
                    "SELECT secret_id FROM connector_credentials
                     WHERE tenant_id = ?1 AND connection_id = ?2
                       AND issuer = ?3 AND subject = ?4",
                    vec![tenant_id, connection_id, issuer, id],
                ),
                None => (
                    "SELECT secret_id FROM connector_credentials
                     WHERE tenant_id = ?1 AND connection_id = ?2",
                    vec![tenant_id, connection_id],
                ),
            };
            let mut stmt = conn
                .prepare(sql)
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(params), |row| row.get(0))
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            rows.collect::<rusqlite::Result<Vec<String>>>()
                .map(|ids| ids.into_iter().map(SecretRef::from_stored).collect())
                .map_err(|e| StoreError::Internal(e.to_string()))
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))?
    }

    async fn delete_rows(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: Option<&Slot>,
    ) -> Result<bool, StoreError> {
        // The reference goes first: a row pointing at a deleted secret reads
        // as no credential, while a secret nothing references only lingers.
        let secret_refs = self.secret_refs(tenant_id, connection_id, subject).await?;
        let (tenant, connection) = (tenant_id.to_string(), connection_id.to_string());
        let slot = subject.map(|s| {
            let (issuer, id) = slot_columns(s);
            (issuer.to_string(), id.to_string())
        });
        let writer = self.db.writer.clone();
        let removed = tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let removed = match slot {
                Some((issuer, id)) => conn.execute(
                    "DELETE FROM connector_credentials
                     WHERE tenant_id = ?1 AND connection_id = ?2
                       AND issuer = ?3 AND subject = ?4",
                    rusqlite::params![tenant, connection, issuer, id],
                ),
                None => conn.execute(
                    "DELETE FROM connector_credentials
                     WHERE tenant_id = ?1 AND connection_id = ?2",
                    rusqlite::params![tenant, connection],
                ),
            }
            .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok::<_, StoreError>(removed > 0)
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))??;
        for secret_ref in secret_refs {
            self.secrets.delete(tenant_id, &secret_ref).await?;
        }
        Ok(removed)
    }

    /// Forget every holder's credential for one connection. `false` when
    /// there was none to forget.
    pub async fn delete(&self, tenant_id: &str, connection_id: &str) -> Result<bool, StoreError> {
        self.delete_rows(tenant_id, connection_id, None).await
    }

    async fn upsert_row(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Slot,
        secret_ref: &SecretRef,
    ) -> Result<(), StoreError> {
        let (issuer, id) = slot_columns(subject);
        let (tenant_id, connection_id, issuer, id, secret_id) = (
            tenant_id.to_string(),
            connection_id.to_string(),
            issuer.to_string(),
            id.to_string(),
            secret_ref.as_str().to_string(),
        );
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            conn.execute(
                "INSERT INTO connector_credentials
                     (tenant_id, connection_id, issuer, subject, secret_id)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(tenant_id, connection_id, issuer, subject)
                 DO UPDATE SET secret_id = excluded.secret_id",
                rusqlite::params![tenant_id, connection_id, issuer, id, secret_id],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| StoreError::Internal(e.to_string()))?
    }
}

#[async_trait]
impl CredentialStore for SqliteCredentialStore {
    async fn get(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Slot,
    ) -> Option<Credential> {
        let secret_ref = self
            .secret_refs(tenant_id, connection_id, Some(subject))
            .await
            .ok()?
            .into_iter()
            .next()?;
        match self.secrets.get(tenant_id, &secret_ref).await {
            Ok(Some(bytes)) => serde_json::from_slice(&bytes).ok(),
            Ok(None) => None,
            Err(e) => {
                tracing::warn!(connection = %connection_id, error = %e,
                    "cannot read a stored credential");
                None
            }
        }
    }

    async fn put(
        &self,
        tenant_id: &str,
        connection_id: &str,
        subject: &Slot,
        credential: Credential,
    ) -> Result<(), String> {
        if matches!(subject, Slot::Of(s) if s.id.is_empty() || s.issuer.as_str().is_empty()) {
            return Err("a person's subject names both a source and an id".to_string());
        }
        // A slot keeps its secret id across rotations, so a refresh replaces
        // the value in place and nothing else moves.
        let existing = self
            .secret_refs(tenant_id, connection_id, Some(subject))
            .await
            .map_err(|e| e.to_string())?
            .into_iter()
            .next();
        let secret_ref = existing.unwrap_or_else(SecretRef::mint);
        let encoded = serde_json::to_vec(&credential).map_err(|e| e.to_string())?;
        self.secrets
            .put(tenant_id, &secret_ref, &encoded)
            .await
            .map_err(|e| e.to_string())?;
        self.upsert_row(tenant_id, connection_id, subject, &secret_ref)
            .await
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::oauth::{ClientId, Tokens};
    use crate::providers::sqlite::SqliteSecretStore;
    use std::time::Duration;
    use uuid::Uuid;

    /// A file per test: `:memory:` opens a shared-cache database, so in-memory
    /// stores would see each other's rows.
    fn temp_store() -> (SqliteCredentialStore, SqliteDb, std::path::PathBuf) {
        let path =
            std::env::temp_dir().join(format!("core-connector-credentials-{}.db", Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        let secrets = Arc::new(SqliteSecretStore::new(db.clone(), None));
        (
            SqliteCredentialStore::new(db.clone(), secrets).unwrap(),
            db,
            path,
        )
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    fn oauth(access: &str) -> Credential {
        Credential::Oauth(Box::new(Tokens {
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
        }))
    }

    fn token(value: &str) -> Credential {
        Credential::Static {
            token: value.into(),
        }
    }

    /// What `get` answers with is the shape that was written, so a slot filled
    /// by one command is never read as the other's.
    fn access(credential: Credential) -> String {
        match credential {
            Credential::Oauth(tokens) => tokens.access_token,
            Credential::Static { token } => token,
        }
    }

    fn secret_count(db: &SqliteDb) -> i64 {
        let conn = db.writer.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM secrets", [], |r| r.get(0))
            .unwrap()
    }

    #[tokio::test]
    async fn a_credential_round_trips_and_a_refresh_replaces_it() {
        let (store, db, path) = temp_store();

        assert!(store
            .get("default", "linear", &Slot::Shared)
            .await
            .is_none());

        let first = oauth("access-1");
        store
            .put("default", "linear", &Slot::Shared, first.clone())
            .await
            .unwrap();
        assert_eq!(
            store.get("default", "linear", &Slot::Shared).await,
            Some(first)
        );

        // Rotation overwrites in place rather than accumulating rows.
        let second = oauth("access-2");
        store
            .put("default", "linear", &Slot::Shared, second.clone())
            .await
            .unwrap();
        assert_eq!(
            store.get("default", "linear", &Slot::Shared).await,
            Some(second)
        );
        assert_eq!(secret_count(&db), 1, "one slot is one secret");
        cleanup(&path);
    }

    /// One slot per holder: two persons on one connection are two credentials,
    /// and the shared slot is a third.
    #[tokio::test]
    async fn each_holder_of_a_connection_has_their_own_slot() {
        let (store, db, path) = temp_store();
        let alex = Slot::Of(Subject::new(Issuer::slack(), "T1:U1"));
        let sam = Slot::Of(Subject::new(Issuer::slack(), "T1:U2"));

        store
            .put("default", "gmail", &alex, oauth("alex-token"))
            .await
            .unwrap();
        store
            .put("default", "gmail", &sam, oauth("sam-token"))
            .await
            .unwrap();

        assert_eq!(
            store.get("default", "gmail", &alex).await.map(access),
            Some("alex-token".into())
        );
        assert_eq!(
            store.get("default", "gmail", &sam).await.map(access),
            Some("sam-token".into())
        );
        assert!(
            store.get("default", "gmail", &Slot::Shared).await.is_none(),
            "nobody filled the shared slot"
        );

        assert_eq!(secret_count(&db), 2, "one secret per holder");
        cleanup(&path);
    }

    /// One slot, either kind: setting a token over a login replaces it, which
    /// is how a connection switched from `auth = "oauth"` to `auth = "token"`
    /// stops carrying the grant it no longer declares.
    #[tokio::test]
    async fn a_static_token_shares_the_slot_with_a_grant() {
        let (store, _db, path) = temp_store();

        store
            .put("default", "github", &Slot::Shared, token("ghp_1"))
            .await
            .unwrap();
        assert_eq!(
            store
                .get("default", "github", &Slot::Shared)
                .await
                .map(access)
                .unwrap(),
            "ghp_1"
        );

        let granted = oauth("access-1");
        store
            .put("default", "github", &Slot::Shared, granted.clone())
            .await
            .unwrap();
        assert_eq!(
            store.get("default", "github", &Slot::Shared).await,
            Some(granted)
        );
        cleanup(&path);
    }

    #[tokio::test]
    async fn tenants_and_connections_do_not_leak_into_each_other() {
        let (store, _db, path) = temp_store();

        store
            .put("a", "linear", &Slot::Shared, oauth("a-linear"))
            .await
            .unwrap();
        store
            .put("b", "linear", &Slot::Shared, oauth("b-linear"))
            .await
            .unwrap();
        store
            .put("a", "sentry", &Slot::Shared, oauth("a-sentry"))
            .await
            .unwrap();

        assert_eq!(
            store
                .get("a", "linear", &Slot::Shared)
                .await
                .map(access)
                .unwrap(),
            "a-linear"
        );
        assert_eq!(
            store
                .get("b", "linear", &Slot::Shared)
                .await
                .map(access)
                .unwrap(),
            "b-linear"
        );
        assert!(store.get("b", "sentry", &Slot::Shared).await.is_none());
        cleanup(&path);
    }

    /// A connection taken out of the file loses every holder's credential,
    /// not just the shared one — and the secrets go with the references.
    #[tokio::test]
    async fn retain_forgets_every_holder_of_a_removed_connection() {
        let (store, db, path) = temp_store();

        store
            .put("default", "sentry", &Slot::Shared, oauth("company"))
            .await
            .unwrap();
        store
            .put(
                "default",
                "gmail",
                &Slot::Of(Subject::new(Issuer::slack(), "T1:U1")),
                oauth("personal"),
            )
            .await
            .unwrap();

        let forgotten = store
            .retain("default", &["sentry".to_string()])
            .await
            .unwrap();
        assert_eq!(forgotten, ["gmail"]);
        assert!(store
            .get(
                "default",
                "gmail",
                &Slot::Of(Subject::new(Issuer::slack(), "T1:U1"))
            )
            .await
            .is_none());
        assert!(store
            .get("default", "sentry", &Slot::Shared)
            .await
            .is_some());
        assert_eq!(secret_count(&db), 1, "the secret went with the reference");
        cleanup(&path);
    }
}
