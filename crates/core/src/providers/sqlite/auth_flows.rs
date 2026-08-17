//! Authorization links and the flows they start.
//!
//! `minted ──attach──▶ pending ──claim──▶ claimed`, then deleted on success
//! or back to `minted` on failure. A link is spent by authorizing, not by
//! being clicked, so a stranger cannot burn the link somebody is waiting on.
//!
//! Both handles are stored hashed: the link token, and the OAuth `state` the
//! redirect carries.

use chrono::{DateTime, Utc};
use rusqlite::OptionalExtension;

use crate::connectors::Slot;
use crate::event_store::StoreError;
use crate::runtime::secret::SecretRef;

use super::SqliteDb;

/// How long a claimed row may sit before the sweeper calls its callback
/// dead. Longer than any live code exchange.
pub const CLAIM_TTL: chrono::Duration = chrono::Duration::minutes(10);

/// What a link stands for.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    pub link_hash: String,
    pub tenant_id: String,
    pub connection_id: String,
    pub subject: Slot,
}

/// A claimed flow, ready to redeem.
#[derive(Debug, Clone, PartialEq)]
pub struct Flow {
    pub link_hash: String,
    pub tenant_id: String,
    pub connection_id: String,
    pub subject: Slot,
    /// The PKCE half of the flow, in the secret store.
    pub pending_secret: SecretRef,
}

pub struct SqliteAuthFlows {
    db: SqliteDb,
}

fn internal<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Internal(e.to_string())
}

fn subject_of(column: &str) -> Result<Slot, StoreError> {
    serde_json::from_str(column).map_err(internal)
}

impl SqliteAuthFlows {
    pub fn new(db: SqliteDb) -> Self {
        Self { db }
    }

    /// Record a link about to be handed out.
    pub async fn mint(
        &self,
        link: Link,
        now: DateTime<Utc>,
        expires_at: DateTime<Utc>,
    ) -> Result<(), StoreError> {
        let subject = serde_json::to_string(&link.subject).map_err(internal)?;
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(internal)?;
            conn.execute(
                "INSERT INTO auth_flows
                     (link_hash, tenant_id, connection_id, subject, state,
                      created_at, updated_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, 'minted', ?5, ?5, ?6)",
                rusqlite::params![
                    link.link_hash,
                    link.tenant_id,
                    link.connection_id,
                    subject,
                    now.to_rfc3339(),
                    expires_at.to_rfc3339(),
                ],
            )
            .map_err(internal)?;
            Ok(())
        })
        .await
        .map_err(internal)?
    }

    /// What a presented link stands for. `None` when no live row answers to
    /// it. A claimed row answers nothing: its callback is in flight.
    pub async fn resolve(
        &self,
        link_hash: &str,
        now: DateTime<Utc>,
    ) -> Result<Option<Link>, StoreError> {
        let link_hash = link_hash.to_string();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().map_err(internal)?;
            let row = conn
                .query_row(
                    "SELECT tenant_id, connection_id, subject FROM auth_flows
                     WHERE link_hash = ?1 AND state != 'claimed' AND expires_at > ?2",
                    rusqlite::params![link_hash, now.to_rfc3339()],
                    |r| {
                        Ok((
                            r.get::<_, String>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, String>(2)?,
                        ))
                    },
                )
                .optional()
                .map_err(internal)?;
            let Some((tenant_id, connection_id, subject)) = row else {
                return Ok(None);
            };
            Ok(Some(Link {
                link_hash,
                tenant_id,
                connection_id,
                subject: subject_of(&subject)?,
            }))
        })
        .await
        .map_err(internal)?
    }

    /// Hang an OAuth handshake off a minted link. A second click replaces
    /// the first, and the secret it displaced comes back for deletion.
    pub async fn attach(
        &self,
        link_hash: &str,
        state_hash: &str,
        secret: &SecretRef,
        now: DateTime<Utc>,
    ) -> Result<Option<SecretRef>, StoreError> {
        let (link_hash, state_hash) = (link_hash.to_string(), state_hash.to_string());
        let secret = secret.as_str().to_string();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer.lock().map_err(internal)?;
            let tx = conn.transaction().map_err(internal)?;
            let displaced: Option<String> = tx
                .query_row(
                    "SELECT pending_secret FROM auth_flows WHERE link_hash = ?1",
                    rusqlite::params![link_hash],
                    |r| r.get(0),
                )
                .optional()
                .map_err(internal)?
                .flatten();
            tx.execute(
                "UPDATE auth_flows
                    SET state = 'pending', state_hash = ?2, pending_secret = ?3, updated_at = ?4
                  WHERE link_hash = ?1 AND state != 'claimed'",
                rusqlite::params![link_hash, state_hash, secret, now.to_rfc3339()],
            )
            .map_err(internal)?;
            tx.commit().map_err(internal)?;
            Ok(displaced.map(SecretRef::from_stored))
        })
        .await
        .map_err(internal)?
    }

    /// Move one `pending` flow to `claimed` and answer with it. `None` when
    /// the row is absent, already claimed, or expired.
    pub async fn claim(
        &self,
        state_hash: &str,
        now: DateTime<Utc>,
    ) -> Result<Option<Flow>, StoreError> {
        let state_hash = state_hash.to_string();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(internal)?;
            let claimed = conn
                .execute(
                    "UPDATE auth_flows SET state = 'claimed', updated_at = ?2
                     WHERE state_hash = ?1 AND state = 'pending' AND expires_at > ?2",
                    rusqlite::params![state_hash, now.to_rfc3339()],
                )
                .map_err(internal)?;
            if claimed == 0 {
                return Ok(None);
            }
            let row = conn
                .query_row(
                    "SELECT link_hash, tenant_id, connection_id, subject, pending_secret
                     FROM auth_flows WHERE state_hash = ?1",
                    rusqlite::params![state_hash],
                    |r| {
                        Ok((
                            r.get::<_, String>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, String>(2)?,
                            r.get::<_, String>(3)?,
                            r.get::<_, Option<String>>(4)?,
                        ))
                    },
                )
                .optional()
                .map_err(internal)?;
            let Some((link_hash, tenant_id, connection_id, subject, secret)) = row else {
                return Ok(None);
            };
            let Some(secret) = secret else {
                return Ok(None);
            };
            Ok(Some(Flow {
                link_hash,
                tenant_id,
                connection_id,
                subject: subject_of(&subject)?,
                pending_secret: SecretRef::from_stored(secret),
            }))
        })
        .await
        .map_err(internal)?
    }

    /// The credential is stored, so the link is done.
    pub async fn complete(&self, flow: &Flow) -> Result<(), StoreError> {
        let link_hash = flow.link_hash.clone();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(internal)?;
            conn.execute(
                "DELETE FROM auth_flows WHERE link_hash = ?1",
                rusqlite::params![link_hash],
            )
            .map_err(internal)?;
            Ok(())
        })
        .await
        .map_err(internal)?
    }

    /// The handshake failed. The link goes back to unused.
    pub async fn release(&self, flow: &Flow) -> Result<(), StoreError> {
        let link_hash = flow.link_hash.clone();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(internal)?;
            conn.execute(
                "UPDATE auth_flows
                    SET state = 'minted', state_hash = NULL, pending_secret = NULL
                  WHERE link_hash = ?1",
                rusqlite::params![link_hash],
            )
            .map_err(internal)?;
            Ok(())
        })
        .await
        .map_err(internal)?
    }

    /// Delete links past their deadline and claimed rows whose callback
    /// died. Answers with the unreferenced pending secrets and their tenants.
    pub async fn sweep(&self, now: DateTime<Utc>) -> Result<Vec<(String, SecretRef)>, StoreError> {
        let stale_before = (now - CLAIM_TTL).to_rfc3339();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer.lock().map_err(internal)?;
            let tx = conn.transaction().map_err(internal)?;
            let dead: Vec<(String, String, Option<String>)> = {
                let mut stmt = tx
                    .prepare(
                        "SELECT link_hash, tenant_id, pending_secret
                         FROM auth_flows
                         WHERE expires_at <= ?1
                            OR (state = 'claimed' AND updated_at <= ?2)",
                    )
                    .map_err(internal)?;
                let mapped = stmt
                    .query_map(rusqlite::params![now.to_rfc3339(), stale_before], |r| {
                        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                    })
                    .map_err(internal)?;
                mapped.collect::<rusqlite::Result<_>>().map_err(internal)?
            };
            let mut orphaned = Vec::new();
            for (link_hash, tenant_id, secret) in dead {
                tx.execute(
                    "DELETE FROM auth_flows WHERE link_hash = ?1",
                    rusqlite::params![link_hash],
                )
                .map_err(internal)?;
                if let Some(secret) = secret {
                    orphaned.push((tenant_id, SecretRef::from_stored(secret)));
                }
            }
            tx.commit().map_err(internal)?;
            Ok(orphaned)
        })
        .await
        .map_err(internal)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Issuer, Subject};
    use std::time::Duration;
    use uuid::Uuid;

    const LINK_TTL: chrono::Duration = chrono::Duration::hours(4);

    fn temp() -> (SqliteAuthFlows, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("core-auth-flows-{}.db", Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        (SqliteAuthFlows::new(db), path)
    }

    fn cleanup(path: &std::path::Path) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    fn link(hash: &str) -> Link {
        Link {
            link_hash: hash.to_string(),
            tenant_id: "default".to_string(),
            connection_id: "gmail".to_string(),
            subject: Slot::Of(Subject::new(Issuer::slack(), "T1:U1")),
        }
    }

    fn secret(id: &str) -> SecretRef {
        SecretRef::from_stored(id.to_string())
    }

    async fn minted(flows: &SqliteAuthFlows, hash: &str, now: DateTime<Utc>) {
        flows.mint(link(hash), now, now + LINK_TTL).await.unwrap();
    }

    /// Mint, click, claim once, complete. A replayed redirect claims
    /// nothing, and the spent link resolves to nothing.
    #[tokio::test]
    async fn a_link_is_spent_by_authorizing() {
        let (flows, path) = temp();
        let now = Utc::now();

        minted(&flows, "L1", now).await;
        assert_eq!(flows.resolve("L1", now).await.unwrap(), Some(link("L1")));

        flows
            .attach("L1", "S1", &secret("sec-1"), now)
            .await
            .unwrap();
        let claimed = flows.claim("S1", now).await.unwrap().expect("first claim");
        assert_eq!(
            claimed.subject,
            Slot::Of(Subject::new(Issuer::slack(), "T1:U1"))
        );

        assert!(
            flows.claim("S1", now).await.unwrap().is_none(),
            "a replayed redirect stops"
        );

        flows.complete(&claimed).await.unwrap();
        assert!(flows.resolve("L1", now).await.unwrap().is_none());
        cleanup(&path);
    }

    /// Declining at the provider does not burn the link.
    #[tokio::test]
    async fn a_failed_handshake_leaves_the_link_usable() {
        let (flows, path) = temp();
        let now = Utc::now();

        minted(&flows, "L1", now).await;
        flows
            .attach("L1", "S1", &secret("sec-1"), now)
            .await
            .unwrap();
        let claimed = flows.claim("S1", now).await.unwrap().unwrap();
        flows.release(&claimed).await.unwrap();

        assert_eq!(flows.resolve("L1", now).await.unwrap(), Some(link("L1")));
        assert!(
            flows.claim("S1", now).await.unwrap().is_none(),
            "the released handshake is not reusable, only the link is"
        );
        cleanup(&path);
    }

    /// A second click replaces the first handshake and hands back the
    /// displaced secret, so none is stranded in the secret store.
    #[tokio::test]
    async fn a_second_click_replaces_the_first_handshake() {
        let (flows, path) = temp();
        let now = Utc::now();

        minted(&flows, "L1", now).await;
        assert_eq!(
            flows
                .attach("L1", "S1", &secret("sec-1"), now)
                .await
                .unwrap(),
            None,
            "the first click displaces nothing"
        );
        assert_eq!(
            flows
                .attach("L1", "S2", &secret("sec-2"), now)
                .await
                .unwrap(),
            Some(secret("sec-1"))
        );

        assert!(flows.claim("S1", now).await.unwrap().is_none());
        assert!(flows.claim("S2", now).await.unwrap().is_some());
        cleanup(&path);
    }

    /// An unknown or expired link resolves to nothing and claims nothing.
    #[tokio::test]
    async fn an_expired_or_unknown_link_is_refused() {
        let (flows, path) = temp();
        let now = Utc::now();

        assert!(flows.resolve("nope", now).await.unwrap().is_none());

        minted(&flows, "L1", now).await;
        flows
            .attach("L1", "S1", &secret("sec-1"), now)
            .await
            .unwrap();
        let later = now + LINK_TTL + chrono::Duration::seconds(1);
        assert!(flows.resolve("L1", later).await.unwrap().is_none());
        assert!(flows.claim("S1", later).await.unwrap().is_none());
        cleanup(&path);
    }

    /// A link past its deadline goes. A claimed flow whose callback died
    /// goes on its own shorter bound. Their secrets come back for deletion.
    #[tokio::test]
    async fn the_sweeper_clears_expired_links_and_dead_claims() {
        let (flows, path) = temp();
        let now = Utc::now();

        minted(&flows, "unclicked", now).await;
        minted(&flows, "crashed", now).await;
        flows
            .attach("crashed", "S1", &secret("sec-1"), now)
            .await
            .unwrap();
        flows.claim("S1", now).await.unwrap().expect("claimed");

        // Before either bound the sweeper touches nothing: the person may
        // still be looking at the consent page.
        assert!(flows.sweep(now).await.unwrap().is_empty());

        let dead_claim = now + CLAIM_TTL + chrono::Duration::seconds(1);
        assert_eq!(
            flows.sweep(dead_claim).await.unwrap(),
            vec![("default".to_string(), secret("sec-1"))],
            "the dead claim goes first, and its secret with it"
        );
        assert!(
            flows
                .resolve("unclicked", dead_claim)
                .await
                .unwrap()
                .is_some(),
            "an unclicked link keeps its full deadline"
        );

        let expired = now + LINK_TTL + chrono::Duration::seconds(1);
        assert!(flows.sweep(expired).await.unwrap().is_empty());
        assert!(flows.resolve("unclicked", expired).await.unwrap().is_none());
        cleanup(&path);
    }
}
