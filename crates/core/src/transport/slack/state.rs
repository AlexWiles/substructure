use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension};

use crate::event_store::StoreError;
use crate::providers::sqlite::{parse_dt, spawn_err, SqliteDb};

/// The durable half of a live stream slot. The `sent` map is not here: it
/// folds back out of the event log.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StreamRow {
    pub turn_id: String,
    pub start_seq: u64,
    pub started_at: DateTime<Utc>,
    pub ts: Option<String>,
    pub version: u64,
}

/// What a turn's upsert found in place.
#[derive(Debug, PartialEq)]
pub(super) struct TurnSlot {
    pub version: u64,
    /// True when the row already carried this turn: a redelivery.
    pub resumed: bool,
}

/// Slack-owned stream state. Survives restarts so a turn keeps writing into
/// the message it started. Nothing outside the Slack adapter reads it.
pub struct StreamStore {
    db: SqliteDb,
}

impl StreamStore {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        Ok(Self { db })
    }

    pub(super) async fn load(
        &self,
        tenant_id: &str,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<StreamRow>, StoreError> {
        let (tenant_id, session_id, turn_id) = (
            tenant_id.to_string(),
            session_id.to_string(),
            turn_id.to_string(),
        );
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open()?;
            do_load(&conn, &tenant_id, &session_id, &turn_id)
        })
        .await
        .map_err(spawn_err)?
    }

    /// Another turn of this session that still holds a message. One thread
    /// shows one open stream, so a turn seeing this one waits its turn.
    pub(super) async fn open_other(
        &self,
        tenant_id: &str,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<StreamRow>, StoreError> {
        let (tenant_id, session_id, turn_id) = (
            tenant_id.to_string(),
            session_id.to_string(),
            turn_id.to_string(),
        );
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open()?;
            conn.query_row(
                "SELECT turn_id, start_seq, started_at, ts, version
                 FROM slack_turn_streams
                 WHERE tenant_id = ?1 AND session_id = ?2 AND turn_id <> ?3
                   AND ts IS NOT NULL
                 ORDER BY start_seq DESC LIMIT 1",
                rusqlite::params![tenant_id, session_id, turn_id],
                read_row,
            )
            .optional()
            .map_err(|e| StoreError::Internal(e.to_string()))
        })
        .await
        .map_err(spawn_err)?
    }

    /// Record the turn. A redelivery keeps everything the turn already had —
    /// it must not forget its open stream — and only bumps the fence.
    pub(super) async fn upsert_turn(
        &self,
        tenant_id: &str,
        session_id: &str,
        turn_id: &str,
        start_seq: u64,
        started_at: DateTime<Utc>,
    ) -> Result<TurnSlot, StoreError> {
        let (tenant_id, session_id, turn_id) = (
            tenant_id.to_string(),
            session_id.to_string(),
            turn_id.to_string(),
        );
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let tx = conn
                .transaction()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let resumed = do_load(&tx, &tenant_id, &session_id, &turn_id)?.is_some();
            tx.execute(
                "INSERT INTO slack_turn_streams
                   (tenant_id, session_id, turn_id, start_seq, started_at, ts, version, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, NULL, 1, ?6)
                 ON CONFLICT (tenant_id, session_id, turn_id) DO UPDATE SET
                   version = slack_turn_streams.version + 1,
                   updated_at = excluded.updated_at",
                rusqlite::params![
                    tenant_id,
                    session_id,
                    turn_id,
                    start_seq,
                    started_at.to_rfc3339(),
                    Utc::now().to_rfc3339(),
                ],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;
            let version = tx
                .query_row(
                    "SELECT version FROM slack_turn_streams
                     WHERE tenant_id = ?1 AND session_id = ?2 AND turn_id = ?3",
                    rusqlite::params![tenant_id, session_id, turn_id],
                    |row| row.get::<_, u64>(0),
                )
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            tx.commit()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok(TurnSlot { version, resumed })
        })
        .await
        .map_err(spawn_err)?
    }

    /// Record the stream's message. The version guard makes this the fence:
    /// exactly one writer per turn wins; a loser reloads and adopts.
    pub(super) async fn set_ts(
        &self,
        tenant_id: &str,
        session_id: &str,
        turn_id: &str,
        ts: &str,
        expected_version: u64,
    ) -> Result<bool, StoreError> {
        let (tenant_id, session_id, turn_id, ts) = (
            tenant_id.to_string(),
            session_id.to_string(),
            turn_id.to_string(),
            ts.to_string(),
        );
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            let updated = conn
                .execute(
                    "UPDATE slack_turn_streams
                     SET ts = ?1, version = version + 1, updated_at = ?2
                     WHERE tenant_id = ?3 AND session_id = ?4
                       AND turn_id = ?5 AND version = ?6",
                    rusqlite::params![
                        ts,
                        Utc::now().to_rfc3339(),
                        tenant_id,
                        session_id,
                        turn_id,
                        expected_version,
                    ],
                )
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok(updated == 1)
        })
        .await
        .map_err(spawn_err)?
    }

    /// Remove the turn's row. Scoped so a redelivered settle cannot drop a
    /// newer turn's slot.
    pub(super) async fn clear_turn(
        &self,
        tenant_id: &str,
        session_id: &str,
        turn_id: &str,
    ) -> Result<(), StoreError> {
        let (tenant_id, session_id, turn_id) = (
            tenant_id.to_string(),
            session_id.to_string(),
            turn_id.to_string(),
        );
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            conn.execute(
                "DELETE FROM slack_turn_streams
                 WHERE tenant_id = ?1 AND session_id = ?2 AND turn_id = ?3",
                rusqlite::params![tenant_id, session_id, turn_id],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(spawn_err)?
    }

    /// Remove every turn's row: a cancel ends them all.
    pub(super) async fn clear(&self, tenant_id: &str, session_id: &str) -> Result<(), StoreError> {
        let (tenant_id, session_id) = (tenant_id.to_string(), session_id.to_string());
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer
                .lock()
                .map_err(|e| StoreError::Internal(e.to_string()))?;
            conn.execute(
                "DELETE FROM slack_turn_streams WHERE tenant_id = ?1 AND session_id = ?2",
                rusqlite::params![tenant_id, session_id],
            )
            .map_err(|e| StoreError::Internal(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(spawn_err)?
    }
}

fn do_load(
    conn: &Connection,
    tenant_id: &str,
    session_id: &str,
    turn_id: &str,
) -> Result<Option<StreamRow>, StoreError> {
    conn.query_row(
        "SELECT turn_id, start_seq, started_at, ts, version
         FROM slack_turn_streams
         WHERE tenant_id = ?1 AND session_id = ?2 AND turn_id = ?3",
        rusqlite::params![tenant_id, session_id, turn_id],
        read_row,
    )
    .optional()
    .map_err(|e| StoreError::Internal(e.to_string()))
}

fn read_row(row: &rusqlite::Row) -> rusqlite::Result<StreamRow> {
    Ok(StreamRow {
        turn_id: row.get(0)?,
        start_seq: row.get(1)?,
        started_at: row
            .get::<_, String>(2)
            .map(|s| parse_dt(&s).unwrap_or_else(Utc::now))?,
        ts: row.get(3)?,
        version: row.get(4)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// `:memory:` is one shared db per process; each test gets a file.
    fn store(name: &str) -> (StreamStore, String) {
        let path = std::env::temp_dir().join(format!(
            "core-slack-streams-{name}-{}.db",
            uuid::Uuid::now_v7()
        ));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        (StreamStore::new(db).unwrap(), name.to_string())
    }

    fn at(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + secs, 0).unwrap()
    }

    #[tokio::test]
    async fn a_turn_records_and_clears_its_row() {
        let (s, t) = store("t-record");
        assert_eq!(s.load(&t, "s", "turn-1").await.unwrap(), None);
        let slot = s.upsert_turn(&t, "s", "turn-1", 5, at(0)).await.unwrap();
        assert!(!slot.resumed);
        assert!(s
            .set_ts(&t, "s", "turn-1", "111.222", slot.version)
            .await
            .unwrap());
        let row = s.load(&t, "s", "turn-1").await.unwrap().unwrap();
        assert_eq!(row.turn_id, "turn-1");
        assert_eq!(row.start_seq, 5);
        assert_eq!(row.started_at, at(0));
        assert_eq!(row.ts.as_deref(), Some("111.222"));
        s.clear_turn(&t, "s", "turn-1").await.unwrap();
        assert_eq!(s.load(&t, "s", "turn-1").await.unwrap(), None);
    }

    #[tokio::test]
    async fn a_redelivered_turn_keeps_its_stream() {
        // The crash-and-replay path: the same turn must not forget its ts.
        let (s, t) = store("t-redeliver");
        let slot = s.upsert_turn(&t, "s", "turn-1", 5, at(0)).await.unwrap();
        assert!(s
            .set_ts(&t, "s", "turn-1", "111.222", slot.version)
            .await
            .unwrap());
        let again = s.upsert_turn(&t, "s", "turn-1", 5, at(0)).await.unwrap();
        assert!(again.resumed);
        assert_eq!(
            s.load(&t, "s", "turn-1")
                .await
                .unwrap()
                .unwrap()
                .ts
                .as_deref(),
            Some("111.222")
        );
        // A new turn starts clean, and leaves the turn before it alone.
        let next = s.upsert_turn(&t, "s", "turn-2", 9, at(60)).await.unwrap();
        assert!(!next.resumed);
        assert_eq!(s.load(&t, "s", "turn-2").await.unwrap().unwrap().ts, None);
        assert_eq!(
            s.load(&t, "s", "turn-1")
                .await
                .unwrap()
                .unwrap()
                .ts
                .as_deref(),
            Some("111.222")
        );
    }

    /// The queued-turn guard: turn-2 must see that turn-1 still holds a
    /// message, and stop seeing it the moment turn-1 settles.
    #[tokio::test]
    async fn a_turn_sees_another_turns_open_message() {
        let (s, t) = store("t-open-other");
        let one = s.upsert_turn(&t, "s", "turn-1", 5, at(0)).await.unwrap();
        s.upsert_turn(&t, "s", "turn-2", 9, at(60)).await.unwrap();
        // A turn with no message yet blocks nobody.
        assert_eq!(s.open_other(&t, "s", "turn-2").await.unwrap(), None);

        s.set_ts(&t, "s", "turn-1", "111.222", one.version)
            .await
            .unwrap();
        let blocking = s.open_other(&t, "s", "turn-2").await.unwrap().unwrap();
        assert_eq!(blocking.turn_id, "turn-1");
        // Its own message is not another's.
        assert_eq!(s.open_other(&t, "s", "turn-1").await.unwrap(), None);

        s.clear_turn(&t, "s", "turn-1").await.unwrap();
        assert_eq!(s.open_other(&t, "s", "turn-2").await.unwrap(), None);
    }

    #[tokio::test]
    async fn set_ts_is_a_version_fence() {
        let (s, t) = store("t-fence");
        let slot = s.upsert_turn(&t, "s", "turn-1", 5, at(0)).await.unwrap();
        assert!(s
            .set_ts(&t, "s", "turn-1", "111.222", slot.version)
            .await
            .unwrap());
        // A second writer with the stale version loses.
        assert!(!s
            .set_ts(&t, "s", "turn-1", "333.444", slot.version)
            .await
            .unwrap());
        assert_eq!(
            s.load(&t, "s", "turn-1")
                .await
                .unwrap()
                .unwrap()
                .ts
                .as_deref(),
            Some("111.222")
        );
        // The wrong turn cannot write at all.
        let row = s.load(&t, "s", "turn-1").await.unwrap().unwrap();
        assert!(!s
            .set_ts(&t, "s", "turn-9", "555.666", row.version)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn clears_are_scoped_to_their_turn_except_cancel() {
        let (s, t) = store("t-clear");
        s.upsert_turn(&t, "s", "turn-1", 5, at(0)).await.unwrap();
        s.upsert_turn(&t, "s", "turn-2", 9, at(0)).await.unwrap();
        // A settle takes its own turn's row and no other.
        s.clear_turn(&t, "s", "turn-1").await.unwrap();
        assert_eq!(s.load(&t, "s", "turn-1").await.unwrap(), None);
        assert!(s.load(&t, "s", "turn-2").await.unwrap().is_some());
        // A cancel ends them all.
        s.clear(&t, "s").await.unwrap();
        assert_eq!(s.load(&t, "s", "turn-2").await.unwrap(), None);
    }
}
