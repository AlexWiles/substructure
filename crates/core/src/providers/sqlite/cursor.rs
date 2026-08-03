use async_trait::async_trait;
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};

use crate::event_store::{Seq, StoreError};
use crate::processor::{CursorError, ProcessorCursorStore, StreamCursor, StreamRef};

use super::{parse_dt, SqliteDb};

/// Stream heads come from `snapshots`, written by the event store on every
/// append: `seq` there is the last event in the stream, and `shard_key` its
/// shard hash. Pending work is the join of that against the cursors below.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS projection_cursors (
    projection_name TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    version         INTEGER NOT NULL,
    owner_id        TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (projection_name, tenant_id, session_id)
);

CREATE TABLE IF NOT EXISTS projection_seeds (
    projection_name TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL
);
";

pub struct SqliteCursorStore {
    db: SqliteDb,
}

impl SqliteCursorStore {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        db.run_schema(SCHEMA)?;
        Ok(Self { db })
    }
}

#[async_trait]
impl ProcessorCursorStore for SqliteCursorStore {
    async fn load_cursor(
        &self,
        processor: &str,
        stream: &StreamRef,
    ) -> Result<StreamCursor, CursorError> {
        let processor = processor.to_string();
        let stream = stream.clone();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().map_err(message)?;
            do_load_cursor(&conn, &processor, &stream)
        })
        .await
        .map_err(message)?
    }

    async fn compare_and_set_cursor(
        &self,
        processor: &str,
        stream: &StreamRef,
        expected_version: u64,
        new_seq: Seq,
        owner_id: Option<&str>,
    ) -> Result<bool, CursorError> {
        let processor = processor.to_string();
        let stream = stream.clone();
        let owner_id = owner_id.map(str::to_string);
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer.lock().map_err(message)?;
            do_compare_and_set(
                &mut conn,
                &processor,
                &stream,
                expected_version,
                new_seq,
                owner_id.as_deref(),
            )
        })
        .await
        .map_err(message)?
    }

    async fn pending_streams(
        &self,
        processor: &str,
        shard_id: u32,
        shard_count: u32,
        limit: usize,
    ) -> Result<Vec<StreamRef>, CursorError> {
        let processor = processor.to_string();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().map_err(message)?;
            do_pending_streams(&conn, &processor, shard_id, shard_count, limit)
        })
        .await
        .map_err(message)?
    }

    async fn seed_at_tail(&self, processor: &str) -> Result<(), CursorError> {
        let processor = processor.to_string();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer.lock().map_err(message)?;
            do_seed_at_tail(&mut conn, &processor)
        })
        .await
        .map_err(message)?
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn message(e: impl ToString) -> CursorError {
    CursorError::Message(e.to_string())
}

fn do_load_cursor(
    conn: &Connection,
    processor: &str,
    stream: &StreamRef,
) -> Result<StreamCursor, CursorError> {
    let row = conn
        .query_row(
            "SELECT seq, version, updated_at FROM projection_cursors
             WHERE projection_name = ?1 AND tenant_id = ?2 AND session_id = ?3",
            rusqlite::params![processor, stream.tenant_id, stream.session_id],
            |row| {
                Ok((
                    row.get::<_, u64>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(message)?;

    match row {
        Some((seq, version, updated_at)) => Ok(StreamCursor {
            seq: Seq(seq),
            version,
            updated_at: parse_dt(&updated_at).unwrap_or_else(Utc::now),
        }),
        None => Ok(StreamCursor {
            seq: Seq(0),
            version: 0,
            updated_at: Utc::now(),
        }),
    }
}

fn do_compare_and_set(
    conn: &mut Connection,
    processor: &str,
    stream: &StreamRef,
    expected_version: u64,
    new_seq: Seq,
    owner_id: Option<&str>,
) -> Result<bool, CursorError> {
    let tx = conn.transaction().map_err(message)?;

    if expected_version == 0 {
        tx.execute(
            "INSERT OR IGNORE INTO projection_cursors
                 (projection_name, tenant_id, session_id, seq, version, owner_id, updated_at)
             VALUES (?1, ?2, ?3, 0, 0, NULL, ?4)",
            rusqlite::params![
                processor,
                stream.tenant_id,
                stream.session_id,
                Utc::now().to_rfc3339()
            ],
        )
        .map_err(message)?;
    }

    let updated = tx
        .execute(
            "UPDATE projection_cursors
             SET seq = ?1,
                 version = version + 1,
                 owner_id = ?2,
                 updated_at = ?3
             WHERE projection_name = ?4
               AND tenant_id = ?5
               AND session_id = ?6
               AND version = ?7",
            rusqlite::params![
                new_seq.0,
                owner_id,
                Utc::now().to_rfc3339(),
                processor,
                stream.tenant_id,
                stream.session_id,
                expected_version,
            ],
        )
        .map_err(message)?;

    tx.commit().map_err(message)?;

    Ok(updated == 1)
}

fn do_pending_streams(
    conn: &Connection,
    processor: &str,
    shard_id: u32,
    shard_count: u32,
    limit: usize,
) -> Result<Vec<StreamRef>, CursorError> {
    let mut stmt = conn
        .prepare(
            "SELECT s.tenant_id, s.session_id
             FROM snapshots s
             LEFT JOIN projection_cursors c
               ON c.projection_name = ?1
              AND c.tenant_id = s.tenant_id
              AND c.session_id = s.session_id
             WHERE s.seq > COALESCE(c.seq, 0)
               AND (?2 = 1 OR s.shard_key % ?2 = ?3)
             LIMIT ?4",
        )
        .map_err(message)?;

    let rows = stmt
        .query_map(
            rusqlite::params![processor, shard_count, shard_id, limit as i64],
            |row| {
                Ok(StreamRef {
                    tenant_id: row.get(0)?,
                    session_id: row.get(1)?,
                })
            },
        )
        .map_err(message)?;

    rows.map(|row| row.map_err(message)).collect()
}

fn do_seed_at_tail(conn: &mut Connection, processor: &str) -> Result<(), CursorError> {
    let tx = conn.transaction().map_err(message)?;
    let now = Utc::now().to_rfc3339();

    // Claiming the seed row is what makes this once-only: a second call must
    // not rewind streams the processor has already read past.
    let claimed = tx
        .execute(
            "INSERT OR IGNORE INTO projection_seeds (projection_name, created_at) VALUES (?1, ?2)",
            rusqlite::params![processor, now],
        )
        .map_err(message)?;

    if claimed == 1 {
        tx.execute(
            "INSERT OR IGNORE INTO projection_cursors
                 (projection_name, tenant_id, session_id, seq, version, owner_id, updated_at)
             SELECT ?1, tenant_id, session_id, seq, 0, NULL, ?2 FROM snapshots",
            rusqlite::params![processor, now],
        )
        .map_err(message)?;
    }

    tx.commit().map_err(message)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;
    use crate::providers::sqlite::event_store::SqliteEventStore;
    use crate::shard::shard_key;

    /// A cursor store over a db that also carries the event store's schema —
    /// `pending_streams` reads the stream heads it writes.
    fn temp_store() -> (SqliteCursorStore, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("core-cursor-{}.db", Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        SqliteEventStore::new(db.clone()).unwrap();
        (SqliteCursorStore::new(db.clone()).unwrap(), path)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    /// Park a stream head without going through the event store.
    fn head(store: &SqliteCursorStore, session_id: &str, seq: u64) {
        store
            .db
            .writer
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO snapshots (tenant_id, session_id, seq, shard_key, data)
                 VALUES ('t1', ?1, ?2, ?3, '{}')
                 ON CONFLICT(tenant_id, session_id) DO UPDATE SET seq = excluded.seq",
                rusqlite::params![session_id, seq, shard_key(session_id) as i64],
            )
            .unwrap();
    }

    fn stream(session_id: &str) -> StreamRef {
        StreamRef {
            tenant_id: "t1".to_string(),
            session_id: session_id.to_string(),
        }
    }

    async fn pending(store: &SqliteCursorStore) -> Vec<String> {
        let mut ids: Vec<String> = store
            .pending_streams("p1", 0, 1, 100)
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.session_id)
            .collect();
        ids.sort();
        ids
    }

    #[tokio::test]
    async fn a_stream_is_pending_until_its_cursor_reaches_the_head() {
        let (store, path) = temp_store();
        head(&store, "s1", 3);
        head(&store, "s2", 1);

        assert_eq!(pending(&store).await, vec!["s1", "s2"]);

        assert!(store
            .compare_and_set_cursor("p1", &stream("s1"), 0, Seq(3), None)
            .await
            .unwrap());
        assert_eq!(pending(&store).await, vec!["s2"]);

        // New events past the cursor put the stream back on the work list.
        head(&store, "s1", 5);
        assert_eq!(pending(&store).await, vec!["s1", "s2"]);
        cleanup(&path);
    }

    /// Each processor reads at its own pace: one catching up leaves the other's
    /// work list alone.
    #[tokio::test]
    async fn cursors_are_per_processor() {
        let (store, path) = temp_store();
        head(&store, "s1", 2);

        store
            .compare_and_set_cursor("p1", &stream("s1"), 0, Seq(2), None)
            .await
            .unwrap();

        assert!(pending(&store).await.is_empty());
        assert_eq!(
            store.pending_streams("p2", 0, 1, 100).await.unwrap().len(),
            1
        );
        cleanup(&path);
    }

    #[tokio::test]
    async fn a_stale_version_loses_the_cas() {
        let (store, path) = temp_store();
        head(&store, "s1", 4);

        assert!(store
            .compare_and_set_cursor("p1", &stream("s1"), 0, Seq(2), Some("a"))
            .await
            .unwrap());

        let cursor = store.load_cursor("p1", &stream("s1")).await.unwrap();
        assert_eq!(cursor.seq, Seq(2));
        assert_eq!(cursor.version, 1);

        // The other owner is still holding version 0.
        assert!(!store
            .compare_and_set_cursor("p1", &stream("s1"), 0, Seq(4), Some("b"))
            .await
            .unwrap());
        assert_eq!(
            store.load_cursor("p1", &stream("s1")).await.unwrap().seq,
            Seq(2)
        );
        cleanup(&path);
    }

    #[tokio::test]
    async fn seeding_at_tail_skips_history_once() {
        let (store, path) = temp_store();
        head(&store, "s1", 7);

        store.seed_at_tail("p1").await.unwrap();
        assert!(pending(&store).await.is_empty());
        assert_eq!(
            store.load_cursor("p1", &stream("s1")).await.unwrap().seq,
            Seq(7)
        );

        // A stream born after the seed still starts from its first event.
        head(&store, "s2", 1);
        assert_eq!(pending(&store).await, vec!["s2"]);

        // A restart must not rewind a stream already read past.
        store
            .compare_and_set_cursor("p1", &stream("s2"), 0, Seq(1), None)
            .await
            .unwrap();
        head(&store, "s2", 4);
        store.seed_at_tail("p1").await.unwrap();
        assert_eq!(
            store.load_cursor("p1", &stream("s2")).await.unwrap().seq,
            Seq(1)
        );
        cleanup(&path);
    }

    #[tokio::test]
    async fn a_shard_only_claims_its_own_streams() {
        let (store, path) = temp_store();
        let ids: Vec<String> = (0..16).map(|i| format!("s{i}")).collect();
        for id in &ids {
            head(&store, id, 1);
        }

        let mut claimed: Vec<String> = Vec::new();
        for shard_id in 0..4 {
            let streams = store.pending_streams("p1", shard_id, 4, 100).await.unwrap();
            for s in &streams {
                assert_eq!(shard_key(&s.session_id) % 4, u64::from(shard_id));
            }
            claimed.extend(streams.into_iter().map(|s| s.session_id));
        }

        claimed.sort();
        let mut expected = ids;
        expected.sort();
        assert_eq!(claimed, expected, "every stream lands in exactly one shard");
        cleanup(&path);
    }
}
